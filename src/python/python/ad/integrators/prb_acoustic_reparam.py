from __future__ import annotations # Delayed parsing of type annotations
from typing import Optional, Tuple, Callable, Any, Union

import drjit as dr
import mitsuba as mi
import gc

from .prb_acoustic import PRBAcousticIntegrator
from .common import RBIntegrator, _ReparamWrapper, mis_weight

class PRBAcousticReparamIntegrator(PRBAcousticIntegrator):
    def __init__(self, props):
        super().__init__(props)

        # The reparameterization is computed stochastically and removes
        # gradient bias at the cost of additional variance. Use this parameter
        # to disable the reparameterization after a certain path depth to
        # control this tradeoff. A value of zero disables it completely.
        self.reparam_max_depth = props.get('reparam_max_depth', self.max_depth)

        # Specifies the number of auxiliary rays used to evaluate the
        # reparameterization
        self.reparam_rays = props.get('reparam_rays', 16)

        # Specifies the von Mises Fisher distribution parameter for sampling
        # auxiliary rays in Bangaru et al.'s [2020] parameterization
        self.reparam_kappa = props.get('reparam_kappa', 1e5)

        # Harmonic weight exponent in Bangaru et al.'s [2020] parameterization
        self.reparam_exp = props.get('reparam_exp', 3.0)

        # Enable antithetic sampling in the reparameterization?
        self.reparam_antithetic = props.get('reparam_antithetic', True)

        # Unroll the loop tracing auxiliary rays in the reparameterization?
        self.reparam_unroll = props.get('reparam_unroll', False)


    def reparam(self,
                scene: mi.Scene,
                rng: mi.PCG32,
                params: Any,
                ray: mi.Ray3f,
                depth: mi.UInt32,
                active: mi.Bool):
        """
        Helper function to reparameterize rays internally and within ADIntegrator
        """

        # Potentially disable the reparameterization completely
        if self.reparam_max_depth == 0:
            return dr.detach(ray.d, True), mi.Float(1)

        active = active & (depth < self.reparam_max_depth)
        return mi.ad.reparameterize_ray(scene, rng, params, ray,
                                        num_rays=self.reparam_rays,
                                        kappa=self.reparam_kappa,
                                        exponent=self.reparam_exp,
                                        antithetic=self.reparam_antithetic,
                                        unroll=self.reparam_unroll,
                                        active=active)

    def sample(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               sensor: mi.Sensor,
               ray: mi.Ray3f,
               block: mi.ImageBlock,
               δH: Optional[mi.ImageBlock],
               state_in_δHL: Optional[mi.Spectrum],
               state_in_δHdT: Optional[mi.Spectrum],
               active: mi.Bool,
               reparam: Optional[
                   Callable[[mi.Ray3f, mi.Bool], Tuple[mi.Ray3f, mi.Float]]],
               **_ # Absorbs unused arguments
    ) -> Tuple[mi.Spectrum, mi.Bool, mi.Spectrum]:

        # Rendering a primal image? (vs performing forward/reverse-mode AD)
        prb_mode = δH is not None
        primal = mode == dr.ADMode.Primal
        adjoint = prb_mode and (mode == dr.ADMode.Backward or mode == dr.ADMode.Forward)
        assert primal or prb_mode

        # Standard BSDF evaluation context for path tracing
        bsdf_ctx = mi.BSDFContext()

        # --------------------- Configure loop state ----------------------

        distance     = mi.Float(0.0)
        max_distance = self.max_time * self.speed_of_sound

        # Copy input arguments to avoid mutating the caller's state
        depth = mi.UInt32(0)                               # Depth of current vertex
        δHL  = mi.Spectrum(0) if primal else state_in_δHL  # Integral of grad_in * Radiance (accumulated)
        δHdT = mi.Spectrum(0) if primal else state_in_δHdT # Integral of grad_in * (Radiance derived wrt time) (accumulated)
        β = mi.Spectrum(1)                                 # Path throughput weight
        η = mi.Float(1)                                    # Index of refraction
        mis_em = mi.Float(1)                               # Emitter MIS weight
        active = mi.Bool(active)                           # Active SIMD lanes

        if self.skip_direct:
            mis_em = mi.Float(0.)

        # Initialize loop state variables caching the rays and preliminary
        # intersections of the previous (zero-initialized) and current vertex
        ray_prev = dr.zeros(mi.Ray3f)
        ray_cur  = mi.Ray3f(dr.detach(ray))
        pi_prev  = dr.zeros(mi.PreliminaryIntersection3f)
        pi_cur   = scene.ray_intersect_preliminary(ray_cur, coherent=True,
                                                   active=active)

        # Record the following loop in its entirety
        loop = mi.Loop(name="PRB Acoustic (%s)" % mode.name,
                       state=lambda: (distance, #block.tensor(),
                                      sampler, depth, δHL, δHdT, #δL.tensor(),
                                      β, η, mis_em, active,
                                      ray_prev, ray_cur, pi_prev, pi_cur, reparam))

        # Specify the max. number of loop iterations (this can help avoid
        # costly synchronization when when wavefront-style loops are generated)
        loop.set_max_iterations(self.max_depth)

        if adjoint:
            trafo = mi.Transform4f(sensor.world_transform())

        while loop(active):
            active_next = mi.Bool(active)

            # The first path vertex requires some special handling (see below)
            first_vertex = dr.eq(depth, 0)

            # Reparameterized ray (a copy of 'ray_cur' in primal mode)
            ray_reparam = mi.Ray3f(ray_cur)

            weight = mi.Float(1.)
            # Jacobian determinant of the parameterization (1 in primal mode)
            ray_reparam_det = 1

            if adjoint:
                with dr.resume_grad():
                    # Compute a surface interaction of the previous vertex with
                    # derivative tracking (no-op if there is no prev. vertex)
                    si_prev = pi_prev.compute_surface_interaction(
                        ray_prev, mi.RayFlags.All | mi.RayFlags.FollowShape)

                    # Adjust the ray origin of 'ray_cur' so that it follows the
                    # previous shape, then pass this information to 'reparam'
                    ray_reparam.d, ray_reparam_det = reparam(
                        dr.select(first_vertex, ray_cur,
                                  si_prev.spawn_ray(ray_cur.d)), depth)

                    # only relevant for first ray and omnidirectional microphones
                    # UF: I checked: weight is equal to sampled weight (sample_rays).
                    weight[first_vertex] = mi.Spectrum(mi.warp.square_to_von_mises_fisher_pdf(trafo.inverse() @ ray_reparam.d, self.kappa)).x
                    # ray_reparam_det[first_vertex] = 1

                    # Finally, disable all derivatives in 'si_prev', as we are
                    # only interested in tracking derivatives related to the
                    # current interaction in the remainder of this function
                    dr.disable_grad(si_prev)

            # Compute a surface interaction that tracks derivatives arising
            # from differentiable shape parameters (position, normals, etc.)
            # In primal mode, this is just an ordinary ray tracing operation.
            with dr.resume_grad(when=not primal):
                si_cur = pi_cur.compute_surface_interaction(ray_reparam)

            distance += si_cur.t

            # Get the BSDF, potentially computes texture-space differentials
            bsdf = si_cur.bsdf(ray_cur)

            # ---------------------- Direct emission ----------------------

            # Hide the environment emitter if necessary
            if self.hide_emitters:
                active_next &= ~(dr.eq(depth, 0) & ~si_cur.is_valid())

            # Intensity of current emitter weighted by importance (def. by prev bsdf hits) and MIS weight
            # Evaluate the emitter (with derivative tracking if requested)
            with dr.resume_grad(when=not primal):
                emitter = si_cur.emitter(scene)
                Le = β * mis_em * emitter.eval(si_cur, active_next)

            # ---------------------- Emitter sampling ----------------------

            # Should we continue tracing to reach one more vertex?
            active_next &= (depth + 1 < self.max_depth) & si_cur.is_valid()

            # Get the BSDF, potentially computes texture-space differentials.
            bsdf_cur = si_cur.bsdf(ray_cur)

            # Is emitter sampling even possible on the current vertex?
            active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

            # If so, randomly sample an emitter without derivative tracking.
            ds, em_weight = scene.sample_emitter_direction(si_cur, sampler.next_2d(), True, active_em)
            active_em &= dr.neq(ds.pdf, 0.0)

            with dr.resume_grad(when=not primal):
                em_ray_det = 1

                if adjoint:
                    # Create a surface interaction that follows the shape's
                    # motion while ignoring any reparameterization from the
                    # previous ray. This ensures the subsequent reparameterization
                    # accounts for occluder's motion relatively to the current shape.
                    si_cur_follow = pi_cur.compute_surface_interaction(
                        ray_cur, mi.RayFlags.All | mi.RayFlags.FollowShape)

                    # Reparameterize the ray towards the emitter
                    em_ray = si_cur_follow.spawn_ray_to(ds.p)
                    em_ray.d, em_ray_det = reparam(em_ray, depth + 1, active_em)
                    ds.d = em_ray.d

                    # Given the detached emitter sample, *recompute* its
                    # contribution with AD to enable light source optimization
                    em_val = scene.eval_emitter_direction(si_cur, ds, active_em)
                    em_weight = dr.select(dr.neq(ds.pdf, 0), em_val / ds.pdf, 0)
                    dr.disable_grad(ds.d)

                # Evaluate BSDF * cos(theta) differentiably
                wo = si_cur.to_local(ds.d)
                bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si_cur,
                                                               wo, active_em)
                mis_direct = dr.select(ds.delta, 1, mis_weight(ds.pdf, bsdf_pdf_em))
                Lr_dir = β * mis_direct * bsdf_value_em * em_weight * em_ray_det

            # ------------------ Detached BSDF sampling -------------------

            bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si_cur,
                                                   sampler.next_1d(),
                                                   sampler.next_2d(),
                                                   active_next)

            bsdf_sample_delta = mi.has_flag(bsdf_sample.sampled_type,
                                            mi.BSDFFlags.Delta)

            # ---- PRB-style tracking of time derivatives -----
            # TODO (MW): Move to function of `prb_acoustic`?

            δHdTt_cur = mi.Float(0.)
            if prb_mode and self.track_time_derivatives:
                # This is executed in the PRB primal and adjoint passes
                active_time      = active & si_cur.is_valid()
                active_time_next = active_em
                with dr.resume_grad():
                    # The surface interaction can be invalid, in which case we don't want it to have any influence
                    T     = dr.select(active_time,             # Full distance of current path
                                    dr.detach(distance), 0.)
                    T_dir = dr.select(active_time_next,        # Full distance of direct emitter path
                                      dr.detach(distance + dr.norm(ds.p - si_cur.p)), 0.)
                    dr.enable_grad(T, T_dir)

                    Le_pos     = mi.Point2f(ray.wavelengths.x,
                                            block.size().y * T / max_distance)
                    Lr_dir_pos = mi.Point2f(ray.wavelengths.x,
                                            block.size().y * T_dir / max_distance)

                    δHLe     = dr.detach(Le)     * δH.read(pos=Le_pos)[0]
                    δHLr_dir = dr.detach(Lr_dir) * δH.read(pos=Lr_dir_pos)[0]

                    dr.forward_from(T)
                    dr.forward_from(T_dir)

                    δHdTLe = dr.detach(dr.grad(δHLe))
                    δHdTLr_dir = dr.detach(dr.grad(δHLr_dir))

                if primal:
                    # PRB primal
                    δHdT = δHdT + δHdTLe + δHdTLr_dir
                elif adjoint:
                    # PRB adjoint (backward)
                    with dr.resume_grad():
                        # The surface interaction can be invalid, in which case we don't want it to have any influence
                        t_cur     = dr.select(active_time,      si_cur.t,                 0.)
                        #t_cur_dir = dr.select(active_time_next, dr.norm(em_ray.o - si_cur.p), 0.)
                        t_cur_dir = dr.select(active_time_next, dr.norm(ds.p - si_cur.p), 0.)

                        # attention: the second summand accounts for the direct light segment!
                        δHdTt_cur = t_cur * δHdT + t_cur_dir * δHdTLr_dir
                    δHdT_cur = δHdT
                    δHdT = δHdT - δHdTLe - δHdTLr_dir

            # put and accumulate current (differential) radiance

            Le_pos     = mi.Point2f(ray.wavelengths.x,
                                    block.size().y * distance / max_distance)
            Lr_dir_pos = mi.Point2f(ray.wavelengths.x,
                                    block.size().y * (distance + dr.norm(ds.p - si_cur.p)) / max_distance)
            if prb_mode:
                # backward_from(δHLx) is the same as splatting_and_backward_gradient_image but we can store it this way
                with dr.resume_grad(when=not primal):
                    δHLe     = Le     * δH.read(pos=Le_pos)[0]
                    δHLr_dir = Lr_dir * δH.read(pos=Lr_dir_pos)[0]
                δHL_prev = δHL
                if primal:
                    δHL = δHL + δHLe + δHLr_dir
                else: # adjoint:
                    δHL = δHL - δHLe - δHLr_dir
            else: # primal
                block.put(pos=Le_pos,     values=mi.Vector2f(Le.x,     1.0), active=(Le.x     > 0.))
                block.put(pos=Lr_dir_pos, values=mi.Vector2f(Lr_dir.x, 1.0), active=(Lr_dir.x > 0.))

            # ---- Update loop variables based on current interaction -----

            η     *= bsdf_sample.eta
            β     *= bsdf_weight
            # the rest will be updated at the end of the loop

            # -------------------- Stopping criterion ---------------------

            # Don't run another iteration if the throughput has reached zero
            β_max = dr.max(β)
            active_next &= dr.neq(β_max, 0)
            active_next &= distance <= max_distance

            # Russian roulette stopping probability (must cancel out ior^2
            # to obtain unitless throughput, enforces a minimum probability)
            rr_prob = dr.minimum(β_max * η**2, .95)

            # Apply only further along the path since, this introduces variance
            rr_active = depth >= self.rr_depth
            β[rr_active] *= dr.rcp(rr_prob)
            rr_continue = sampler.next_1d() < rr_prob
            active_next &= ~rr_active | rr_continue

            # ------------------ Intersect next surface -------------------

            ray_next = si_cur.spawn_ray(si_cur.to_world(bsdf_sample.wo))
            pi_next = scene.ray_intersect_preliminary(ray_next,
                                                      active=active_next)

            # Compute a detached intersection record for the next vertex
            si_next = pi_next.compute_surface_interaction(ray_next)

            # ---------- Compute MIS weight for the next vertex -----------

            ds = mi.DirectionSample3f(scene, si=si_next, ref=si_cur)

            # Probability of sampling 'si_next' using emitter sampling
            # (set to zero if the BSDF doesn't have any smooth components)
            pdf_em = scene.pdf_emitter_direction(
                ref=si_cur, ds=ds, active=~bsdf_sample_delta
            )

            mis_em = mis_weight(bsdf_sample.pdf, pdf_em)

            # ------------------ Differential phase only ------------------

            if adjoint:
                # Clone the sampler to run ahead in the random number sequence
                # without affecting the PRB random walk
                sampler_clone = sampler.clone()

                # 'active_next' value at the next vertex
                active_next_next = active_next & si_next.is_valid() & \
                    (depth + 2 < self.max_depth)

                # Retrieve the BSDFs of the two adjacent vertices
                bsdf_next = si_next.bsdf(ray_next)
                bsdf_prev = si_prev.bsdf(ray_prev)

                # Check if emitter sampling is possible at the next vertex
                active_em_next = active_next_next & mi.has_flag(bsdf_next.flags(),
                                                                mi.BSDFFlags.Smooth)

                # If so, randomly sample an emitter without derivative tracking.
                ds_next, em_weight_next = scene.sample_emitter_direction(
                    si_next, sampler_clone.next_2d(), True, active_em_next)
                active_em_next &= dr.neq(ds_next.pdf, 0.0)

                # Compute the emission sampling contribution at the next vertex
                bsdf_value_em_next, bsdf_pdf_em_next = bsdf_next.eval_pdf(
                    bsdf_ctx, si_next, si_next.to_local(ds_next.d), active_em_next)

                mis_direct_next = dr.select(ds_next.delta, 1,
                                            mis_weight(ds_next.pdf, bsdf_pdf_em_next))
                Lr_dir_next = β * mis_direct_next * bsdf_value_em_next * em_weight_next

                # Generate a detached BSDF sample at the next vertex
                bsdf_sample_next, bsdf_weight_next = bsdf_next.sample(
                    bsdf_ctx, si_next, sampler_clone.next_1d(),
                    sampler_clone.next_2d(), active_next_next
                )

                dist_next_dir = distance + dr.norm(si_next.p - si_cur.p) + dr.norm(ds_next.p - si_next.p)

                Lr_dir_next_pos = mi.Point2f(ray.wavelengths.x,
                                        block.size().y *
                                        dist_next_dir / max_distance)

                δHLr_dir_next = Lr_dir_next * δH.read(pos=Lr_dir_next_pos)[0]

                # Account for adjacent vertices, but only consider derivatives
                # that arise from the reparameterization at 'si_cur.p'
                with dr.resume_grad(ray_reparam):
                    # Compute a surface interaction that only tracks derivatives
                    # that arise from the reparameterization.
                    si_cur_reparam_only = pi_cur.compute_surface_interaction(
                        ray_reparam, mi.RayFlags.All | mi.RayFlags.DetachShape)

                    # Differentiably recompute the outgoing direction at 'prev'
                    # and the incident direction at 'next'
                    wo_prev = dr.normalize(si_cur_reparam_only.p - si_prev.p)
                    wi_next = dr.normalize(si_cur_reparam_only.p - si_next.p)

                    # Compute the emission at the next vertex
                    si_next.wi = si_next.to_local(wi_next)
                    Le_next = β * mis_em * \
                        si_next.emitter(scene).eval(si_next, active_next)

                    dist_next = distance - dr.detach(si_cur.t) + dr.norm(si_cur_reparam_only.p - si_prev.p) + dr.norm(si_next.p - si_cur_reparam_only.p)

                    Le_next_pos = mi.Point2f(ray.wavelengths.x,
                        block.size().y * dist_next / max_distance)

                    δHLe_next = Le_next * δH.read(pos=Le_next_pos)[0]


                    # Value of 'δHL' at the next vertex
                    δHL_next = δHL - dr.detach(δHLe_next) - dr.detach(δHLr_dir_next)


                    # Account for the BSDF of the previous and next vertices
                    bsdf_val_prev = bsdf_prev.eval(bsdf_ctx, si_prev,
                                                   si_prev.to_local(wo_prev))
                    bsdf_val_next = bsdf_next.eval(bsdf_ctx, si_next,
                                                   bsdf_sample_next.wo)

                    extra = mi.Spectrum(δHLe_next)
                    extra[~first_vertex]      += δHL_prev * bsdf_val_prev / dr.maximum(1e-8, dr.detach(bsdf_val_prev))
                    extra[si_next.is_valid()] += δHL_next * bsdf_val_next / dr.maximum(1e-8, dr.detach(bsdf_val_next))

                    #dist_cur = distance - dr.detach(si_cur.t) + dr.norm(si_cur_reparam_only.p - ray_cur.o)
                    #extra += dist_cur * δHdT_cur

                    #T_next = dr.detach(dist_next)
                    #dr.enable_grad(T_next)
                    #pos = mi.Point2f(ray.wavelengths.x,
                    #    block.size().y * T_next / max_distance)
                    #δHLe_next   = dr.detach(Le_next) * δH.read(pos=pos)[0]
                    #dr.forward_from(T_next)
                    #δHdTLe_next = dr.detach(dr.grad(δHLe_next))
                    #extra[si_next.is_valid()] += dist_next * (δHdT-δHdTLe_next)

                    # extra[si_next.is_valid()] += δHLr_dir_next
                    # extra[si_next.is_valid()] += δHL_prev

                with dr.resume_grad():
                    # Recompute 'wo' to propagate derivatives to cosine term
                    wo = si_cur.to_local(ray_next.d)

                    # Re-evaluate BSDF * cos(theta) differentiably
                    bsdf_val = bsdf_cur.eval(bsdf_ctx, si_cur, wo, active_next)

                    # Detached version of the above term and inverse
                    bsdf_val_detach = bsdf_weight * bsdf_sample.pdf
                    inv_bsdf_val_detach = dr.select(dr.neq(bsdf_val_detach, 0),
                                                    dr.rcp(bsdf_val_detach), 0)

                    # Differentiable version of the reflected indirect radiance
                    δHLr_ind = δHL * dr.replace_grad(1, inv_bsdf_val_detach * bsdf_val)

                with dr.resume_grad():
                    # Differentiable Monte Carlo estimate of all contributions
                    δHLo = (δHLe + δHLr_dir + δHLr_ind) * ray_reparam_det

                    if dr.flag(dr.JitFlag.VCallRecord) and not dr.grad_enabled(δHLo):
                        raise Exception(
                            "The contribution computed by the differential "
                            "rendering phase is not attached to the AD graph! "
                            "Raising an exception since this is usually "
                            "indicative of a bug (for example, you may have "
                            "forgotten to call dr.enable_grad(..) on one of "
                            "the scene parameters, or you may be trying to "
                            "optimize a parameter that does not generate "
                            "derivatives in detached PRB.)")

                    # Propagate derivatives
                    if mode == dr.ADMode.Backward:
                        first = dr.detach(δHLo)*weight/dr.detach(weight)
                        vMFcontrib = dr.select(first_vertex, first, mi.Float(0.))

                        # as ray_reparam_det is 1 we do not need to multiply everything with its detached version
                        # as weight/dr.detach(weight) is 1 we do not need to multiply everything with its detached version
                        dr.backward_from(δHLo + δHdTt_cur + extra + vMFcontrib)

            # Differential phases need access to the previous interaction, too
            if adjoint:
                pi_prev  = pi_cur
                ray_prev = ray_cur

            # Provide ray/interaction to the next iteration
            pi_cur   = pi_next
            ray_cur  = ray_next

            depth[si_cur.is_valid()] += 1
            active = active_next

        return (
            δHL,                   # Radiance/differential radiance
            dr.neq(depth, 0),      # Ray validity flag for alpha blending
            δHdT                   # State for the differential phase
        )


    def to_string(self):
        return f'{type(self).__name__}[max_depth = {self.max_depth},' \
               f' rr_depth = { self.rr_depth },' \
               f' reparam_rays = { self.reparam_rays }]'

mi.register_integrator("prb_acoustic_reparam", lambda props: PRBAcousticReparamIntegrator(props))
