from __future__ import annotations # Delayed parsing of type annotations
from typing import Optional, Tuple, Callable, Any

import drjit as dr
import mitsuba as mi

from .common import mis_weight
from .prb_acoustic import PRBAcousticIntegrator

class PRBReparamAcousticIntegrator(PRBAcousticIntegrator):

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
        self.reparam_antithetic = props.get('reparam_antithetic', False)

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
               ray: mi.Ray3f,
               block: mi.ImageBlock,
               δL: Optional[mi.ImageBlock],
               state_in_δL: Optional[mi.Spectrum],
               state_in_δLdG: Optional[mi.Spectrum],
               reparam: Optional[
                   Callable[[mi.Ray3f, mi.Bool], Tuple[mi.Ray3f, mi.Float]]],
               active: mi.Bool,
               **_ # Absorbs unused arguments
    ) -> Tuple[mi.Spectrum, mi.Bool, mi.Spectrum]:

        # Rendering a primal image? (vs performing forward/reverse-mode AD)
        primal = mode == dr.ADMode.Primal
        assert primal or (δL is not None and state_in_δL is not None and state_in_δLdG is not None)

        # Standard BSDF evaluation context for path tracing
        bsdf_ctx = mi.BSDFContext()

        # --------------------- Configure loop state ----------------------

        distance     = mi.Float(0.0)
        max_distance = self.max_time * self.speed_of_sound

        # Copy input arguments to avoid mutating the caller's state
        depth = mi.UInt32(0)                               # Depth of current vertex
        L    = mi.Spectrum(0 if primal else state_in_δL)   # Radiance accumulator
        δLdG = mi.Spectrum(0 if primal else state_in_δLdG) # Radiance*Gaussian accumulator
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
        # TODO: loop should keep track of imageblock and δL
        loop = mi.Loop(name="PRB Reparam Acoustic (%s)" % mode.name,
                       state=lambda: (distance, #block.tensor(),
                                      sampler, depth, L, #δL.tensor(),
                                      β, η, mis_em, active,
                                      ray_prev, ray_cur, pi_prev, pi_cur, reparam))

        # Specify the max. number of loop iterations (this can help avoid
        # costly synchronization when wavefront-style loops are generated)
        loop.set_max_iterations(self.max_depth)

        while loop(active):
            active_next = mi.Bool(active)

            # The first path vertex requires some special handling (see below)
            first_vertex = dr.eq(depth, 0)

            # Reparameterized ray (a copy of 'ray_cur' in primal mode)
            ray_reparam = mi.Ray3f(ray_cur)

            # Jacobian determinant of the parameterization (1 in primal mode)
            ray_reparam_det = 1

            # ----------- Reparameterize (differential phase only) -----------

            if not primal:
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
                    ray_reparam_det[first_vertex] = 1

                    # Finally, disable all derivatives in 'si_prev', as we are
                    # only interested in tracking derivatives related to the
                    # current interaction in the remainder of this function
                    dr.disable_grad(si_prev)

            # ------ Compute detailed record of the current interaction ------

            # Compute a surface interaction that potentially tracks derivatives
            # due to differentiable shape parameters (position, normals, etc.)

            with dr.resume_grad(when=not primal):
                si_cur = pi_cur.compute_surface_interaction(ray_reparam)

            distance += si_cur.t

            # ---------------------- Direct emission ----------------------

            # Hide the environment emitter if necessary
            if self.hide_emitters:
                active_next &= ~(dr.eq(depth, 0) & ~si_cur.is_valid())

            # Evaluate the emitter (with derivative tracking if requested)
            with dr.resume_grad(when=not primal):
                emitter = si_cur.emitter(scene)
                Le = β * mis_em * emitter.eval(si_cur, active_next)

            # ----------------------- Emitter sampling -----------------------

            # Should we continue tracing to reach one more vertex?
            active_next &= (depth + 1 < self.max_depth) & si_cur.is_valid()

            # Get the BSDF, potentially computes texture-space differentials.
            bsdf_cur = si_cur.bsdf(ray_cur)

            # Is emitter sampling even possible on the current vertex?
            active_em = active_next & mi.has_flag(bsdf_cur.flags(), mi.BSDFFlags.Smooth)

            # If so, randomly sample an emitter without derivative tracking.
            ds, em_weight = scene.sample_emitter_direction(
                si_cur, sampler.next_2d(), True, active_em)
            active_em &= dr.neq(ds.pdf, 0.0)

            with dr.resume_grad(when=not primal):
                em_ray_det = 1

                if not primal:
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
                bsdf_value_em, bsdf_pdf_em = bsdf_cur.eval_pdf(bsdf_ctx, si_cur,
                                                               wo, active_em)
                mis_direct = dr.select(ds.delta, 1, mis_weight(ds.pdf, bsdf_pdf_em))
                Lr_dir = β * mis_direct * bsdf_value_em * em_weight * em_ray_det

            # ------------------ Detached BSDF sampling -------------------

            # Perform detached BSDF sampling.
            bsdf_sample, bsdf_weight = bsdf_cur.sample(bsdf_ctx, si_cur,
                                                       sampler.next_1d(),
                                                       sampler.next_2d(),
                                                       active_next)

            bsdf_sample_delta = mi.has_flag(bsdf_sample.sampled_type,
                                            mi.BSDFFlags.Delta)

            # ---- Update loop variables based on current interaction -----

            η     *= bsdf_sample.eta
            β     *= bsdf_weight
            L_prev = L  # Value of 'L' at previous vertex

            # ---- PRB-style tracking of time derivatives -----

            active_time      = active & si_cur.is_valid()
            active_time_next = active_em
            δLdG_Le     = mi.Float(0.)
            δLdG_Lr_dir = mi.Float(0.)
            if δL is not None:
                # This is executed in the PRB primal and adjoint passes
                with dr.resume_grad():
                    # The surface interaction can be invalid, in which case we don't want it to have any influence
                    T     = dr.select(active_time,             # Full distance of current path
                                      dr.detach(distance), 0.)
                    T_dir = dr.select(active_time_next,        # Full distance of direct emitter path
                                      dr.detach(distance + dr.norm(ds.p - si_cur.p)), 0.) 
                    dr.enable_grad(T, T_dir)

                    Le_pos     = mi.Point2f(ray.wavelengths.x - mi.Float(1.0),
                                            block.size().y * T / max_distance)
                    Lr_dir_pos = mi.Point2f(ray.wavelengths.x - mi.Float(1.0),
                                            block.size().y * T_dir / max_distance)
                    
                    δL_Le     = δL.read(pos=Le_pos)[0]
                    δL_Lr_dir = δL.read(pos=Lr_dir_pos)[0]

                    dr.forward_from(T)
                    dr.forward_from(T_dir)

                    δLdG_Le     = dr.detach(dr.grad(δL_Le))
                    δLdG_Lr_dir = dr.detach(dr.grad(δL_Lr_dir))

                # TODO (MW): Verify the multiplication with energy (should be 1 here, luckily)
                # TODO (MW): How to track changes of the emitter radiance?
                δLdG_Le     = dr.detach(Le)     * δLdG_Le
                δLdG_Lr_dir = dr.detach(Lr_dir) * δLdG_Lr_dir

            if primal:
                # PRB primal 
                δLdG = δLdG + δLdG_Le + δLdG_Lr_dir
            elif mode == dr.ADMode.Backward:
                # PRB adjoint (backward)
                with dr.resume_grad():
                    # The surface interaction can be invalid, in which case we don't want it to have any influence
                    t0     = dr.select(active_time,      si_cur.t,                 0.)
                    t0_dir = dr.select(active_time_next, dr.norm(ds.p - si_cur.p), 0.)

                    # TODO (MW): why not -t0 ...?
                    dr.backward_from(t0     * δLdG)
                    dr.backward_from(t0_dir * δLdG_Lr_dir) # <- attention, this accounts for the direct light segment!
                δLdG = δLdG - δL_Le - δL_Lr_dir

            # put and accumulate current (differential) radiance

            Le_pos     = mi.Point2f(ray.wavelengths.x - mi.Float(1.0),
                                    block.size().y * distance / max_distance)
            Lr_dir_pos = mi.Point2f(ray.wavelengths.x - mi.Float(1.0),
                                    block.size().y * (distance + dr.norm(ds.p - si_cur.p)) / max_distance)

            if mode == dr.ADMode.Forward:
                δL.put(pos=Le_pos,     values=mi.Vector2f((L * Le    ).x, 1.0))
                δL.put(pos=Lr_dir_pos, values=mi.Vector2f((L * Lr_dir).x, 1.0))
                L = L + Le + Lr_dir
            elif δL is not None:
                with dr.resume_grad(when=not primal):
                    Le     = Le     * δL.read(pos=Le_pos)[0]
                    Lr_dir = Lr_dir * δL.read(pos=Lr_dir_pos)[0]

            if primal:
                block.put(pos=Le_pos,     values=mi.Vector2f(Le.x,     1.0), active=(Le.x     > 0.))
                block.put(pos=Lr_dir_pos, values=mi.Vector2f(Lr_dir.x, 1.0), active=(Lr_dir.x > 0.))
                L = L + Le + Lr_dir
            elif mode == dr.ADMode.Backward:
                L = L - Le - Lr_dir

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

            if not primal:
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

                    dist_next = distance + si_next.t

                    Le_n_pos     = mi.Point2f(ray.wavelengths.x - mi.Float(1.0),
                                            block.size().y * dist_next / max_distance)
                    Lr_dir_n_pos = mi.Point2f(ray.wavelengths.x - mi.Float(1.0),
                                            block.size().y *
                                            (dist_next + dr.norm(si_cur_reparam_only.p - si_next.p)) / max_distance)

                    if mode != dr.ADMode.Forward:
                        Le_next     = Le_next     * δL.read(pos=Le_n_pos)[0]
                        Lr_dir_next = Lr_dir_next * δL.read(pos=Lr_dir_n_pos)[0]

                    # Value of 'L' at the next vertex
                    L_next = L - dr.detach(Le_next) - dr.detach(Lr_dir_next)

                    # Account for the BSDF of the previous and next vertices
                    bsdf_val_prev = bsdf_prev.eval(bsdf_ctx, si_prev,
                                                   si_prev.to_local(wo_prev))
                    bsdf_val_next = bsdf_next.eval(bsdf_ctx, si_next,
                                                   bsdf_sample_next.wo)

                    extra = mi.Spectrum(Le_next)
                    extra[~first_vertex]      += L_prev * bsdf_val_prev / dr.maximum(1e-8, dr.detach(bsdf_val_prev))
                    extra[si_next.is_valid()] += L_next * bsdf_val_next / dr.maximum(1e-8, dr.detach(bsdf_val_next))

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
                    Lr_ind = L * dr.replace_grad(1, inv_bsdf_val_detach * bsdf_val)

                with dr.resume_grad():
                    # Differentiable Monte Carlo estimate of all contributions
                    Lo = (Le + Lr_dir + Lr_ind) * ray_reparam_det + extra

                    if dr.flag(dr.JitFlag.VCallRecord) and not dr.grad_enabled(Lo):
                        raise Exception(
                            "The contribution computed by the differential "
                            "rendering phase is not attached to the AD graph! "
                            "Raising an exception since this is usually "
                            "indicative of a bug (for example, you may have "
                            "forgotten to call dr.enable_grad(..) on one of "
                            "the scene parameters, or you may be trying to "
                            "optimize a parameter that does not generate "
                            "derivatives in detached PRB.)")

                    # Propagate derivatives from/to 'Lo' based on 'mode'
                    if mode == dr.ADMode.Backward:
                        dr.backward_from(Lo)
                    else:
                        # TODO: next term is missing
                        if dr.grad_enabled(Le * ray_reparam_det):
                            δL.put(pos=Le_pos,
                                   values=mi.Vector2f(dr.forward_to(Le * ray_reparam_det).x, 1.0))
                        if dr.grad_enabled(Lr_dir * ray_reparam_det):
                            δL.put(pos=Lr_dir_pos,
                                   values=mi.Vector2f(dr.forward_to(Lr_dir * ray_reparam_det).x, 1.0))
                        if dr.grad_enabled(Lr_ind * ray_reparam_det):
                            L = L + dr.forward_to(Lr_ind * ray_reparam_det)

            # Differential phases need access to the previous interaction, too
            if not primal:
                pi_prev  = pi_cur
                ray_prev = ray_cur

            # Provide ray/interaction to the next iteration
            pi_cur   = pi_next
            ray_cur  = ray_next

            depth[si_cur.is_valid()] += 1
            active = active_next

        return (
            L if primal else δL, # Radiance/differential radiance
            dr.neq(depth, 0),    # Ray validity flag for alpha blending
            L,                    # State for the differential phase
            δLdG
        )

    def to_string(self):
        return f'{type(self).__name__}[max_depth = {self.max_depth},' \
               f' rr_depth = { self.rr_depth },' \
               f' reparam_rays = { self.reparam_rays }]'

mi.register_integrator("prb_reparam_acoustic", lambda props: PRBReparamAcousticIntegrator(props))
