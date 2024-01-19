from __future__ import annotations # Delayed parsing of type annotations
from typing import Optional, Tuple, Callable, Any, Union

import drjit as dr
import mitsuba as mi
import gc

from .common import RBIntegrator, _ReparamWrapper, mis_weight

class PRBAcousticIntegrator(RBIntegrator):
    def __init__(self, props = mi.Properties()):
        super().__init__(props)

        self.max_time    = props.get("max_time", 1.)
        self.speed_of_sound = props.get("speed_of_sound", 343.)
        if self.max_time <= 0. or self.speed_of_sound <= 0.:
            raise Exception("\"max_time\" and \"speed_of_sound\" must be set to a value greater than zero!")

        self.skip_direct = props.get("skip_direct", False)

        self.track_time_derivatives = props.get("track_time_derivatives", True)

        max_depth = props.get('max_depth', 6)
        if max_depth < 0 and max_depth != -1:
            raise Exception("\"max_depth\" must be set to -1 (infinite) or a value >= 0")

        # Map -1 (infinity) to 2^32-1 bounces
        self.max_depth = max_depth if max_depth != -1 else 0xffffffff

        self.rr_depth = props.get('rr_depth', self.max_depth)
        if self.rr_depth <= 0:
            raise Exception("\"rr_depth\" must be set to a value greater than zero!")

    def render(self: mi.SamplingIntegrator,
               scene: mi.Scene,
               sensor: Union[int, mi.Sensor] = 0,
               seed: int = 0,
               spp: int = 0,
               develop: bool = True,
               evaluate: bool = True) -> mi.TensorXf:

        if not develop:
            raise Exception("develop=True must be specified when "
                            "invoking AD integrators")

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.prepare(
                sensor=sensor,
                seed=seed,
                spp=spp,
                aovs=self.aovs()
            )

            # Generate a set of rays starting at the sensor
            ray, weight, _, _ = self.sample_rays(scene, sensor, sampler)

            # Prepare an ImageBlock as specified by the film
            block = film.create_block()

            # Launch the Monte Carlo sampling process in primal mode
            self.sample(
                mode=dr.ADMode.Primal,
                scene=scene,
                sampler=sampler,
                ray=ray,
                block=block,
                δL=None,
                state_in_δL=None,
                state_in_δLdG=None,
                reparam=None,
                active=mi.Bool(True)
            )

            # Explicitly delete any remaining unused variables
            del sampler, ray, weight#, pos, L, valid
            gc.collect()

            # Perform the weight division and return an image tensor
            film.put_block(block)
            self.primal_image = film.develop()

            return self.primal_image

    def sample_rays(
        self,
        scene: mi.Scene,
        sensor: mi.Sensor,
        sampler: mi.Sampler,
        reparam: Callable[[mi.Ray3f, mi.UInt32, mi.Bool],
                          Tuple[mi.Vector3f, mi.Float]] = None
    ) -> Tuple[mi.RayDifferential3f, mi.Spectrum, mi.Vector2f, mi.Float]:

        film = sensor.film()
        film_size = film.crop_size()
        rfilter = film.rfilter()
        border_size = rfilter.border_size()

        if film.sample_border():
            film_size += 2 * border_size

        spp = sampler.sample_count()

        # Compute discrete sample position
        idx = dr.arange(mi.UInt32, film_size.x * spp) #dr.prod(film_size) * spp)

        # Try to avoid a division by an unknown constant if we can help it
        log_spp = dr.log2i(spp)
        if 1 << log_spp == spp:
            idx >>= dr.opaque(mi.UInt32, log_spp)
        else:
            idx //= dr.opaque(mi.UInt32, spp)

        # Compute the position on the image plane
        pos = mi.Vector2i(idx, 0 * idx)

        # Re-scale the position to [0, 1]^2
        scale = dr.rcp(mi.ScalarVector2f(film_size))
        pos_adjusted = mi.Vector2f(pos) * scale

        aperture_sample = mi.Vector2f(0.0)
        if sensor.needs_aperture_sample():
            aperture_sample = sampler.next_2d()

        time = 0.0 # sensor.shutter_open()

        wavelength_sample = 0.
        if mi.is_spectral:
            wavelength_sample = mi.Float(idx) + 1.0

        with dr.resume_grad():
            ray, weight = sensor.sample_ray_differential(
                time=time,
                sample1=wavelength_sample,
                sample2=pos_adjusted,
                sample3=aperture_sample
            )

        reparam_det = 1.0

        if reparam is not None:
            with dr.resume_grad():
                # Reparameterize the camera ray
                _, reparam_det = reparam(ray=dr.detach(ray), depth=mi.UInt32(0))

        return ray, weight, pos, reparam_det

    def prepare(self,
                sensor: mi.Sensor,
                seed: int = 0,
                spp: int = 0,
                aovs: list = []):

        film = sensor.film()
        sampler = sensor.sampler().clone()

        if spp != 0:
            sampler.set_sample_count(spp)

        spp = sampler.sample_count()
        sampler.set_samples_per_wavefront(spp)

        film_size = film.crop_size()

        if film.sample_border():
            film_size += 2 * film.rfilter().border_size()

        wavefront_size = film_size.x * spp # dr.prod(film_size) * spp

        if wavefront_size > 2**32:
            raise Exception(
                "The total number of Monte Carlo samples required by this "
                "rendering task (%i) exceeds 2^32 = 4294967296. Please use "
                "fewer samples per pixel or render using multiple passes."
                % wavefront_size)

        sampler.seed(seed, wavefront_size)
        film.prepare(aovs)

        return sampler, spp

    def sample(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               block: mi.ImageBlock,
               δL: Optional[mi.ImageBlock],
               state_in_δL: Optional[mi.Spectrum],
               state_in_δLdG: Optional[mi.Spectrum],
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
        ray = mi.Ray3f(dr.detach(ray))
        depth = mi.UInt32(0)                               # Depth of current vertex
        L    = mi.Spectrum(0 if primal else state_in_δL)   # Radiance accumulator
        δLdG = mi.Spectrum(0 if primal else state_in_δLdG) # Radiance*Gaussian accumulator
        β = mi.Spectrum(1)                                 # Path throughput weight
        η = mi.Float(1)                                    # Index of refraction
        active = mi.Bool(active)                           # Active SIMD lanes

        # Variables caching information from the previous bounce
        prev_si         = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf   = mi.Float(0.) if self.skip_direct else mi.Float(1.)
        prev_bsdf_delta = mi.Bool(True)

        # Record the following loop in its entirety
        # TODO: loop should keep track of imageblock and δL
        loop = mi.Loop(name="PRB Acoustic (%s)" % mode.name,
                       state=lambda: (distance, #block.tensor(),
                                      sampler, ray, depth, L, δLdG, #δL.tensor(),
                                      β, η, active,
                                      prev_si, prev_bsdf_pdf, prev_bsdf_delta))

        # Specify the max. number of loop iterations (this can help avoid
        # costly synchronization when when wavefront-style loops are generated)
        loop.set_max_iterations(self.max_depth)

        while loop(active):
            active_next = mi.Bool(active)

            # Compute a surface interaction that tracks derivatives arising
            # from differentiable shape parameters (position, normals, etc.)
            # In primal mode, this is just an ordinary ray tracing operation.
            with dr.resume_grad(when=not primal):
                si = scene.ray_intersect(ray,
                                         ray_flags=mi.RayFlags.All,
                                         coherent=dr.eq(depth, 0))

            distance += si.t

            # Get the BSDF, potentially computes texture-space differentials
            bsdf = si.bsdf(ray)

            # ---------------------- Direct emission ----------------------

            # Hide the environment emitter if necessary
            if self.hide_emitters:
                active_next &= ~(dr.eq(depth, 0) & ~si.is_valid())

            # Compute MIS weight for emitter sample from previous bounce
            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

            mis = mis_weight(
                prev_bsdf_pdf,
                scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
            )

            with dr.resume_grad(when=not primal):
                Le = β * mis * ds.emitter.eval(si, active_next)

            # ---------------------- Emitter sampling ----------------------

            # Should we continue tracing to reach one more vertex?
            active_next &= (depth + 1 < self.max_depth) & si.is_valid()

            # Is emitter sampling even possible on the current vertex?
            active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

            # If so, randomly sample an emitter without derivative tracking.
            ds, em_weight = scene.sample_emitter_direction(
                si, sampler.next_2d(), True, active_em)
            active_em &= dr.neq(ds.pdf, 0.0)

            with dr.resume_grad(when=not primal):
                if not primal:
                    # Given the detached emitter sample, *recompute* its
                    # contribution with AD to enable light source optimization
                    ds.d = dr.replace_grad(ds.d, dr.normalize(ds.p - si.p))
                    em_val = scene.eval_emitter_direction(si, ds, active_em)
                    em_weight = dr.replace_grad(em_weight, dr.select(dr.neq(ds.pdf, 0), em_val / ds.pdf, 0))
                    dr.disable_grad(ds.d)

                # Evaluate BSDF * cos(theta) differentiably
                wo = si.to_local(ds.d)
                bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
                mis_em = dr.select(ds.delta, 1, mis_weight(ds.pdf, bsdf_pdf_em))
                Lr_dir = β * mis_em * bsdf_value_em * em_weight

            # ------------------ Detached BSDF sampling -------------------

            bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                                   sampler.next_1d(),
                                                   sampler.next_2d(),
                                                   active_next)

            # ---- Update loop variables based on current interaction -----

            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
            η *= bsdf_sample.eta
            β *= bsdf_weight

            # Information about the current vertex needed by the next iteration

            prev_si = dr.detach(si, True)
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            # ---- PRB-style tracking of time derivatives -----

            if self.track_time_derivatives:
                active_time      = active & si.is_valid()
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
                                        dr.detach(distance + dr.norm(ds.p - si.p)), 0.) 
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
                        t0     = dr.select(active_time,      si.t,                 0.)
                        t0_dir = dr.select(active_time_next, dr.norm(ds.p - si.p), 0.)

                        # TODO (MW): why not -t0 ...?
                        dr.backward_from(t0     * δLdG)
                        dr.backward_from(t0_dir * δLdG_Lr_dir) # <- attention, this accounts for the direct light segment!
                    δLdG = δLdG - δL_Le - δL_Lr_dir

            # put and accumulate current (differential) radiance

            Le_pos     = mi.Point2f(ray.wavelengths.x - mi.Float(1.0),
                                    block.size().y * distance / max_distance)
            Lr_dir_pos = mi.Point2f(ray.wavelengths.x - mi.Float(1.0),
                                    block.size().y * (distance + dr.norm(ds.p - si.p)) / max_distance)

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

            # ------------------ Differential phase only ------------------

            if not primal:
                with dr.resume_grad():
                    # Recompute 'wo' to propagate derivatives to cosine term
                    wo = si.to_local(ray.d)

                    # Re-evaluate BSDF * cos(theta) differentiably
                    bsdf_val = bsdf.eval(bsdf_ctx, si, wo, active_next)

                    # Detached version of the above term and inverse
                    bsdf_val_det = bsdf_weight * bsdf_sample.pdf
                    inv_bsdf_val_det = dr.select(dr.neq(bsdf_val_det, 0),
                                                 dr.rcp(bsdf_val_det), 0)

                    # Differentiable version of the reflected indirect radiance
                    Lr_ind = dr.replace_grad(1, inv_bsdf_val_det * bsdf_val)

                    # Differentiable Monte Carlo estimate of all contributions
                    Lo = Le + Lr_dir + L * Lr_ind

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
                        if dr.grad_enabled(Le):
                            δL.put(pos=Le_pos,
                                   values=mi.Vector2f(dr.forward_to(Le).x, 1.0))
                        if dr.grad_enabled(Lr_dir):
                            δL.put(pos=Lr_dir_pos,
                                   values=mi.Vector2f(dr.forward_to(Lr_dir).x, 1.0))
                        if dr.grad_enabled(Lr_ind):
                            L = L + dr.forward_to(Lr_ind)

            depth[si.is_valid()] += 1
            active = active_next

        return (
            L if primal else δL, # Radiance/differential radiance
            dr.neq(depth, 0),    # Ray validity flag for alpha blending
            L,                   # State for the differential phase
            δLdG
        )

    def render_forward(self: mi.SamplingIntegrator,
                       scene: mi.Scene,
                       params: Any,
                       sensor: Union[int, mi.Sensor] = 0,
                       seed: int = 0,
                       spp: int = 0) -> mi.TensorXf:

        mi.Log(mi.LogLevel.Warn,
               "Acoustic Forward Mode is untested and probably "
               "biased, especially when using reparam!")

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()
        aovs = self.aovs()

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.prepare(sensor, seed, spp, aovs)

            # When the underlying integrator supports reparameterizations,
            # perform necessary initialization steps and wrap the result using
            # the _ReparamWrapper abstraction defined above
            if hasattr(self, 'reparam'):
                reparam = _ReparamWrapper(
                    scene=scene,
                    params=params,
                    reparam=self.reparam,
                    wavefront_size=sampler.wavefront_size(),
                    seed=seed
                )
            else:
                reparam = None

            # Generate a set of rays starting at the sensor, keep track of
            # derivatives wrt. sample positions ('pos') if there are any
            ray, weight, _, det = self.sample_rays(scene, sensor,
                                                     sampler, reparam)

            δL, valid, L, δLdG = self.sample(
                mode=dr.ADMode.Forward,
                scene=scene,
                sampler=sampler,
                ray=ray,
                block=film.create_block(),
                δL=film.create_block(),
                state_in_δL=mi.Spectrum(0.),
                state_in_δLdG=mi.Spectrum(0.),
                reparam=reparam,
                active=mi.Bool(True)
            )

            sample_pos_deriv = None # disable by default

            with dr.resume_grad():
                if True and reparam is not None:
                    L[~valid] = 0.0
                    sample_pos_deriv = dr.mean(L.x * weight.x * det)

                    # Compute the derivative of the reparameterized image ..
                    dr.forward_to(sample_pos_deriv, flags=dr.ADFlag.ClearInterior | dr.ADFlag.ClearEdges)

                    dr.schedule(sample_pos_deriv, dr.grad(sample_pos_deriv))

            film.put_block(δL)

            # Explicitly delete any remaining unused variables
            del sampler, ray, weight, L, valid, δL, params #, state_out, pos

            # Probably a little overkill, but why not.. If there are any
            # DrJit arrays to be collected by Python's cyclic GC, then
            # freeing them may enable loop simplifications in dr.eval().
            gc.collect()

            result_grad = film.develop()

            # Potentially add the derivative of the reparameterized samples
            if sample_pos_deriv is not None:
                with dr.resume_grad():
                    result_grad += dr.grad(sample_pos_deriv)

        return result_grad

    def render_backward(self: mi.SamplingIntegrator,
                        scene: mi.Scene,
                        params: Any,
                        grad_in: mi.TensorXf,
                        sensor: Union[int, mi.Sensor] = 0,
                        seed: int = 0,
                        spp: int = 0) -> None:

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()
        aovs = self.aovs()

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.prepare(sensor, seed, spp, aovs)

            # When the underlying integrator supports reparameterizations,
            # perform necessary initialization steps and wrap the result using
            # the _ReparamWrapper abstraction defined above
            if hasattr(self, 'reparam'):
                reparam = _ReparamWrapper(
                    scene=scene,
                    params=params,
                    reparam=self.reparam,
                    wavefront_size=sampler.wavefront_size(),
                    seed=seed
                )
            else:
                reparam = None

            # Generate a set of rays starting at the sensor, keep track of
            # derivatives wrt. sample positions ('pos') if there are any
            ray, weight, _, det = self.sample_rays(scene, sensor,
                                                     sampler, reparam)

            δL = mi.ImageBlock(grad_in,
                               rfilter=film.rfilter(),
                               border=film.sample_border(),
                               y_only=hasattr(self, 'reparam'))

            # # Clear the dummy data splatted on the film above
            # film.clear()
            block = film.create_block()

            # Launch the Monte Carlo sampling process in primal mode (1)
            L, valid, state_out_δL, state_out_δLdG = self.sample(
                mode=dr.ADMode.Primal,
                scene=scene,
                sampler=sampler.clone(),
                ray=ray,
                block=block,
                δL=δL,
                state_in_δL=None,
                state_in_δLdG=None,
                reparam=None,
                active=mi.Bool(True)
            )

            # Launch Monte Carlo sampling in backward AD mode (2)
            L_2, valid_2, state_out_δL_2, state_out_δLdG_2 = self.sample(
                mode=dr.ADMode.Backward,
                scene=scene,
                sampler=sampler,
                ray=ray,
                block=block,
                δL=δL,
                state_in_δL=state_out_δL,
                state_in_δLdG=state_out_δLdG,
                reparam=reparam,
                active=mi.Bool(True)
            )

            # Propagate gradient image to sample positions if necessary
            if reparam is not None:
                with dr.resume_grad():
                    # After reparameterizing the camera ray, we need to evaluate
                    #   Σ (fi Li det)
                    #  ---------------
                    #   Σ (fi det)
                    L[~valid] = 0.0
                    dr.backward(dr.mean(L * weight * det))

            # We don't need any of the outputs here
            del L, L_2, valid, valid_2, state_out_δL, state_out_δLdG, state_out_δL_2, state_out_δLdG_2, δL, \
                ray, weight, det, sampler #, pos

            gc.collect()

            # Run kernel representing side effects of the above
            dr.eval()

mi.register_integrator("prb_acoustic", lambda props: PRBAcousticIntegrator(props))
