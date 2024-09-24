from __future__ import annotations # Delayed parsing of type annotations
from typing import Optional, Tuple, Callable, Any, Union

import drjit as dr
import mitsuba as mi
import gc

from .common import RBIntegrator, _ReparamWrapper, mis_weight

class UnrolledAcousticIntegrator(RBIntegrator):
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

        self.kappa = props.get('kappa', 0)

        self.rr_depth = props.get('rr_depth', self.max_depth)
        if self.rr_depth <= 0:
            raise Exception("\"rr_depth\" must be set to a value greater than zero!")


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
            ray, weight = self.sample_rays(scene, sensor, sampler)

            # Prepare an ImageBlock as specified by the film
            block = film.create_block()

            # Launch the Monte Carlo sampling process in primal mode
            self.sample(
                mode=dr.ADMode.Primal,
                scene=scene,
                sampler=sampler,
                sensor=None,
                ray=ray,
                block=block,
                δH=None,
                state_in_δHL=None,
                state_in_δHdT=None,
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

        wavelength_sample = mi.Float(0.)
        if film_size.x > 1:
            # Compute discrete sample position
            idx = dr.arange(mi.UInt32, film_size.x * spp) #dr.prod(film_size) * spp)

            # Try to avoid a division by an unknown constant if we can help it
            log_spp = dr.log2i(spp)
            if 1 << log_spp == spp:
                idx >>= dr.opaque(mi.UInt32, log_spp)
            else:
                idx //= dr.opaque(mi.UInt32, spp)

            wavelength_sample = mi.Float(idx)

        aperture_sample = mi.Vector2f(0.0)
        if sensor.needs_aperture_sample():
            aperture_sample = sampler.next_2d()

        time = 0.0 # sensor.shutter_open()

        with dr.resume_grad():
            ray, weight = sensor.sample_ray_differential(
                time=time,
                sample1=wavelength_sample,
                sample2=aperture_sample,
                sample3=aperture_sample
            )

        return ray, weight

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

        # Standard BSDF evaluation context for path tracing
        bsdf_ctx = mi.BSDFContext()

        # --------------------- Configure loop state ----------------------

        distance     = mi.Float(0.0)
        max_distance = self.max_time * self.speed_of_sound

        # Copy input arguments to avoid mutating the caller's state
        depth = mi.UInt32(0)  # Depth of current vertex
        ray = mi.Ray3f(ray)
        ray.d, ray_det = (ray.d, mi.Float(1)) if reparam is None else reparam(ray, depth)
        β = mi.Spectrum(1)                                 # Path throughput weight
        η = mi.Float(1)                                    # Index of refraction
        active = mi.Bool(active)                           # Active SIMD lanes

        # Variables caching information from the previous bounce
        prev_si         = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf   = mi.Float(0.) if self.skip_direct else mi.Float(1.)
        prev_bsdf_delta = mi.Bool(True)


        # Specify the max. number of loop iterations (this can help avoid
        # costly synchronization when when wavefront-style loops are generated)
        with dr.resume_grad():
            for iter in range(self.max_depth):
                active_next = mi.Bool(active)

                # Compute a surface interaction that tracks derivatives arising
                # from differentiable shape parameters (position, normals, etc.)
                # In primal mode, this is just an ordinary ray tracing operation.

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

                # Intensity of current emitter weighted by importance (def. by prev bsdf hits)
                Le = β * mis * ds.emitter.eval(si, active_next) * ray_det

                # ---------------------- Emitter sampling ----------------------

                # Should we continue tracing to reach one more vertex?
                active_next &= (depth + 1 < self.max_depth) & si.is_valid()

                # Is emitter sampling even possible on the current vertex?
                active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

                # If so, randomly sample an emitter without derivative tracking.
                with dr.suspend_grad():
                    ds, em_weight = scene.sample_emitter_direction(
                        si, sampler.next_2d(), True, active_em)

                active_em &= dr.neq(ds.pdf, 0.0)

                # Evaluate BSDF * cos(theta) differentiably
                wo = si.to_local(ds.d)
                bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
                mis_em = dr.select(ds.delta, 1, mis_weight(ds.pdf, bsdf_pdf_em))
                Lr_dir = β * mis_em * bsdf_value_em * em_weight * ray_det

                # ------------------ Detached BSDF sampling -------------------

                with dr.suspend_grad():
                    bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                                           sampler.next_1d(),
                                                           sampler.next_2d(),
                                                           active_next)

                Le_pos     = mi.Point2f(ray.wavelengths.x - mi.Float(1.0),
                                        block.size().y * distance / max_distance)
                Lr_dir_pos = mi.Point2f(ray.wavelengths.x - mi.Float(1.0),
                                        block.size().y * (distance + dr.norm(ds.p - si.p)) / max_distance)
                block.put(pos=Le_pos,     values=mi.Vector2f(Le.x,     1.0), active=(Le.x     > 0.))
                block.put(pos=Lr_dir_pos, values=mi.Vector2f(Lr_dir.x, 1.0), active=(Lr_dir.x > 0.))

                # ---- Update loop variables based on current interaction -----
                ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
                ray.d, new_ray_det = (ray.d, mi.Float(1)) if reparam is None else reparam(ray, depth)
                ray_det = ray_det * new_ray_det

                η *= bsdf_sample.eta
                β *= bsdf_weight

                # Information about the current vertex needed by the next iteration

                prev_si = si
                prev_bsdf_pdf = bsdf_sample.pdf
                prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

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


                depth[si.is_valid()] += 1
                active = active_next

        return block

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

    def render_forward(self: mi.SamplingIntegrator,
                       scene: mi.Scene,
                       params: Any,
                       sensor: Union[int, mi.Sensor] = 0,
                       seed: int = 0,
                       spp: int = 0) -> mi.TensorXf:

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

            reparam = _ReparamWrapper(
                scene=scene,
                params=params,
                reparam=self.reparam,
                wavefront_size=sampler.wavefront_size(),
                seed=seed
            )
            # Generate a set of rays starting at the sensor
            ray, weight = self.sample_rays(scene, sensor, sampler)

            # Prepare an ImageBlock as specified by the film
            block = film.create_block()

        # Launch the Monte Carlo sampling process in primal mode
        self.sample(
            mode=dr.ADMode.Primal,
            scene=scene,
            sampler=sampler,
            sensor=None,
            ray=ray,
            block=block,
            δH=None,
            state_in_δHL=None,
            state_in_δHdT=None,
            reparam=reparam,
            active=mi.Bool(True)
        )

        # Explicitly delete any remaining unused variables
        del sampler, ray, weight#, pos, L, valid
        gc.collect()

        # Perform the weight division and return an image tensor
        film.put_block(block)
        self.primal_image = film.develop()

        dr.forward_to(self.primal_image)

        return dr.grad(self.primal_image)

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

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.prepare(
                sensor=sensor,
                seed=seed,
                spp=spp,
                aovs=self.aovs()
            )

            reparam = _ReparamWrapper(
                scene=scene,
                params=params,
                reparam=self.reparam,
                wavefront_size=sampler.wavefront_size(),
                seed=seed
            )

            # Generate a set of rays starting at the sensor
            ray, weight = self.sample_rays(scene, sensor, sampler)

            # Prepare an ImageBlock as specified by the film
            block = film.create_block()

        with dr.resume_grad():
            # Launch the Monte Carlo sampling process in primal mode
            self.sample(
                mode=dr.ADMode.Primal,
                scene=scene,
                sampler=sampler,
                sensor=None,
                ray=ray,
                block=block,
                δH=None,
                state_in_δHL=None,
                state_in_δHdT=None,
                reparam=reparam,
                active=mi.Bool(True)
            )

            film.put_block(block)
            # Explicitly delete any remaining unused variables
            # del sampler, ray, weight#, pos, L, valid
            gc.collect()


            # This step launches a kernel
            dr.schedule(block.tensor())
            image = film.develop()

            # Differentiate sample splatting and weight division steps to
            # retrieve the adjoint radiance
            dr.set_grad(image, grad_in)
            dr.enqueue(dr.ADMode.Backward, image)
            dr.traverse(mi.Float, dr.ADMode.Backward)

        dr.set_grad(image, grad_in)
        dr.backward_from(image)



mi.register_integrator("unrolled_acoustic_reparam", lambda props: UnrolledAcousticIntegrator(props))
