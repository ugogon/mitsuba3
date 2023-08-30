#include <tuple>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/progress.h>
#include <mitsuba/core/timer.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
// #include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class AcousticPathIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(MonteCarloIntegrator, m_samples_per_pass, m_max_depth, m_hide_emitters,
                   m_render_timer, m_stop)
    MI_IMPORT_TYPES(Scene, Sensor, Film, ImageBlock, Medium, Sampler, BSDFPtr)

    AcousticPathIntegrator(const Properties &props) : Base(props) {
        m_max_time = props.get<float>("max_time", 1.f);
        if (m_max_time <= 0.f)
            Throw("\"max_time\" must be set to a value greater than zero!");

        m_skip_direct = props.get<bool>("skip_direct", false);
        m_enable_hit_model = props.get<bool>("enable_hit_model", true);
        m_enable_emitter_sampling = props.get<bool>("enable_emitter_sampling", true);
    }

    TensorXf render(Scene *scene,
                    Sensor *sensor,
                    uint32_t seed,
                    uint32_t spp,
                    bool develop,
                    bool evaluate) override {
        ScopedPhase sp(ProfilerPhase::Render);
        m_stop = false;

        Film *film = sensor->film();
        ScalarVector2u film_size = film->crop_size();

        // Potentially adjust the number of samples per pixel if spp != 0
        Sampler *sampler = sensor->sampler();
        if (spp)
            sampler->set_sample_count(spp);
        spp = sampler->sample_count();

        uint32_t spp_per_pass = (m_samples_per_pass == (uint32_t) -1)
                                        ? spp
                                        : std::min(m_samples_per_pass, spp);

        if ((spp % spp_per_pass) != 0)
            Throw("sample_count (%d) must be a multiple of spp_per_pass (%d).",
                  spp, spp_per_pass);

        uint32_t n_passes = spp / spp_per_pass;

        film->prepare({ }); // TODO: remove?

        m_render_timer.reset();

        TensorXf result;
        if constexpr (!dr::is_jit_v<Float>) {
            // TODO
            NotImplementedError("AcousticPathIntegrator::render Scalar case");
         } else {
            ref<ProgressReporter> progress = new ProgressReporter("Rendering");

            size_t wavefront_size = (size_t) film_size.x() * // time bins
                                    (size_t) film_size.y() * // wav bins
                                    (size_t) spp_per_pass,
                   wavefront_size_limit = 0xffffffffu;

            if (wavefront_size > wavefront_size_limit) {
                spp_per_pass /=
                    (uint32_t)((wavefront_size + wavefront_size_limit - 1) /
                               wavefront_size_limit);
                n_passes       = spp / spp_per_pass;
                wavefront_size = (size_t) film_size.x() * (size_t) film_size.y() *
                                 (size_t) spp_per_pass;

                Log(Warn,
                    "The requested rendering task involves %zu Monte Carlo "
                    "samples, which exceeds the upper limit of 2^32 = 4294967296 "
                    "for this variant. Mitsuba will instead split the rendering "
                    "task into %zu smaller passes to avoid exceeding the limits.",
                    wavefront_size, n_passes);
            }

            dr::sync_thread(); // Separate from scene initialization (for timings)

            Log(Info, "Starting render job (%ux%u, %u sample%s%s)",
                film_size.x(), film_size.y(), spp, spp == 1 ? "" : "s",
                n_passes > 1 ? tfm::format(", %u passes", n_passes) : "");

            if (n_passes > 1 && !evaluate) {
                Log(Warn, "render(): forcing 'evaluate=true' since multi-pass "
                          "rendering was requested.");
                evaluate = true;
            }

            // Inform the sampler about the passes (needed in vectorized modes)
            sampler->set_samples_per_wavefront(spp_per_pass);

            // Seed the underlying random number generators, if applicable
            sampler->seed(seed, (uint32_t) wavefront_size);

            // Allocate a large image block that will receive the entire rendering
            ref<ImageBlock> block = film->create_block();
            block->set_offset(film->crop_offset());
            block->set_coalesce(false); // TODO: remove?

            UInt32 band_id = dr::arange<UInt32>((uint32_t) wavefront_size);
            band_id /= dr::opaque<UInt32>(film_size.x() * spp_per_pass);

            Timer timer;

            for (size_t i = 0; i < n_passes; i++) {
                render_sample(scene, sensor, sampler, block, band_id);
                progress->update((i + 1) / (ScalarFloat) n_passes);

                if (n_passes > 1) {
                    sampler->advance();
                    sampler->schedule_state();
                    dr::eval(block->tensor());
                }
            }

            std::cout << "wavefront_size: " << wavefront_size << std::endl;

            film->put_block(block);

            if (n_passes == 1 && jit_flag(JitFlag::VCallRecord) &&
                jit_flag(JitFlag::LoopRecord)) {
                Log(Info, "Computation graph recorded. (took %s)",
                    util::time_string((float) timer.reset(), true));
            }

            if (develop) {
                result = film->develop();
                dr::schedule(result);
            } else {
                film->schedule_storage();
            }

            if (evaluate) {
                dr::eval();

                if (n_passes == 1 && jit_flag(JitFlag::VCallRecord) &&
                    jit_flag(JitFlag::LoopRecord)) {
                    Log(Info, "Code generation finished. (took %s)",
                        util::time_string((float) timer.value(), true));

                    /* Separate computation graph recording from the actual
                       rendering time in single-pass mode */
                    m_render_timer.reset();
                }

                dr::sync_thread();
            }
        }

        if (!m_stop && (evaluate || !dr::is_jit_v<Float>))
            Log(Info, "Rendering finished. (took %s)",
                util::time_string((float) m_render_timer.value(), true));

        return result;
    }

    /// default function signature for proper inheritance
    std::pair<Spectrum, Bool> sample(const Scene *,
                                     Sampler *,
                                     const RayDifferential3f &,
                                     const Medium * /* medium */,
                                     Float * /* aovs */,
                                     Bool) const override {
        NotImplementedError("AcousticPathIntegrator::sample default arguments");
    }

    std::pair<Spectrum, Mask> sample(const Scene *scene,
                                     Sampler *sampler,
                                     const RayDifferential3f &ray_,
                                     ImageBlock *block,
                                     const UInt32 band_id,
                                     Bool active) const {
        MI_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        if (unlikely(m_max_depth == 0))
            return { 0.f, false };

        // --------------------- Configure loop state ----------------------

        Ray3f ray                      = Ray3f(ray_);
        Spectrum throughput            = 1.f;
        // Spectrum result               = 0.f;
        // Float eta                     = 1.f;
        UInt32 depth                   = 0;
        Float distance                 = 0.f;
        const ScalarFloat max_distance = m_max_time * MI_SOUND_SPEED;

        // If m_hide_emitters == false, the environment emitter will be visible
        Mask valid_ray                 = !m_hide_emitters && dr::neq(scene->environment(), nullptr);
        // Mask hit_emitter_before = false;

        // Variables caching information from the previous bounce
        Interaction3f prev_si          = dr::zeros<Interaction3f>();
        Float         prev_bsdf_pdf    = m_skip_direct ? 0.f : 1.f;
        Bool          prev_bsdf_delta  = true;
        BSDFContext   bsdf_ctx;

        /* Set up a Dr.Jit loop. This optimizes away to a normal loop in scalar
           mode, and it generates either a a megakernel (default) or
           wavefront-style renderer in JIT variants. This can be controlled by
           passing the '-W' command line flag to the mitsuba binary or
           enabling/disabling the JitFlag.LoopRecord bit in Dr.Jit.

           The first argument identifies the loop by name, which is helpful for
           debugging. The subsequent list registers all variables that encode
           the loop state variables. This is crucial: omitting a variable may
           lead to undefined behavior. */
        dr::Loop<Bool> loop("AcousticPath", sampler, ray, throughput, /* result, */
                            /* eta, */ depth, distance, valid_ray, /* hit_emitter_before, */
                            prev_si, prev_bsdf_pdf,
                            prev_bsdf_delta, active);

        /* Inform the loop about the maximum number of loop iterations.
           This accelerates wavefront-style rendering by avoiding costly
           synchronization points that check the 'active' flag. */
        loop.set_max_iterations(m_max_depth);

        while (loop(active)) {
            /* dr::Loop implicitly masks all code in the loop using the 'active'
               flag, so there is no need to pass it to every function */

            SurfaceInteraction3f si =
                scene->ray_intersect(ray,
                                     /* ray_flags = */ +RayFlags::All,
                                     /* coherent = */ dr::eq(depth, 0u));

            distance += si.t;

            // ---------------------- Direct emission ----------------------

            // TODO: hit_emitter(_before) ausbauen
            /* dr::any_or() checks for active entries in the provided boolean
               array. JIT/Megakernel modes can't do this test efficiently as
               each Monte Carlo sample runs independently. In this case,
               dr::any_or<..>() returns the template argument (true) which means
               that the 'if' statement is always conservatively taken. */
            Bool hit_emitter = dr::neq(si.emitter(scene), nullptr);
            if (m_enable_hit_model && dr::any_or<true>(hit_emitter)) {
                DirectionSample3f ds(scene, si, prev_si);
                Float em_pdf = 0.f;

                if (dr::any_or<true>(!prev_bsdf_delta))
                    em_pdf = scene->pdf_emitter_direction(prev_si, ds,
                                                          !prev_bsdf_delta);

                // Compute MIS weight for emitter sample from previous bounce
                Float mis_bsdf = mis_weight(prev_bsdf_pdf, em_pdf);

                /* TODO REMOVE
                 * mitsuba2:
                 * emission_weight = mis_bsdf
                */

                // Put the result while checking for double hits (rays that are traced through the detector)
                Float time_frac = (distance / max_distance) * block->size().x();
                Bool valid_hit  = hit_emitter; // && !hit_emitter_before;
                // hist->put(
                //     { time_frac, band_id },
                //     { },
                //     throughput * ds.emitter->eval(si, prev_bsdf_pdf > 0.f) * mis_bsdf,
                //     valid_hit);

                // TODO wofür wird das gebraucht?
                // hit_emitter_before = hit_emitter;
            }

            // Continue tracing the path at this point?
            Bool active_next = (depth + 1 < m_max_depth) && si.is_valid();

            if (dr::none_or<false>(active_next))
                break; // early exit for scalar mode

            BSDFPtr bsdf = si.bsdf(ray);

            // ---------------------- Emitter sampling ----------------------

            // Perform emitter sampling?
            // TODO: BSDFFlags::DiffuseReflection?
            Mask active_em = active_next && has_flag(bsdf->flags(), BSDFFlags::Smooth);

            DirectionSample3f ds = dr::zeros<DirectionSample3f>();
            Spectrum em_weight = dr::zeros<Spectrum>();
            Vector3f wo = dr::zeros<Vector3f>();

            if (m_enable_emitter_sampling && dr::any_or<true>(active_em)) {
                // Sample the emitter
                std::tie(ds, em_weight) = scene->sample_emitter_direction(
                    si, sampler->next_2d(), true, active_em);
                active_em &= dr::neq(ds.pdf, 0.f);

                /* Given the detached emitter sample, recompute its contribution
                   with AD to enable light source optimization. */
                if (dr::grad_enabled(si.p)) {
                    ds.d = dr::normalize(ds.p - si.p);
                    Spectrum em_val = scene->eval_emitter_direction(si, ds, active_em);
                    em_weight = dr::select(dr::neq(ds.pdf, 0), em_val / ds.pdf, 0);
                }

                wo = si.to_local(ds.d);
            }

            // ------ Evaluate BSDF * cos(theta) and sample direction -------

            Float sample_1 = sampler->next_1d();
            Point2f sample_2 = sampler->next_2d();

            auto [bsdf_val, bsdf_pdf, bsdf_sample, bsdf_weight]
                = bsdf->eval_pdf_sample(bsdf_ctx, si, wo, sample_1, sample_2);

            // --------------- Emitter sampling contribution ----------------

            if (dr::any_or<true>(active_em)) {
                bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

                // Compute the MIS weight
                Float mis_em =
                    dr::select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));

                // Put the result while
                Float time_frac = ((distance + ds.dist) / max_distance) * block->size().x();
                Spectrum em_throughput = throughput * bsdf_val * em_weight * mis_em;
                // TODO is this check required?
                active_em = active_em && dr::any(dr::neq(em_throughput, 0.f));
                // hist->put({ time_frac, band_id }, { }, em_throughput, active_em);
            }

            // ---------------------- BSDF sampling ----------------------

            /* TODO REMOVE
             * mitsuba2:
             * bs       = bsdf_sample
             * bsdf_val = bsdf_weight
            */

            bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi);

            ray = si.spawn_ray(si.to_world(bsdf_sample.wo));

            /* When the path tracer is differentiated, we must be careful that
               the generated Monte Carlo samples are detached (i.e. don't track
               derivatives) to avoid bias resulting from the combination of moving
               samples and discontinuous visibility. We need to re-evaluate the
               BSDF differentiably with the detached sample in that case. */
            if (dr::grad_enabled(ray)) {
                ray = dr::detach<true>(ray);

                // Recompute 'wo' to propagate derivatives to cosine term
                Vector3f wo_2 = si.to_local(ray.d);
                auto [bsdf_val_2, bsdf_pdf_2] = bsdf->eval_pdf(bsdf_ctx, si, wo_2, active);
                bsdf_weight[bsdf_pdf_2 > 0.f] = bsdf_val_2 / dr::detach(bsdf_pdf_2);
            }

            // ------ Update loop variables based on current interaction ------

            throughput *= bsdf_weight;
            valid_ray |= active && si.is_valid() &&
                         !has_flag(bsdf_sample.sampled_type, BSDFFlags::Null);

            // Information about the current vertex needed by the next iteration
            prev_si = si;
            prev_bsdf_pdf = bsdf_sample.pdf;
            prev_bsdf_delta = has_flag(bsdf_sample.sampled_type, BSDFFlags::Delta);

            // -------------------- Stopping criterion ---------------------

            dr::masked(depth, si.is_valid()) += 1;

            Float throughput_max = dr::max(unpolarized_spectrum(throughput));

            // TODO: include rr sampling?
            /*
            Float rr_prob = dr::minimum(throughput_max * dr::sqr(eta), .95f);
            Mask rr_active = depth >= m_rr_depth,
                 rr_continue = sampler->next_1d() < rr_prob;
            */

            /* Differentiable variants of the renderer require the the russian
               roulette sampling weight to be detached to avoid bias. This is a
               no-op in non-differentiable variants. */
            /*
            throughput[rr_active] *= dr::rcp(dr::detach(rr_prob));
            */

            active = active_next && // (!rr_active || rr_continue) &&
                     dr::neq(throughput_max, 0.f);
        }

        return {
            /* spec  = */ dr::select(valid_ray, throughput, 0.f),
            /* valid = */ valid_ray
        };
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "AcousticPathIntegrator["
            <<  "\n  stop = "            << m_stop
            << ",\n  max_depth = "       << m_max_depth
            // << ",\n  wavelength_bins = " << m_wavelength_bins
            <<  "\n]";
        return oss.str();
    }

    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        return dr::select(pdf_a > 0.f, pdf_a / (pdf_a + pdf_b), 0.f);
    }

    /**
     * \brief Perform a Mueller matrix multiplication in polarized modes, and a
     * fused multiply-add otherwise.
     */
    Spectrum spec_fma(const Spectrum &a, const Spectrum &b,
                      const Spectrum &c) const {
        if constexpr (is_polarized_v<Spectrum>)
            return a * b + c;
        else
            return dr::fmadd(a, b, c);
    }

    MI_DECLARE_CLASS()

protected:

    void render_sample(const Scene *scene,
                       const Sensor *sensor,
                       Sampler *sampler,
                       ImageBlock *block,
                       // Float *aovs,
                       // const Vector2f &pos,
                       const UInt32 band_id,
                       Mask active = true) const {

        Point2f direction_sample = sampler->next_2d(active);
        Float wavelength_sample = Float(band_id) + 1.f;

        auto [ray, ray_weight] = sensor->sample_ray_differential(
            0., wavelength_sample, { 0., 0. }, direction_sample);

        sample(scene, sampler, ray, block, band_id, active);

        sampler->advance();
    }

protected:
    float m_max_time;

    bool m_skip_direct;
    bool m_enable_hit_model;
    bool m_enable_emitter_sampling;
};

MI_IMPLEMENT_CLASS_VARIANT(AcousticPathIntegrator, MonteCarloIntegrator)
MI_EXPORT_PLUGIN(AcousticPathIntegrator, "Acoustic Path Tracer integrator");
NAMESPACE_END(mitsuba)
