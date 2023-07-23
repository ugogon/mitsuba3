// #include <random>
// #include <map>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/progress.h>
#include <mitsuba/core/properties.h>
// #include <mitsuba/render/bsdf.h>
// #include <mitsuba/render/emitter.h>
#include <mitsuba/render/histogram.h>
#include <mitsuba/render/integrator.h>
// #include <mitsuba/render/records.h>
// #include <enoki/stl.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class AcousticPathIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(MonteCarloIntegrator, m_samples_per_pass, m_max_depth, m_render_timer, m_stop)
    MI_IMPORT_TYPES(Scene, Sensor, Film, Sampler, Histogram)

    AcousticPathIntegrator(const Properties &props) : Base(props) {
        // TODO

        // m_time_step_count = 0;
        // m_hide_emitters = props.get<bool>("hide_emitters", false);

        // m_max_time = props.get<float>("max_time", 1.0f);
        // if (m_max_time <= 0)
            // Throw("\"max_time\" must be set to a value greater than zero!");

        std::vector<std::string> wavelengths_str =
            string::tokenize(props.get<std::string>("wavelength_bins"), " ,");

        // m_wav_bin_count = wavelengths_str.size();
        size_t wav_bin_count = wavelengths_str.size();

        // Allocate space
        m_wavelength_bins = dr::zeros<TensorXf>(wav_bin_count);

        // Copy and convert to wavelengths
        for (size_t i = 0; i < wav_bin_count; ++i) {
            try {
                Float wav = std::stod(wavelengths_str[i]);
                dr::scatter(m_wavelength_bins.array(), wav, UInt32(i));
            } catch (...) {
                Throw("Could not parse floating point value '%s'",
                      wavelengths_str[i]);
            }
        }

        // m_skip_direct = props.get<bool>("skip_direct", false);
        // m_enable_hit_model = props.get<bool>("enable_hit_model", true);
        // m_enable_emitter_sampling = props.get<bool>("enable_emitter_sampling", true);
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
        // m_time_step_count = film_size.x(); // TODO

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

        // Determine output channels and prepare the film with this information
        // size_t n_channels = film->prepare(aov_names()); // TODO

        m_render_timer.reset();

        // TensorXf result; // TODO
        if constexpr (!dr::is_jit_v<Float>) {
            // TODO
            NotImplementedError("TimeDependentIntergrator::render Scalar case");
         } else {
            ref<ProgressReporter> progress = new ProgressReporter("Rendering");

            size_t wavefront_size = (size_t) film_size.x() *
                                    (size_t) film_size.y() * (size_t) spp_per_pass,
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

            // TODO: was geben idx und band_id an?
            UInt32 idx = dr::arange<UInt32>((uint32_t) wavefront_size);
            if (spp_per_pass > 1)
                idx /= dr::opaque<UInt32>(spp_per_pass);

            UInt32 band_id = dr::zeros<UInt32>((uint32_t) wavefront_size);
            if (film_size.y() > 1)
                band_id = idx % film_size.y();

            ref<Histogram> hist = new Histogram(film_size, 1, film->rfilter());
            hist->clear();

            for (size_t i = 0; i < n_passes; i++) {
                render_sample(scene, sensor, sampler, hist, band_id);
                progress->update( (i + 1) / (ScalarFloat) n_passes);

                // TODO: sampler->advance() required?
            }

            std::cout << "wavefront_size: " << wavefront_size << std::endl;

            film->put_block(hist);

            // TODO: develop and evaluate stuff?
         }

        if (!m_stop && (evaluate || !dr::is_jit_v<Float>))
            Log(Info, "Rendering finished. (took %s)",
                util::time_string((float) m_render_timer.value(), true));

        return { }; // TODO: Rückgabewert in mi3 unterscheidet sich zu mi2
    }

    std::pair<Spectrum, Mask> trace_acoustic_ray(const Scene *scene,
                                                 Sampler *sampler,
                                                 const Ray3f &ray_,
                                                 Histogram *hist,
                                                 const UInt32 band_id,
                                                 Mask active) const {
        // MI_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        // if (unlikely(m_max_depth == 0))
        //     return { 0.f, false };

        // // --------------------- Configure loop state ----------------------

        // Ray3f ray = Ray3f(ray_);

        // Float distance = 0.f;
        // // const ScalarFloat = max_distance = m_max_time * MI_SOUND_SPEED

        // return { 0.f, false };
    }

    //! @}
    // =============================================================

    // TODO
    std::string to_string() const override {
        return tfm::format("AcousticPathIntegrator[\n"
            "  max_depth = %u,\n"
            "  stop = %u\n"
            "]", m_max_depth, m_stop);
    }

    // TODO: needed? what does that do?
    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        return dr::select(pdf_a > 0.f, pdf_a / (pdf_a + pdf_b), 0.f);
    }

    /// Get the bins for each which we integrate
    TensorXf wavelength_bins() const { return m_wavelength_bins; }

    MI_DECLARE_CLASS()

protected:

    void render_sample(const Scene *scene,
                  const Sensor *sensor,
                  Sampler *sampler,
                  Histogram *hist,
                  const UInt32 band_id,
                  Mask active = true) const {
        Point2f direction_sample = sampler->next_2d(active);
        Float wavelength_sample = dr::gather<Float>(m_wavelength_bins.array(), band_id, active);
        auto [ray, ray_weight] = sensor->sample_ray(0, wavelength_sample, { 0., 0. }, direction_sample);
        trace_acoustic_ray(scene, sampler, ray, hist, band_id, active);
        sampler->advance();
    }

protected:
    // float m_max_time;
    // size_t m_time_step_count;
    // size_t m_wav_bin_count;
    TensorXf m_wavelength_bins;

    // bool m_skip_direct;
    // bool m_enable_hit_model;
    // bool m_enable_emitter_sampling;
};

MI_IMPLEMENT_CLASS_VARIANT(AcousticPathIntegrator, MonteCarloIntegrator)
MI_EXPORT_PLUGIN(AcousticPathIntegrator, "Acoustic Path Tracer integrator");
NAMESPACE_END(mitsuba)
