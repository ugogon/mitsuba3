// #include <random>
// #include <map>
// #include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
// #include <mitsuba/render/bsdf.h>
// #include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
// #include <mitsuba/render/records.h>
// #include <enoki/stl.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class AcousticPathIntegrator : public TimeDependentIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(TimeDependentIntegrator, m_max_depth, m_stop)
    MI_IMPORT_TYPES(Scene, Sensor, Sampler, Histogram)

    AcousticPathIntegrator(const Properties &props) : Base(props) {
        // TODO
        // m_skip_direct = props.get<bool>("skip_direct", false);
        // m_enable_hit_model = props.get<bool>("enable_hit_model", true);
        // m_enable_emitter_sampling = props.get<bool>("enable_emitter_sampling", true);
    }

    std::pair<Spectrum, Mask> trace_acoustic_ray(const Scene *scene,
                                                 Sampler *sampler,
                                                 const Ray3f &ray,
                                                 Histogram *hist,
                                                 const UInt32 band_id,
                                                 Mask active) const override {
        NotImplementedError("trace_acoustic_ray");
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

    MI_DECLARE_CLASS()
protected:
    // bool m_skip_direct;
    // bool m_enable_hit_model;
    // bool m_enable_emitter_sampling;
};

MI_IMPLEMENT_CLASS_VARIANT(AcousticPathIntegrator, TimeDependentIntegrator)
MI_EXPORT_PLUGIN(AcousticPathIntegrator, "Acoustic Path Tracer integrator");
NAMESPACE_END(mitsuba)
