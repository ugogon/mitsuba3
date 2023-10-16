#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/render/fresnel.h>

/*
 * Alternatively use blendbsf together with diffuse bsdf, conductor bsdf and weight set to scattering
*/

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class AcousticBSDF final : public BSDF<Float, Spectrum> {
public:
    MI_IMPORT_BASE(BSDF, m_flags, m_components)
    MI_IMPORT_TYPES(Texture)

    AcousticBSDF(const Properties &props) : Base(props) {
        m_scatter = props.texture<Texture>("scattering");
        m_absorpt = props.texture<Texture>("absorption");
        m_flags = BSDFFlags::DeltaReflection | BSDFFlags::DiffuseReflection | BSDFFlags::FrontSide;
        dr::set_attr(this, "flags", m_flags);
        m_components.push_back(m_flags);
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_object("scattering", m_scatter.get(), +ParamFlags::Differentiable);
        callback->put_object("absorption", m_absorpt.get(), +ParamFlags::Differentiable);
    }

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext & /* ctx */,
                                             const SurfaceInteraction3f &si,
                                             Float sample1,
                                             const Point2f &sample2,
                                             Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        // Specular component
        auto bs_specular = dr::zeros<BSDFSample3f>();
        bs_specular.wo  = reflect(si.wi);
        bs_specular.pdf = 1.f;
        bs_specular.eta = 1.f;
        bs_specular.sampled_type = +BSDFFlags::DeltaReflection;
        bs_specular.sampled_component = 0;

        // Diffuse component
        auto bs_diffuse = dr::zeros<BSDFSample3f>();
        bs_diffuse.wo = warp::square_to_cosine_hemisphere(sample2);
        bs_diffuse.pdf = warp::square_to_cosine_hemisphere_pdf(bs_diffuse.wo);
        bs_diffuse.eta = 1.f;
        bs_diffuse.sampled_type = +BSDFFlags::DiffuseReflection;
        bs_diffuse.sampled_component = 0;

        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        active &= cos_theta_i > 0.f;

        Float scatter = dr::clamp(m_scatter->eval(si, active).x(), 0.f, 1.f);
        Mask scatter_sample = sample1 > scatter;

        BSDFSample3f bs = dr::zeros<BSDFSample3f>();
        dr::masked(bs, active && scatter_sample)  = bs_specular;
        dr::masked(bs, active && !scatter_sample) = bs_diffuse;

        UnpolarizedSpectrum reflectance = 1.f - m_absorpt->eval(si, active);

        return { bs, reflectance & (active && bs.pdf > 0.f) };
    }

    Spectrum eval(const BSDFContext & /* ctx */, const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);
        active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

        Float scatter = dr::clamp(m_scatter->eval(si, active).x(), 0.f, 1.f);

        UnpolarizedSpectrum reflectance = 1.f - m_absorpt->eval(si, active);
        UnpolarizedSpectrum value_diffuse  = reflectance * dr::InvPi<Float> * cos_theta_o;

        return (value_diffuse * scatter) & active;
    }

    Float pdf(const BSDFContext & /* ctx */, const SurfaceInteraction3f &si,
              const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

        Float scatter = dr::clamp(m_scatter->eval(si, active).x(), 0.f, 1.f);
        Float pdf_diffuse = warp::square_to_cosine_hemisphere_pdf(wo);

        return dr::select(active, pdf_diffuse * scatter, 0.f);
    }

    std::pair<Spectrum, Float> eval_pdf(const BSDFContext & /* ctx */,
                                        const SurfaceInteraction3f &si,
                                        const Vector3f &wo,
                                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

        Float scatter = dr::clamp(m_scatter->eval(si, active).x(), 0.f, 1.f);

        Float pdf_diffuse = warp::square_to_cosine_hemisphere_pdf(wo);

        UnpolarizedSpectrum reflectance = 1.f - m_absorpt->eval(si, active);
        UnpolarizedSpectrum value_diffuse  = reflectance * dr::InvPi<Float> * cos_theta_o;

        return {
            (value_diffuse * scatter) & active,
            dr::select(active, pdf_diffuse * scatter, 0.f)
        };
    }

    Spectrum eval_diffuse_reflectance(const SurfaceInteraction3f &si,
                                      Mask active) const override {
        return 1.f - m_absorpt->eval(si, active);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "AcousticBSDF[" << std::endl
            << "  absorption = " << string::indent(m_absorpt) << "," << std::endl
            << "  scattering = " << string::indent(m_scatter) << "," << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    ref<Texture> m_scatter;
    ref<Texture> m_absorpt;
};

MI_IMPLEMENT_CLASS_VARIANT(AcousticBSDF, BSDF)
MI_EXPORT_PLUGIN(AcousticBSDF, "BSDF of Sound")
NAMESPACE_END(mitsuba)
