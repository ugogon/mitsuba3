#pragma once

#include <mitsuba/core/fwd.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/object.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/render/fwd.h>
#include <drjit/dynamic.h>
#include <drjit/tensor.h>

NAMESPACE_BEGIN(mitsuba)

/**
 * \brief Storage for the energy decay envelope of all recorded spectral
 * wavelengths
 *
 * This class contains all the information regarding the decay of energy per
 * wavelength. For each (discrete) time step the recorded energy can be stored.
 */

template <typename Float, typename Spectrum>
class MI_EXPORT_LIB Histogram : public Object {
public:
    MI_IMPORT_TYPES(ReconstructionFilter)

    /**
     * \brief Construct a new histogram with the specified amount of bins for wavelengths / time
     *
     * \param size
     *     number of time bins, number of wavelength bins
     * \param channel_count
     *     channel count is currently expected to always be 1
     * \param filter
     *     reconstruction filter to be applied along the time axis
     * \param border
     *     enable usage of border region for wide reconstruction filter (non box)
     */
    Histogram(const ScalarVector2u &size,
              uint32_t channel_count,
              const ReconstructionFilter *filter = nullptr,
              bool border = true);

    /**
     * Merge two histograms (simply adding all the recorded data and weights)
     */
    void put_block(const Histogram *hist);

    /**
     * \brief Accumulate a single sample or a wavefront of samples into the
     * histogram.
     *
     * \remark This variant of the put() function assumes that the histogram
     * has a standard layout, namely: \c RGB, potentially \c alpha, and a \c
     * weight channel. Use the other variant if the channel configuration
     * deviations from this default.
     *
     * \param pos
     *    Denotes the sample position in fractional pixel coordinates
     *
     * \param wavelengths
     *    Sample wavelengths in nanometers
     *
     * \param value
     *    Sample value associated with the specified wavelengths
     *
     * \param alpha
     *    Alpha value associated with the sample
     */
    void put(const Point2f &pos,
             const Wavelength &wavelengths,
             const Spectrum &value,
             Float alpha = 1.f,
             Float weight = 1.f,
             Mask active = true) {
        DRJIT_MARK_USED(wavelengths);

        UnpolarizedSpectrum spec_u = unpolarized_spectrum(value);

        Color3f rgb;
        if constexpr (is_spectral_v<Spectrum>)
            rgb = spectrum_to_srgb(spec_u, wavelengths, active);
        else if constexpr (is_monochromatic_v<Spectrum>)
            rgb = spec_u.x();
        else
            rgb = spec_u;

        Float values[5] = { rgb.x(), rgb.y(), rgb.z(), 0, 0 };

        if (m_channel_count == 4) {
            values[3] = weight;
        } else if (m_channel_count == 5) {
            values[3] = alpha;
            values[4] = weight;
        } else {
            Throw("Histogram::put(): non-standard image block configuration! (AOVs?)");
        }

        put(pos, values, active);
    }

    void put(const Point2f &pos, const Float *values, Mask active = true);

    // void read(const Point2f &pos, const Float *values, Mask active = true); // TODO

    /// Clear everything to zero.
    void clear();

    // =============================================================
    //! @{ \name Accesors
    // =============================================================

    /// Set the current hist offset.
    void set_offset(const ScalarPoint2i &offset) { m_offset = offset; }

    /// Set the block size. This potentially destroys the hist's content.
    void set_size(const ScalarVector2u &size);

    /// Return the current hist offset
    const ScalarPoint2i &offset() const { return m_offset; }

    /// return the current block size
    const ScalarVector2u &size() const { return m_size; }

    /// return the bitmap's width in pixels
    uint32_t width() const { return m_size.x(); }

    /// return the bitmap's height in pixels
    uint32_t height() const { return m_size.y(); }

    /// Return the number of channels stored by the histogram
    uint32_t channel_count() const { return m_channel_count; }

    /// Return the border region used by the reconstruction filter
    uint32_t border_size() const { return m_border_size; }

    /// Does the image block have a border region?
    bool has_border() const { return m_border_size != 0; }

    /// Return the reconstruction filter underlying the Histogram
    const ReconstructionFilter *rfilter() const { return m_rfilter; }

    /// Return the underlying spectrum buffer
    TensorXf &tensor() { return m_tensor; }

    /// Return the underlying spectrum buffer (const version)
    const TensorXf &tensor() const { const_cast<Histogram&>(*this).tensor(); }

    /// Return the underlying counts for every bin
    TensorXf &counts() { return m_counts; }

    /// Return the underlying counts for every bin (const version)
    const TensorXf &counts() const { const_cast<Histogram&>(*this).counts(); }

    //! @}
    // =============================================================

    // std::string to_string() const override; // TODO

    MI_DECLARE_CLASS()
protected:
    // Virtual destructor
    virtual ~Histogram();

    // TODO: needed?
    // Implementation detail to atomically accumulate a value into the image block
    // void accum(Float value, UInt32 index, Bool active);
protected:
    ScalarPoint2i m_offset; // TODO: needed?
    ScalarVector2u m_size;
    uint32_t m_channel_count;
    uint32_t m_border_size; // TODO: needed?
    TensorXf m_tensor;
    TensorXf m_counts;
    Float *m_weights;
    ref<const ReconstructionFilter> m_rfilter;
};

MI_EXTERN_CLASS(Histogram)
NAMESPACE_END(mitsuba)
