#pragma once

// TODO: what needed?
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
              size_t channel_count,
              const ReconstructionFilter *filter = nullptr,
              bool border = true);

    /// Clear everything to zero.
    void clear();

    // =============================================================
    //! @{ \name Accesors
    // =============================================================

    /// Set the current hist offset.
    void set_offset(const ScalarPoint2i &offset) { m_offset = offset; }

    /// Set the block size. This potentially destroys the hist's content.
    void set_size(const ScalarVector2u &size);

    /// Return the current histogram size
    const ScalarVector2u &size() const { return m_size; }

    /// Return the width (time bins)
    size_t width() const { return m_size.x(); }

    /// Return the height (wav bins)
    size_t height() const { return m_size.y(); }

    /// Return the number of channels stored by the histogram
    size_t channel_count() const { return (size_t) m_channel_count; }

    /// Return the border region used by the reconstruction filter
    uint32_t border_size() const { return m_border_size; }

    /// Return the current hist offset
    const ScalarPoint2i &offset() const { return m_offset; }

    /// Return the underlying spectrum buffer
    TensorXf &data() { return m_data; }

    /// Return the underlying spectrum buffer (const version)
    const TensorXf &data() const { return m_data; }

    /// Return the underlying counts for every bin
    TensorXf &counts() { return m_counts; }

    /// Return the underlying counts for every bin (const version)
    const TensorXf &counts() const { return m_counts; }

    //! @}
    // =============================================================

    // std::string to_string() const override;

    MI_DECLARE_CLASS()

protected:
    // Virtual destructor
    virtual ~Histogram();

protected:
    size_t m_channel_count;
    ScalarVector2u m_size;
    ScalarPoint2i m_offset;
    uint32_t m_border_size;
    TensorXf m_data;
    TensorXf m_counts;
    ref<const ReconstructionFilter> m_rfilter;
    Float *m_weights;
};

MI_EXTERN_CLASS(Histogram)
NAMESPACE_END(mitsuba)
