// TODO: what needed?
#include <mitsuba/render/histogram.h>
#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/profiler.h>
#include <drjit/loop.h>
//#include <mitsuba/core/rfilter.h>

NAMESPACE_BEGIN(mitsuba)

MI_VARIANT
Histogram<Float, Spectrum>::Histogram(const ScalarVector2u &size,
                                      size_t channel_count,
                                      const ReconstructionFilter *rfilter,
                                      bool border)
    : m_channel_count((uint32_t) channel_count), m_size(0), m_offset(0), m_rfilter(rfilter), m_weights(nullptr) {

    // TODO: needed?
    // Detect if a box filter is being used, and just discard it in that case
    if (rfilter && rfilter->is_box_filter())
        m_rfilter = nullptr;

    m_border_size = (m_rfilter && border) ? m_rfilter->border_size() : 0u;

    if (m_rfilter) {
        // Temporary buffers used in put()
        int filter_size = (int) std::ceil(2 * m_rfilter->radius()) + 1;
        m_weights = new Float[filter_size];
    }

    set_size(size);
}

MI_VARIANT
Histogram<Float, Spectrum>::~Histogram() {
    if (m_weights)
        delete[] m_weights;
}

MI_VARIANT void
Histogram<Float, Spectrum>::clear() {
    size_t total_size = m_channel_count * dr::prod(m_size + 2 * ScalarVector2i(m_border_size, 0));

    if constexpr (!dr::is_jit_v<Float>) {
        // TODO needed?
        NotImplementedError("Histogram::clear Scalar case");
    } else {
        m_data = dr::zeros<TensorXf>(total_size);
        m_counts = dr::zeros<TensorXf>(total_size);
    }
}

MI_VARIANT void
Histogram<Float, Spectrum>::set_size(const ScalarVector2u &size) {
    if (size == m_size)
        return;
    m_size = size;

    size_t total_size = m_channel_count * dr::prod(m_size + 2 * ScalarVector2i(m_border_size, 0));

    // Allocate empty buffer
    // TODO does this work properly?
    m_data = dr::zeros<TensorXf>(total_size);
    m_counts = dr::zeros<TensorXf>(total_size);
}

MI_IMPLEMENT_CLASS_VARIANT(Histogram, Object)
MI_INSTANTIATE_CLASS(Histogram)
NAMESPACE_END(mitsuba)
