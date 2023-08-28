// TODO: what needed?
#include <mitsuba/render/histogram.h>
#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/profiler.h>
// #include <drjit/loop.h>
//#include <mitsuba/core/rfilter.h>

NAMESPACE_BEGIN(mitsuba)

MI_VARIANT
Histogram<Float, Spectrum>::Histogram(const ScalarVector2u &size,
                                      uint32_t channel_count,
                                      const ReconstructionFilter *rfilter,
                                      bool border)
    : m_channel_count(channel_count), m_size(0), m_offset(0), m_rfilter(rfilter), m_weights(nullptr) {

    // TODO: needed?
    // Detect if a box filter is being used, and just discard it in that case
    if (rfilter && rfilter->is_box_filter())
        m_rfilter = nullptr;

    m_border_size = (m_rfilter && border) ? m_rfilter->border_size() : 0u;

    if (m_rfilter) {
        // Temporary buffers used in put()
        size_t filter_size = dr::ceil(2 * m_rfilter->radius()) + 1;
        m_weights = new Float[filter_size];
    }

    set_size(size);
}

MI_VARIANT Histogram<Float, Spectrum>::~Histogram() {
    if (m_weights)
        delete[] m_weights;
}

MI_VARIANT void Histogram<Float, Spectrum>::clear() {
    using Array = typename TensorXf::Array;

    ScalarVector2u size_ext = m_size + 2 * ScalarVector2u(m_border_size, 0);

    size_t size_flat = m_channel_count * dr::prod(size_ext),
           shape[3]  = { size_ext.y(), size_ext.x(), m_channel_count };

    m_tensor = TensorXf(dr::zeros<Array>(size_flat), 3, shape);
    m_counts = TensorXf(dr::zeros<Array>(size_flat), 3, shape);
}

MI_VARIANT void Histogram<Float, Spectrum>::set_size(const ScalarVector2u &size) {
    using Array = typename TensorXf::Array;

    if (size == m_size)
        return;

    ScalarVector2u size_ext = size + 2 * m_border_size;

    size_t size_flat = m_channel_count * dr::prod(size_ext),
           shape[3]  = { size_ext.y(), size_ext.x(), m_channel_count };

    m_tensor = TensorXf(dr::zeros<Array>(size_flat), 3, shape);
    m_counts = TensorXf(dr::zeros<Array>(size_flat), 3, shape);

    m_size = size;
}

MI_VARIANT void Histogram<Float, Spectrum>::put_block(const Histogram *hist) { }

MI_VARIANT void
Histogram<Float, Spectrum>::put(const Point2f &pos,
                                const Float *values,
                                Mask active) {
}

MI_IMPLEMENT_CLASS_VARIANT(Histogram, Object)
MI_INSTANTIATE_CLASS(Histogram)
NAMESPACE_END(mitsuba)
