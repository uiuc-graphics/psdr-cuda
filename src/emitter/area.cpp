#include <misc/Exception.h>
#include <psdr/core/frame.h>
#include <psdr/core/intersection.h>
#include <psdr/shape/mesh.h>
#include <psdr/emitter/area.h>

NAMESPACE_BEGIN(psdr_jit)

void AreaLight::configure() {
    PSDR_ASSERT((m_mesh != nullptr) && m_mesh->m_ready);
    m_sampling_weight = m_mesh->m_total_area*
        rgb2luminance<false>(detach<false>(m_radiance))[0];
    m_ready = true;
}


SpectrumC AreaLight::eval(const IntersectionC &its, MaskC active) const {
    PSDR_ASSERT(m_ready);
    return select(active && FrameC::cos_theta(its.wi) > 0.f, detach<false>(m_radiance), 0.f);
}


SpectrumD AreaLight::eval(const IntersectionD &its, MaskD active) const {
    PSDR_ASSERT(m_ready);
    return select(active && FrameD::cos_theta(its.wi) > 0.f, m_radiance, 0.f);
}


PositionSampleC AreaLight::sample_position(const Vector3fC &ref_p, const Vector2fC &sample2, MaskC active) const {
    return __sample_position<false>(sample2, active);
}


PositionSampleD AreaLight::sample_position(const Vector3fD &ref_p, const Vector2fD &sample2, MaskD active) const {
    return __sample_position<true >(sample2, active);
}


template <bool ad>
PositionSample<ad> AreaLight::__sample_position(const Vector2f<ad> &sample2, Mask<ad> active) const {
    PSDR_ASSERT(m_ready);
    // PositionSample<ad> rs = m_mesh->sample_position(sample2, active);
    // return rs;
    return m_mesh->sample_position(sample2, active);
}


FloatC AreaLight::sample_position_pdf(const Vector3fC &ref_p, const IntersectionC &its, MaskC active) const {
    return m_sampling_weight*its.shape->sample_position_pdf(its, active);
}


FloatD AreaLight::sample_position_pdf(const Vector3fD &ref_p, const IntersectionD &its, MaskD active) const {

    // FloatD temp = its.shape->sample_position_pdf(its, active);
    return m_sampling_weight*its.shape->sample_position_pdfD(its, active);
    // return m_sampling_weight*its.shape->sample_position_pdf(its, active);
    // return __sample_position_pdf<true>(ref_p, its, active);
}


template <bool ad>
Float<ad> AreaLight::__sample_position_pdf(const Vector3f<ad> &ref_p, const Intersection<ad> &its, Mask<ad> active) const {
    // return 1.f;

    // auto temp = its.shape->sample_position_pdf(its, active);
    return 1.f;
    // Float<ad> rs;
    // if constexpr (!ad) {
    //     rs = m_sampling_weight*its.shape->sample_position_pdf(detach(its), detach(active));
    // } else {
    //     rs = m_sampling_weight*its.shape->sample_position_pdf(its, active);
    // }
    // // Float<ad> rs = m_sampling_weight*its.shape->sample_position_pdf(its, active);
    // return rs;
}


std::string AreaLight::to_string() const {
    std::ostringstream oss;
    std::operator<<(oss, "AreaLight[radiance = ");
    oss << m_radiance;
    std::operator<<(oss, ", sampling_weight = ");
    oss << m_sampling_weight;
    std::operator<<(oss, "]");
    return oss.str();
}

NAMESPACE_END(psdr_jit)
