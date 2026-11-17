#pragma once

#include <psdr/psdr.h>
#include <psdr/core/records.h>

NAMESPACE_BEGIN(psdr_jit)

PSDR_CLASS_DECL_BEGIN(Emitter,, Object)
public:
    Emitter() : Object("Emitter") {}
    virtual ~Emitter() override {}

    virtual void configure() = 0;

    // Returns the emitted radiance at its.p in direction its.wi

    virtual SpectrumC eval(const IntersectionC &its, MaskC active = true) const = 0;
    virtual SpectrumD eval(const IntersectionD &its, MaskD active = true) const = 0;

    SpectrumC evalC(const IntersectionC &its, MaskC active = true) const {
        return eval(its, active);
    };
    SpectrumD evalD(const IntersectionD &its, MaskD active = true) const {
        return eval(its, active);
    };

    virtual PositionSampleC sample_position(const Vector3fC &ref_p, const Vector2fC &sample2, MaskC active = true) const = 0;
    virtual PositionSampleD sample_position(const Vector3fD &ref_p, const Vector2fD &sample2, MaskD active = true) const = 0;

    PositionSampleC sample_positionC(const Vector3fC &ref_p, const Vector2fC &sample2, MaskC active = true) const {
        return sample_position(ref_p, sample2, active);
    };
    PositionSampleD sample_positionD(const Vector3fD &ref_p, const Vector2fD &sample2, MaskD active = true) const {
        return sample_position(ref_p, sample2, active);
    };


    virtual FloatC sample_position_pdf(const Vector3fC &ref_p, const IntersectionC &its, MaskC active = true) const = 0;
    virtual FloatD sample_position_pdf(const Vector3fD &ref_p, const IntersectionD &its, MaskD active = true) const = 0;

    FloatC sample_position_pdfC(const Vector3fC &ref_p, const IntersectionC &its, MaskC active = true) const {
        return sample_position_pdf(ref_p, its, active);
    };
    FloatD sample_position_pdfD(const Vector3fD &ref_p, const IntersectionD &its, MaskD active = true) const {
        return sample_position_pdf(ref_p, its, active);
    };

    bool m_ready = false;
    float m_sampling_weight = 1.f;

    static constexpr const char *Domain = "Emitter";

PSDR_CLASS_DECL_END(Emitter)

NAMESPACE_END(psdr_jit)

DRJIT_CALL_BEGIN(psdr_jit::Emitter)
    DRJIT_CALL_METHOD(eval)
    DRJIT_CALL_METHOD(evalC)
    DRJIT_CALL_METHOD(evalD)
    DRJIT_CALL_METHOD(sample_position)
    DRJIT_CALL_METHOD(sample_positionC)
    DRJIT_CALL_METHOD(sample_positionD)
    DRJIT_CALL_METHOD(sample_position_pdf)
    DRJIT_CALL_METHOD(sample_position_pdfC)
    DRJIT_CALL_METHOD(sample_position_pdfD)
DRJIT_CALL_END()
