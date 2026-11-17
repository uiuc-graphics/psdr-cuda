#pragma once

#include "integrator.h"

NAMESPACE_BEGIN(psdr_jit)

PSDR_CLASS_DECL_BEGIN(FieldExtractionIntegrator, final, Integrator)
public:
    FieldExtractionIntegrator(const char *field);

    std::string m_field;
    std::string m_object;

protected:
    SpectrumC Li(const Scene &scene, Sampler &sampler, const RayC &ray, MaskC active = true) const override;
    SpectrumD Li(const Scene &scene, Sampler &sampler, const RayD &ray, MaskD active = true) const override;

    template <bool ad>
    Spectrum<ad> __Li(const Scene &scene, const Ray<ad> &ray, Mask<ad> active) const;
PSDR_CLASS_DECL_END(FieldExtractionIntegrator)

NAMESPACE_END(psdr_jit)
