#pragma once

#include "integrator.h"

NAMESPACE_BEGIN(psdr_jit)

PSDR_CLASS_DECL_BEGIN(CollocatedIntegrator, final, Integrator)
public:
    CollocatedIntegrator(float intensity) : m_intensity(FloatD(intensity)) {}
    CollocatedIntegrator(const FloatD &intensity) : m_intensity(intensity) {}

    FloatD m_intensity;

protected:
    SpectrumC Li(const Scene &scene, Sampler &sampler, const RayC &ray, MaskC active = true) const override;
    SpectrumD Li(const Scene &scene, Sampler &sampler, const RayD &ray, MaskD active = true) const override;

    template <bool ad>
    Spectrum<ad> __Li(const Scene &scene, const Ray<ad> &ray, Mask<ad> active) const;
PSDR_CLASS_DECL_END(CollocatedIntegrator)

NAMESPACE_END(psdr_jit)
