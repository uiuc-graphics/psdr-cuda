#pragma once

#include <memory>
#include "integrator.h"

NAMESPACE_BEGIN(psdr_jit)

PSDR_CLASS_DECL_BEGIN(DirectIntegrator, final, Integrator)
public:
    DirectIntegrator(int mis = 1);
    virtual ~DirectIntegrator();

    bool m_hide_emitters = false; 
    void preprocess_secondary_edges(const Scene &scene, int sensor_id, const ScalarVector4i &reso, int nrounds = 1, int seed = 0);

protected:
    SpectrumC Li(const Scene &scene, Sampler &sampler, const RayC &ray, MaskC active = true) const override;
    SpectrumD Li(const Scene &scene, Sampler &sampler, const RayD &ray, MaskD active = true) const override;

    template <bool ad>
    Spectrum<ad> __Li(const Scene &scene, Sampler &sampler, const Ray<ad> &ray, Mask<ad> active) const;

    void render_secondary_edges(const Scene &scene, int sensor_id, SpectrumD &result) const override;


    template <bool ad>
    std::pair<IntC, Spectrum<ad>> eval_secondary_edge(const Scene &scene, const Sensor &sensor, const Vector3fC &sample3) const;

    int m_mis;

    std::vector<std::shared_ptr<const HyperCubeDistribution3f>> m_guides;

PSDR_CLASS_DECL_END(PathTracer)

NAMESPACE_END(psdr_jit)
