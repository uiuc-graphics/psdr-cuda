#include <psdr/core/ray.h>
#include <psdr/core/intersection.h>
#include <psdr/bsdf/bsdf.h>
#include <psdr/emitter/emitter.h>
#include <psdr/shape/mesh.h>
#include <psdr/scene/scene.h>
#include <psdr/integrator/collocated.h>

NAMESPACE_BEGIN(psdr_jit)

SpectrumC CollocatedIntegrator::Li(const Scene &scene, Sampler &sampler, const RayC &ray, MaskC active) const {
    return __Li<false>(scene, ray, active);
}


SpectrumD CollocatedIntegrator::Li(const Scene &scene, Sampler &sampler, const RayD &ray, MaskD active) const {
    return __Li<true>(scene, ray, active);
}


template <bool ad>
Spectrum<ad> CollocatedIntegrator::__Li(const Scene &scene, const Ray<ad> &ray, Mask<ad> active) const {
    Intersection<ad> its = scene.ray_intersect<ad>(ray, active);
    Spectrum<ad> result;
    // std::cout << "here" << std::endl;
    // return its.p;

    if constexpr (ad) {
        active &= its.is_valid();
        if ( scene.m_bsdfs.size() == 1U || scene.m_meshes.size() == 1U ) {
            const BSDF *bsdf = scene.m_meshes[0]->m_bsdf;
            result = bsdf->evalD(its, its.wi, active)/sqr(its.t);
        } else {
            BSDFArray<ad> bsdf_array = its.shape->bsdf();
            
            result = bsdf_array->evalD(its, its.wi, active)/sqr(its.t);
        }
        result *= m_intensity;
    } else {
        active &= its.is_valid();
        if ( scene.m_bsdfs.size() == 1U || scene.m_meshes.size() == 1U ) {
            const BSDF *bsdf = scene.m_meshes[0]->m_bsdf;
            result = bsdf->evalC(its, its.wi, active)/sqr(its.t);
        } else {
            BSDFArray<ad> bsdf_array = its.shape->bsdf();
            
            result = bsdf_array->evalC(its, its.wi, active)/sqr(its.t);
        }
        result *= detach<false>(m_intensity);
    }

    // result[!active] = Spectrum<ad>(1.0);

    // Misses may carry a zero distance: mask after division so 0/0 does not
    // erase silhouette contributions when edge rays straddle the mesh.
    return select(active, result, 0.f);
}

NAMESPACE_END(psdr_jit)
