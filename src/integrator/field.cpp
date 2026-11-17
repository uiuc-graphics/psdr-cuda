#include <misc/Exception.h>
#include <psdr/core/ray.h>
#include <psdr/core/intersection.h>
#include <psdr/scene/scene.h>
#include <psdr/integrator/field.h>
#include <psdr/bsdf/bsdf.h>
#include <psdr/shape/mesh.h>

NAMESPACE_BEGIN(psdr_jit)

FieldExtractionIntegrator::FieldExtractionIntegrator(const char *field) {
    std::istringstream fields(field);
    std::string field_name;
    fields >> field_name;
    PSDR_ASSERT_MSG(
        field_name == "bsdf" ||
        field_name == "segmentation" ||
        field_name == "silhouette" ||
        field_name == "position"   ||
        field_name == "depth"      ||
        field_name == "geoNormal"  ||
        field_name == "shNormal"   ||
        field_name == "uv",
        "Unsupported field: " + field_name
    );
    m_field = field_name;

    fields >> m_object;

}


SpectrumC FieldExtractionIntegrator::Li(const Scene &scene, Sampler &sampler, const RayC &ray, MaskC active) const {
    return __Li<false>(scene, ray, active);
}


SpectrumD FieldExtractionIntegrator::Li(const Scene &scene, Sampler &sampler, const RayD &ray, MaskD active) const {
    return __Li<true>(scene, ray, active);
}


template <bool ad>
Spectrum<ad> FieldExtractionIntegrator::__Li(const Scene &scene, const Ray<ad> &ray, Mask<ad> active) const {
    Vector3f<ad> result;
    Intersection<ad> its = scene.ray_intersect<ad>(ray);

    BSDFArray<ad> bsdf_array = its.shape->bsdf();
    if ( scene.m_emitter_env != nullptr ) {
        // Skip reflectance computations for intersections on the bounding mesh
        active &= neq(bsdf_array, nullptr);
    }
    Mask<ad> valid_obj(1);
    if (m_object != "") {
        if constexpr ( !ad ) { 
            valid_obj = its.shape->get_obj_mask(m_object);
        } else {
            valid_obj = detach<false>(its.shape)->get_obj_mask(m_object);
        }
    }


    if ( m_field == "segmentation" ) {
        if constexpr ( !ad ) { 
            IntC rresult = (its.shape)->get_obj_id();
            result = Vector3f<ad>(rresult,rresult,rresult);
        } else {
            IntC rresult = detach<false>(its.shape)->get_obj_id();
            result = Vector3f<ad>(rresult,rresult,rresult);
        }
    } else if ( m_field == "bsdf" ) {
        if constexpr (ad) {
            active &= its.is_valid();
            if ( scene.m_bsdfs.size() == 1U || scene.m_meshes.size() == 1U ) {
                const BSDF *bsdf = scene.m_meshes[0]->m_bsdf;
                result = bsdf->evalD(its, its.wi, active);
            } else {
                BSDFArray<ad> bsdf_array = its.shape->bsdf();
                
                result = bsdf_array->evalD(its, its.wi, active);
            }
        } else {
            active &= its.is_valid();
            if ( scene.m_bsdfs.size() == 1U || scene.m_meshes.size() == 1U ) {
                const BSDF *bsdf = scene.m_meshes[0]->m_bsdf;
                result = bsdf->evalC(its, its.wi, active);
            } else {
                BSDFArray<ad> bsdf_array = its.shape->bsdf();
                
                result = bsdf_array->evalC(its, its.wi, active);
            }
        }

    } else if ( m_field == "silhouette" ) {
        result = full<Spectrum<ad>>(1.f);
    } else if ( m_field == "position" ) {
        result = its.p;
    } else if ( m_field == "depth" ) {
        result = its.t;
    } else if ( m_field == "geoNormal" ) {
        result = its.n;
    } else if ( m_field == "shNormal" ) {
        result = its.sh_frame.n;
    } else if ( m_field == "uv" ) {
        result = Vector3f<ad>(its.uv[0], its.uv[1], 0.f);
    } else {
        PSDR_ASSERT(0);
    }

    return result & (active && (its.is_valid()) && valid_obj);

}





NAMESPACE_END(psdr_jit)

