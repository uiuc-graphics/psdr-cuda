#include <misc/Exception.h>
#include <psdr/core/warp.h>
#include <psdr/bsdf/microfacet_pv.h>
#include <psdr/bsdf/ggx.h>

NAMESPACE_BEGIN(psdr_jit)

MicrofacetPerVertex::MicrofacetPerVertex(const Vector3fD &spec_refl, const Vector3fD &diff_refl, const Vector1fD &roughness) :
    m_specularReflectance(spec_refl), m_diffuseReflectance(diff_refl), m_roughness(roughness) {}

SpectrumC MicrofacetPerVertex::eval(const IntersectionC &its, const Vector3fC &wo, MaskC active) const {
    return __eval<false>(its, wo, active);
}

SpectrumD MicrofacetPerVertex::eval(const IntersectionD &its, const Vector3fD &wo, MaskD active) const {
    return __eval<true>(its, wo, active);
}

template <bool ad>
Spectrum<ad> MicrofacetPerVertex::__eval(const Intersection<ad> &_its, const Vector3f<ad> &_wo, Mask<ad> active) const {
    Intersection<ad> its(_its);
    Vector3f<ad> wo(_wo);

    if (m_twoSide) {
        wo.z() = drjit::mulsign(wo.z(), its.wi.z());
        its.wi.z() = drjit::abs(its.wi.z());
    }

    Vector3f<ad> specularReflectance = __interpolate<3, ad>(its, m_specularReflectance, active);
    Vector3f<ad> diffuseReflectance = __interpolate<3, ad>(its, m_diffuseReflectance, active);
    Float<ad> roughness = __interpolate<1, ad>(its, m_roughness, active)[0];

    Float<ad> cos_theta_nv = Frame<ad>::cos_theta(its.wi),// view dir
              cos_theta_nl = Frame<ad>::cos_theta(wo);    // light dir

    active &= (cos_theta_nv > 0.f && cos_theta_nl > 0.f);

    Spectrum<ad> diffuse = diffuseReflectance * InvPi;

    Vector3f<ad> H = normalize(its.wi + wo);
    Float<ad> cos_theta_nh = Frame<ad>::cos_theta(H),
              cos_theta_vh = dot(H, its.wi);

    Spectrum<ad> F0 = specularReflectance;
    Float<ad> alpha = sqr(roughness);
    Float<ad> k = sqr(roughness + 1.f) / 8.f;

    // GGX NDF term
    Float<ad> tmp = alpha / (cos_theta_nh * cos_theta_nh * (sqr(alpha) - 1.f) + 1.f);
    Float<ad> ggx = tmp * tmp * InvPi;

    // Fresnel term
    Float<ad> coeff = cos_theta_vh * (-5.55473f * cos_theta_vh - 6.8316f);
    Spectrum<ad> fresnel = F0 + (1.f - F0) * pow(2.f, coeff);

    // Geometry term
    Float<ad> smithG1 = cos_theta_nv / (cos_theta_nv * (1.f - k) + k);
    Float<ad> smithG2 = cos_theta_nl / (cos_theta_nl * (1.f - k) + k);
    Float<ad> smithG = smithG1 * smithG2;

    Spectrum<ad> numerator = ggx * smithG * fresnel;
    Float<ad> denominator = 4.f * cos_theta_nl * cos_theta_nv;
    Spectrum<ad> specular = numerator / (denominator + 1e-6f);

    Spectrum<ad> value = (diffuse + specular) * cos_theta_nl;

    return select(active, value, 0.f);
}


BSDFSampleC MicrofacetPerVertex::sample(const IntersectionC &its, const Vector3fC &sample, MaskC active) const {
    return __sample<false>(its, sample, active);
}


BSDFSampleD MicrofacetPerVertex::sample(const IntersectionD &its, const Vector3fD &sample, MaskD active) const {
    return __sample<true>(its, sample, active);
}

template <bool ad>
BSDFSample<ad> MicrofacetPerVertex::__sample(const Intersection<ad>& _its, const Vector3f<ad>& sample, Mask<ad> active) const {
    Intersection<ad> its(_its);

    if (m_twoSide) {
        its.wi.z() = drjit::abs(its.wi.z());
    }

    Float<ad> roughness = __interpolate<1, ad>(its, m_roughness, active)[0];

    BSDFSample<ad> bs;
    Float<ad> cos_theta_i = Frame<ad>::cos_theta(its.wi);
    Float<ad> alpha = sqr(roughness);
    GGXDistribution distr(alpha);

    auto m_pair = distr.sample<ad>(its.wi, sample);
    Vector3f<ad> m = std::get<0>(m_pair);
    Float<ad> m_pdf = std::get<1>(m_pair);
    bs.wo = fmsub(Vector3f<ad>(m), 2.f * dot(its.wi, m), its.wi);
    bs.eta = 1.0f; // TODO: Fix later
    bs.pdf = (m_pdf / (4.f * dot(bs.wo, m)));
    bs.is_valid = (cos_theta_i > 0.f && neq(bs.pdf, 0.f) && Frame<ad>::cos_theta(bs.wo) > 0.f) & active;
    return detach(bs);
}


FloatC MicrofacetPerVertex::pdf(const IntersectionC &its, const Vector3fC &wo, MaskC active) const {
    return __pdf<false>(its, wo, active);
}


FloatD MicrofacetPerVertex::pdf(const IntersectionD &its, const Vector3fD &wo, MaskD active) const {
    return __pdf<true>(its, wo, active);
}

template <bool ad>
Float<ad> MicrofacetPerVertex::__pdf(const Intersection<ad> &_its, const Vector3f<ad> &_wo, Mask<ad> active) const {

    Intersection<ad> its(_its);
    Vector3f<ad> wo(_wo);

    if (m_twoSide) {
        wo.z() = drjit::mulsign(wo.z(), its.wi.z());
        its.wi.z() = drjit::abs(its.wi.z());
    }

    Float<ad> roughness = __interpolate<1, ad>(its, m_roughness, active)[0];

    Float<ad> cos_theta_i = Frame<ad>::cos_theta(its.wi),
              cos_theta_o = Frame<ad>::cos_theta(wo);

    Spectrum<ad> m = normalize(wo + its.wi);
    
    active &= cos_theta_i > 0.f && cos_theta_o > 0.f &&
              dot(its.wi, m) > 0.f && dot(wo, m) > 0.f;

    Float<ad> alpha = sqr(roughness);
    GGXDistribution distr(alpha);

    Float<ad> result = (distr.eval<ad>(m) * distr.smith_g1<ad>(its.wi, m) /
                       (4.f * cos_theta_i));
    return detach(result);
}

template <int n, bool ad>
Vectorf<n, ad> MicrofacetPerVertex::__interpolate(const Intersection<ad> &its, const Vectorf<n, true> &v, Mask<ad> active) const {
    Vectorf<n, ad> v_;
    if constexpr ( ad ) {
        v_ = v;
    }
    else {
        v_ = detach<false>(v);
    }
    auto v0 = gather<Vectorf<n, ad>>(v_, its.face_indices[0], active);
    auto v1 = gather<Vectorf<n, ad>>(v_, its.face_indices[1], active);
    auto v2 = gather<Vectorf<n, ad>>(v_, its.face_indices[2], active);

    // Bilinear interpolation.
    auto result = fmadd(v1 - v0, its.bc.x(), fmadd(v2 - v0, its.bc.y(), v0));
    return result;
}

NAMESPACE_END(psdr_jit)
