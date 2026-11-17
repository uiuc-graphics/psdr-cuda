#include <psdr/core/warp.h>
#include <psdr/bsdf/roughdielectric.h>
#include <psdr/bsdf/ggx.h>

NAMESPACE_BEGIN(psdr_jit)

SpectrumC RoughDielectric::eval(const IntersectionC& its, const Vector3fC& wo, MaskC active) const {
    return __eval<false>(its, wo, active);
}


SpectrumD RoughDielectric::eval(const IntersectionD& its, const Vector3fD& wo, MaskD active) const {
    return __eval<true>(its, wo, active);
}


BSDFSampleC RoughDielectric::sample(const IntersectionC& its, const Vector3fC& sample, MaskC active) const {
    return __sample<false>(its, sample, active);
}


BSDFSampleD RoughDielectric::sample(const IntersectionD& its, const Vector3fD& sample, MaskD active) const {
    return __sample<true>(its, sample, active);
}

FloatC RoughDielectric::pdf(const IntersectionC& its, const Vector3fC& wo, MaskC active) const {
    return __pdf<false>(its, wo, active);
}


FloatD RoughDielectric::pdf(const IntersectionD& its, const Vector3fD& wo, MaskD active) const {
    return __pdf<true>(its, wo, active);
}


template <bool ad>
Spectrum<ad> RoughDielectric::__eval(const Intersection<ad>& _its, const Vector3f<ad>& _wo, Mask<ad> active) const {
    Intersection<ad> its(_its);
    Vector3f<ad> wo(_wo);

    if (m_twoSide) {
        wo.z() = drjit::mulsign(wo.z(), its.wi.z());
        its.wi.z() = drjit::abs(its.wi.z());
    }
    Float<ad> cos_theta_i = Frame<ad>::cos_theta(its.wi),
              cos_theta_o = Frame<ad>::cos_theta(wo);

    // Ignore perfectly grazing configurations
    active &= neq(cos_theta_i, 0.f);

    // Determine the type of interaction
    bool has_reflection   = true,
         has_transmission = true;

    Mask<ad> reflect = cos_theta_i * cos_theta_o > 0.f;

    // Determine the relative index of refraction
    Float<ad>    eta, inv_eta;
    if constexpr (ad) {
        eta     = select(cos_theta_i > 0.f, m_eta, m_inv_eta);
        inv_eta = select(cos_theta_i > 0.f, m_inv_eta, m_eta);
    } else {
        eta     = select(cos_theta_i > 0.f, detach<false>(m_eta), detach<false>(m_inv_eta));
        inv_eta = select(cos_theta_i > 0.f, detach<false>(m_inv_eta), detach<false>(m_eta));
    }

    // Compute the half-vector
    Vector3f<ad> m = normalize(its.wi + wo * select(reflect, 1.f, eta));

    // Ensure that the half-vector points into the same hemisphere as the macrosurface normal
    m = mulsign(m, Frame<ad>::cos_theta(m));


    /* Construct the microfacet distribution matching the
           roughness values at the current surface position. */
    Float<ad> alpha_u = m_alpha_u.eval<ad>(its.uv);
    Float<ad> alpha_v = m_alpha_v.eval<ad>(its.uv);
    GGXDistribution distr(alpha_u, alpha_v);

    // Evaluate the microfacet normal distribution
    Float<ad> D = distr.eval<ad>(m);

    // Fresnel factor
    Float<ad> F;
    if constexpr (ad) {
        F = std::get<0>(fresnel_dielectric<ad>(m_eta, dot(its.wi, m)));
    } else {
        F = std::get<0>(fresnel_dielectric<ad>(detach<false>(m_eta), dot(its.wi, m)));
    }

    // Smith's shadow-masking function
    Float<ad> G = distr.G<ad>(its.wi, wo, m);


    Spectrum<ad> result = 0.0f;

    Mask<ad> eval_r = Mask<ad>(has_reflection) && reflect && active,
             eval_t = Mask<ad>(has_transmission) && !reflect && active;

    if (any_or<true>(eval_r)) {
        Spectrum<ad> value = F * D * G / (4.f * abs(cos_theta_i));
        // TODO: add m_specular_reflectance eval
        result[eval_r] = value;
    }

    if (any_or<true>(eval_t)) {
        /* Missing term in the original paper: account for the solid angle
           compression when tracing radiance -- this is necessary for
           bidirectional methods. */
        Float<ad> scale = sqr(inv_eta);
        // Float<ad> scale(1.f);

        // Compute the total amount of transmission
        Spectrum<ad> value = abs(
            (scale * (1.f - F) * D * G * eta * eta * dot(its.wi, m) * dot(wo, m)) /
            (cos_theta_i * sqr(dot(its.wi, m) + eta * dot(wo, m))));

        result[eval_t] = value;
    }
    return result;
}


template <bool ad>
Float<ad> RoughDielectric::__pdf(const Intersection<ad>& _its, const Vector3f<ad>& _wo, Mask<ad> active) const {

    Intersection<ad> its(_its);
    Vector3f<ad> wo(_wo);

    if (m_twoSide) {
        wo.z() = drjit::mulsign(wo.z(), its.wi.z());
        its.wi.z() = drjit::abs(its.wi.z());
    }

    Float<ad> cos_theta_i = Frame<ad>::cos_theta(its.wi),
              cos_theta_o = Frame<ad>::cos_theta(wo);

    active &= neq(cos_theta_i, 0.f);
    Mask<ad> reflect = cos_theta_i * cos_theta_o > 0.f;



    Float<ad> eta;
    if constexpr (ad) {
        eta = select(cos_theta_i > 0.f, m_eta, m_inv_eta);
    } else {
        eta = select(cos_theta_i > 0.f, detach<false>(m_eta), detach<false>(m_inv_eta));
    }

    Vector3f<ad> m = normalize(its.wi + wo * select(reflect, 1.f, eta));
                 m = mulsign(m, Frame<ad>::cos_theta(m));
    active &= dot(its.wi, m) * Frame<ad>::cos_theta(its.wi) > 0.f &&
              dot(wo,    m) * Frame<ad>::cos_theta(wo)    > 0.f;

    Float<ad> dwh_dwo = select(reflect, rcp(4.f * dot(wo, m)),
                                (eta * eta * dot(wo, m)) /
                                sqr(dot(its.wi, m) + eta * dot(wo, m)));
    Float<ad> alpha_u = m_alpha_u.eval<ad>(its.uv);
    Float<ad> alpha_v = m_alpha_v.eval<ad>(its.uv);
    GGXDistribution distr(alpha_u, alpha_v);
    Vector3f<ad> pwi = mulsign(its.wi, Frame<ad>::cos_theta(its.wi));
    Float<ad> prob = distr.eval<ad>(m) * distr.smith_g1<ad>(pwi, m) / Frame<ad>::cos_theta(pwi);

    Float<ad> mmeta;
    if constexpr (ad) {
        mmeta = m_eta;
    } else {
        mmeta = detach<false>(m_eta);
    }

    Float<ad> F = std::get<0>(fresnel_dielectric<ad>(mmeta, dot(its.wi, m)));
    prob *= select(reflect, F, 1.f - F);

    return select(active, prob * abs(dwh_dwo), 0.f);
}


template <bool ad>
BSDFSample<ad> RoughDielectric::__sample(const Intersection<ad>& _its, const Vector3f<ad>& sample, Mask<ad> active) const {

    Intersection<ad> its(_its);

    if (m_twoSide) {
        its.wi.z() = drjit::abs(its.wi.z());
    }

    BSDFSample<ad> bs = zeros<BSDFSample<ad>>();
    Float<ad> cos_theta_i = Frame<ad>::cos_theta(its.wi);
    Float<ad> alpha_u = m_alpha_u.eval<ad>(its.uv);
    Float<ad> alpha_v = m_alpha_v.eval<ad>(its.uv);
    GGXDistribution distr(alpha_u, alpha_v);


    active &= neq(cos_theta_i, 0.f);


    // Sample the microfacet normal
    Vector3f<ad> m;
    std::tie(m, bs.pdf) = distr.sample<ad>(mulsign(its.wi, cos_theta_i), sample);
    active &= neq(bs.pdf, 0.f);

    Float<ad> eta;
    if constexpr (ad) {
        eta = m_eta;
    } else {
        eta = detach<false>(m_eta);
    }

    auto [F, cos_theta_t, eta_it, eta_ti] =
        fresnel_dielectric<ad>(eta, dot(its.wi, m));

    Mask<ad> selected_r, selected_t;
    selected_r = sample.z() <= F && active;
    selected_t = !selected_r && active;
    // Select the lobe to be sampled
    bs.pdf *= select(selected_r, F, 1.f - F);
    bs.eta = select(selected_r, 1.f, eta_it);

    Float<ad> dwh_dwo = 0.f;
    // Reflection sampling
    if (any_or<true>(selected_r)) {
        bs.wo[selected_r] = fmsub(Vector3f<ad>(m), 2.f * dot(its.wi, m), its.wi);
        dwh_dwo = rcp(4.f * dot(bs.wo, m));
    }
    // Transmission sampling
    if (any_or<true>(selected_t)) {
        bs.wo[selected_t] = fmsub(m, fmadd(dot(its.wi, m), eta_ti, cos_theta_t), its.wi * eta_ti);
        dwh_dwo[selected_t] =
            (sqr(bs.eta) * dot(bs.wo, m)) /
             sqr(dot(its.wi, m) + bs.eta * dot(bs.wo, m));
    }


    bs.pdf *= abs(dwh_dwo) * distr.smith_g1<ad>(bs.wo, m);
    bs.is_valid = active && (selected_t || selected_r);
    return detach(bs);
}

NAMESPACE_END(psdr_jit)
