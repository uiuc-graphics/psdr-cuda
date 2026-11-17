#include <misc/Exception.h>
#include <psdr/core/pmf.h>

NAMESPACE_BEGIN(psdr_jit)

void DiscreteDistribution::init(const FloatC &pmf) {
    m_size = static_cast<int>((pmf.size()));
    m_sum = sum(pmf);
    m_pmf = pmf;
    FloatC temp = drjit::migrate(m_pmf, JitBackend::None);
    drjit::sync_thread();
    m_cmf = compute_cdf(temp);
    m_pmf_normalized = pmf/m_sum;
    m_cmf_normalized = m_cmf/m_sum;
}


std::pair<IntC, FloatC> DiscreteDistribution::sample(const FloatC &_samples) const {
    if ( unlikely(m_size == 1) ) {
        return { zeros<IntC>(), full<FloatC>(1.f) };
    }
    FloatC samples = _samples*m_sum;
    IntC idx = binary_search<IntC>(
        0, m_size - 1, [&](IntC i)DRJIT_INLINE_LAMBDA { return gather<FloatC>(m_cmf, i) < samples; }
    );
    return { idx, gather<FloatC>(m_pmf, idx)/m_sum };
}


template <bool ad>
std::pair<IntC, FloatC> DiscreteDistribution::sample_reuse(Float<ad> &samples) const {
    if ( unlikely(m_size == 1) ) {
        return { zeros<IntC>(), full<FloatC>(1.f) };
    }
    samples *= m_sum;
    IntC idx;
    if constexpr ( ad ) {
        idx = binary_search<IntC>(
            0, m_size - 1, [&](IntC i)DRJIT_INLINE_LAMBDA { return gather<FloatC>(m_cmf, i) < detach<false>(samples); }
        );
    } else {
        idx = binary_search<IntC>(
            0, m_size - 1, [&](IntC i)DRJIT_INLINE_LAMBDA { return gather<FloatC>(m_cmf, i) < samples; }
        );
    }
    samples -= gather<FloatC>(m_cmf, idx - 1, idx > 0);
    FloatC pmf = gather<FloatC>(m_pmf, idx);
    masked(samples, pmf > 0.f) /= pmf;
    samples = clamp(samples, 0.f, 1.f);
    return { idx, pmf/m_sum };
}

// Explicit instantiations
template std::pair<IntC, FloatC> DiscreteDistribution::sample_reuse<false>(FloatC&) const;
template std::pair<IntC, FloatC> DiscreteDistribution::sample_reuse<true >(FloatD&) const;

NAMESPACE_END(psdr_jit)
