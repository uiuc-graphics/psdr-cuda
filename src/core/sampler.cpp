#include <misc/Exception.h>
#include <psdr/core/sampler.h>

NAMESPACE_BEGIN(psdr_jit)

template <typename UInt32>
uint64_array_t<UInt32> sample_tea_64(UInt32 v0, UInt32 v1, int rounds = 4) {
    UIntC sum = 0;

    for ( int i = 0; i < rounds; ++i ) {
        sum += 0x9e3779b9;
        v0 += (sl<4>(v1) + 0xa341316c) ^ (v1 + sum) ^ (sr<5>(v1) + 0xc8013ea4);
        v1 += (sl<4>(v0) + 0xad90777d) ^ (v0 + sum) ^ (sr<5>(v0) + 0x7e95761e);
    }

    return uint64_array_t<UInt32>(v0) + sl<32>(uint64_array_t<UInt32>(v1));
}

void Sampler::seed(UInt64C seed_value) {
    seed_indices(seed_value, arange<UInt64C>(seed_value.size()));
}

void Sampler::seed_indices(UInt64C seed_value, const UInt64C &idx) {
    if ( !m_rng )
        m_rng = std::make_unique<PCG32>();

    seed_value += m_base_seed;

    // drjit v1.x: seed(initstate, initseq) - removed first size parameter
    m_rng->seed(sample_tea_64(seed_value, idx), sample_tea_64(idx, seed_value));

    m_sample_count = static_cast<int64_t>(seed_value.size());
}


template <bool ad>
Float<ad> Sampler::next_1d() {
    if ( m_rng == nullptr )
        throw Exception("Sampler::seed() must be invoked before using this sampler!");
    else {
        Float<ad> rs = m_rng->template next_float<Float<ad>>();
        schedule(m_rng->inc, m_rng->state);
        return rs;
    }
}


// Explicit instanciations
template FloatC Sampler::next_1d<false>();
template FloatD Sampler::next_1d<true>();

NAMESPACE_END(psdr_jit)
