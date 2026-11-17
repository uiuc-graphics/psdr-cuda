#pragma once

#include <drjit/random.h>
#include <psdr/psdr.h>

NAMESPACE_BEGIN(psdr_jit)

struct Sampler {
    using PCG32 = drjit::PCG32<IntC>;

    // std::shared_ptr<Sampler> clone();

    void seed(UInt64C seed_value);
    void seed_indices(UInt64C seed_value, const UInt64C &indices);

    inline bool is_ready() const { return m_rng != nullptr; }

    template <bool ad> Float<ad> next_1d();

    template <bool ad> inline Vector2f<ad> next_2d() {
        return Vector2f<ad>(next_1d<ad>(), next_1d<ad>());
    }

    template <bool ad> inline Vector3f<ad> next_3d() {
        return Vector3f<ad>(next_1d<ad>(), next_1d<ad>(), next_1d<ad>());
    }

    template <int n, bool ad> inline Vectorf<n, ad> next_nd() {
        static_assert(n > 0);
        if constexpr ( n == 1 ) {
            return next_1d<ad>();
        } else {
            constexpr int m = n / 2;
            return concat(next_nd<m, ad>(), next_nd<n - m, ad>());
        }
    }

    int64_t m_sample_count = 0;
    uint64_t m_base_seed = PCG32::PCG32_DEFAULT_STATE;
    std::unique_ptr<PCG32> m_rng = nullptr;
};

NAMESPACE_END(psdr_jit)
