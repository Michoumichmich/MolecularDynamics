#pragma once

#include <internal/cpp_utils.hpp>

namespace sim {

template<typename T> using coordinate = sycl::vec<T, 3U>;

/**
 * Computes the symetries
 * @tparam N
 * @return
 */
template<int N> static inline constexpr std::array<sycl::vec<int, 3U>, N> get_symetries() {
    static_assert(N == 1 || N == 27 || N == 125);
    if constexpr (N == 1) {
        return std::array<sycl::vec<int, 3U>, 1>{sycl::vec<int, 3U>{0, 0, 0}};
    } else {
        std::array<sycl::vec<int, 3U>, N> out;
        constexpr int n = internal::icbrt(N);
        constexpr int delta = n / 2;
        for (int i = -delta; i <= delta; ++i) {
            for (int j = -delta; j <= delta; ++j) {
                for (int k = -delta; k <= delta; ++k) { out[n * n * (i + delta) + n * (j + delta) + (k + delta)] = sycl::vec<int, 3U>(i, j, k); }
            }
        }
        return out;
    }
}

template<typename T> constexpr static inline T compute_squared_distance(const coordinate<T>& lhs, const coordinate<T>& rhs) {
    return (lhs[0U] - rhs[0U]) * (lhs[0U] - rhs[0U]) + (lhs[1U] - rhs[1U]) * (lhs[1U] - rhs[1U]) + (lhs[2U] - rhs[2U]) * (lhs[2U] - rhs[2U]);
}


}   // namespace sim
