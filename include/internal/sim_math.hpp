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

template<int N, typename T> constexpr T integral_power_helper(const T& y, const T& x) {
    if constexpr (N < 0) {
        return integral_power_helper<-N, T>(y, T(1) / x);
    } else if constexpr (N == 0) {
        return y;
    } else if constexpr (N == 1) {
        return x * y;
    } else if constexpr (N % 2 == 0) {
        return integral_power_helper<N / 2, T>(y, x * x);
    } else {
        return integral_power_helper<(N - 1) / 2, T>(y * x, x * x);
    }
}


template<int N, typename T> constexpr T integral_power(const T& v) {
    static_assert(integral_power_helper<0>(1, 0) == 1);
    static_assert(integral_power_helper<1>(1, 0) == 0);
    static_assert(integral_power_helper<0>(1, 5) == 1);
    static_assert(integral_power_helper<1>(1, 5) == 5);
    static_assert(integral_power_helper<2>(1, 5) == 25);
    static_assert(integral_power_helper<3>(1, 5) == 125);
    return integral_power_helper<N>(T(1), v);
}

}   // namespace sim
