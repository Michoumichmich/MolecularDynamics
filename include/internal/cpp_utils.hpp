#pragma once

#include <array>
#include <cmath>
#include <random>

#if defined(__has_include)
#    if __has_include(<sycl/sycl.hpp>)
#        include <sycl/sycl.hpp>
#    else
#        include <CL/sycl.hpp>
#    endif
#else
#    include <sycl/sycl.hpp>
#endif

#if defined(__has_include)
#    if __has_include(<span>)
#        include <span>
#    else
#        include "fallback_span.hpp"
#    endif
#else
#    include <span>
#endif


namespace sim::internal {
constexpr auto icbrt(unsigned x) {
    unsigned y = 0, b = 0;
    for (int s = 30; s >= 0; s = s - 3) {
        y = y << 1;
        b = (3 * y * y + 3 * y + 1) << s;
        if (x >= b) {
            x = x - b;
            y = y + 1;
        }
    }
    return y;
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


static inline constexpr size_t isqrt_impl(size_t sq, size_t dlt, size_t value) { return sq <= value ? isqrt_impl(sq + dlt, dlt + 2, value) : (dlt >> 1) - 1; }

static inline constexpr size_t isqrt(size_t value) { return isqrt_impl(1, 3, value); }

static inline constexpr auto strictly_lower_to_linear(int row, int column) {
    // assert row < column
    return (row * (row - 1)) / 2 + column;
}

template<class... args> struct false_type_tpl : std::false_type {};

template<class... args> static inline constexpr void fail_to_compile() { static_assert(false_type_tpl<args...>::value); }

template<typename T> static inline T generate_random_value(T min, T max) {
    static std::mt19937 engine(std::random_device{}());
    if constexpr (std::is_integral<T>::value) {
        std::uniform_int_distribution<T> distribution(min, max);
        return distribution(engine);
    } else if constexpr (std::is_same_v<T, sycl::half>) {
        std::uniform_real_distribution<float> distribution(static_cast<float>(min), static_cast<float>(max));
        return static_cast<sycl::half>(distribution(engine));
    } else {
        std::uniform_real_distribution<T> distribution(min, max);
        return distribution(engine);
    }
}

static inline void assume(bool x) noexcept {
#ifdef __clang__
    if (!(x)) {
        __builtin_trap();
        __builtin_unreachable();
    }
#else
    if (!(x)) { __builtin_unreachable(); }
#endif
}


/*
constexpr auto linear_to_strictly_lower(int index) {
    if (std::is_constant_evaluated()) {
        int row = (int) ((1 + isqrt(8 * index + 1)) / 2);
        int column = index - row * (row - 1) / 2;
        return std::pair(row, column);
    } else {
        int row = (int) ((1 + std::sqrt(8 * index + 1)) / 2);
        int column = index - row * (row - 1) / 2;
        return std::pair(row, column);
    }
}


constexpr auto check_strictly_linear(int i, int j) {
    auto [ii, jj] = linear_to_strictly_lower(strictly_lower_to_linear(i, j));
    return ii == i && jj == j;
}


static_assert(check_strictly_linear(1, 0));
static_assert(check_strictly_linear(1, 0));
static_assert(check_strictly_linear(2, 0));
static_assert(check_strictly_linear(2, 1));
static_assert(check_strictly_linear(3, 2));
*/
}   // namespace sim::internal
