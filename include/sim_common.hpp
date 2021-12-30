#pragma once

#include <array>
#include <cmath>
#include <sycl/sycl.hpp>
#include <vector>

/* Structure de vecteur */
template<typename T> using coordinate = sycl::vec<T, 3U>;

template<typename T> struct simulation_configuration {
    bool use_cutoff = true;
    T r_cut_ = static_cast<T>(10);
    int n_symetries = 27;
    T L_ = static_cast<T>(30);
    T r_star_ = static_cast<T>(3);
    T epsilon_star_ = static_cast<T>(0.2);

    friend std::ostream& operator<<(std::ostream& os, simulation_configuration config) {
        constexpr auto type_to_string = []() -> std::string {
            if constexpr (std::is_same_v<T, sycl::half>) {
                return "sycl::half";
            } else if constexpr (std::is_same_v<T, float>) {
                return "float";
            } else if constexpr (std::is_same_v<T, double>) {
                return "double";
            } else {
                return "unknown";
            }
        };

        os << "Cutoff: " << config.use_cutoff << ", r_cut: " << config.r_cut_ << ", n_symetries: " << config.n_symetries << ", box_width: " << config.L_
           << ", type: " << type_to_string();
        return os;
    }
};

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


template<int N> static constexpr std::array<sycl::vec<int, 3U>, N> get_symetries() {
    static_assert(N == 1 || N == 27 || N == 125);
    if constexpr (N == 1) {
        return std::array<sycl::vec<int, 3U>, 1>{sycl::vec<int, 3U>{0, 0, 0}};
    } else {
        std::array<sycl::vec<int, 3U>, N> out;
        constexpr int n = icbrt(N);
        constexpr int delta = n / 2;
        for (int i = -delta; i <= delta; ++i) {
            for (int j = -delta; j <= delta; ++j) {
                for (int k = -delta; k <= delta; ++k) { out[n * n * (i + delta) + n * (j + delta) + (k + delta)] = sycl::vec<int, 3U>(i, j, k); }
            }
        }
        return out;
    }
}


template<typename T> constexpr T compute_squared_distance(const coordinate<T>& lhs, const coordinate<T>& rhs) {
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


constexpr size_t isqrt_impl(size_t sq, size_t dlt, size_t value) { return sq <= value ? isqrt_impl(sq + dlt, dlt + 2, value) : (dlt >> 1) - 1; }

constexpr size_t isqrt(size_t value) { return isqrt_impl(1, 3, value); }

constexpr auto strictly_lower_to_linear(int row, int column) {
    // assert row < column
    return (row * (row - 1)) / 2 + column;
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