#pragma once

#include "utils.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

using namespace std::string_literals;

template<typename T> using coordinate = sycl::vec<T, 3U>;

template<typename T> struct simulation_configuration {
    static constexpr T m_i = 18;                            // Mass of a particle in some unit
    static constexpr T conversion_force = 0.0001 * 4.186;   //
    static constexpr T constante_R = 0.00199;               //
    static constexpr T dt = 1;                              // 0.1 fs, should be 1.
    static constexpr T T0 = 300;                            // 300 Kelvin

    // Berdensten thermostate
    static constexpr T gamma = 0.01;        // Gamma for the berdensten thermostate, should be 0.01
    static constexpr size_t m_step = 100;   //

    // Lennard jones field config
    static constexpr T r_star_ = static_cast<T>(3);           //
    static constexpr T epsilon_star_ = static_cast<T>(0.2);   //
    bool use_cutoff = true;                                   //
    T r_cut_ = static_cast<T>(10);                            // 10 Angstroms
    int n_symetries = 27;                                     //
    T L_ = static_cast<T>(30);                                //

    // PDB Out settings
    int iter_per_frame = 10;                                       //
    std::string out_file = "unnamed_"s + config_hash() + ".pdb";   // Set an empty name to not save the result.

    [[nodiscard]] std::string config_hash() const {
        return "L:"s + std::to_string(L_)                    //
             + "_sym:" + std::to_string(n_symetries)         //
             + "_rcut:" + std::to_string(r_cut_)             //
             + "_usecut:" + std::to_string(use_cutoff)       //
             + "_dt:" + std::to_string(dt)                   //
             + "_period:" + std::to_string(iter_per_frame)   //
             + "_" + type_to_string();
    }

    static constexpr auto type_to_string() noexcept {
        if constexpr (std::is_same_v<T, sycl::half>) {
            return "sycl::half";
        } else if constexpr (std::is_same_v<T, float>) {
            return "float";
        } else if constexpr (std::is_same_v<T, double>) {
            return "double";
        } else {
            fail_to_compile<T>();
        }
    }


    friend std::ostream& operator<<(std::ostream& os, simulation_configuration config) {
        os << "Cutoff: " << config.use_cutoff           //
           << ", r_cut: " << config.r_cut_              //
           << ", n_symetries: " << config.n_symetries   //
           << ", box_width: " << config.L_              //
           << ", type: " << type_to_string();
        return os;
    }
};

/**
 * Converts a vector of coordinates<Src_T> to a vector of coordinates<Dst_T>
 * @tparam Dst_T
 * @tparam Src_T
 * @param in
 * @return
 */
template<typename Dst_T, typename Src_T> static inline std::vector<coordinate<Dst_T>> coordinate_vector_cast(const std::vector<coordinate<Src_T>>& in) {
    std::vector<coordinate<Dst_T>> out(in.size());
    for (unsigned int i = 0; i < in.size(); ++i) { out[i] = coordinate<Dst_T>{static_cast<Dst_T>(in[i].x()), static_cast<Dst_T>(in[i].y()), static_cast<Dst_T>(in[i].z())}; }
    return out;
}

/**
 * Reads the particule file and returns a vector of coordinates (doubles)
 * @param filename
 * @return
 */
static inline std::vector<coordinate<double>> parse_particule_file(std::string&& filename) {
    auto fs = std::ifstream(filename);
    if (!fs.is_open()) throw std::runtime_error("File not found");
    auto comment = std::string{};
    std::getline(fs, comment);
    std::cout << "Comment is: " << comment << std::endl << std::endl;
    auto coordinates = std::vector<coordinate<double>>{};
    while (!fs.eof()) {
        auto tmp = 0;
        coordinate<double> c{};
        fs >> tmp >> c.x() >> c.y() >> c.z();
        coordinates.emplace_back(c);
    }
    return coordinates;
}

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

template<typename T> constexpr static inline T compute_squared_distance(const coordinate<T>& lhs, const coordinate<T>& rhs) {
    return (lhs[0U] - rhs[0U]) * (lhs[0U] - rhs[0U]) + (lhs[1U] - rhs[1U]) * (lhs[1U] - rhs[1U]) + (lhs[2U] - rhs[2U]) * (lhs[2U] - rhs[2U]);
}