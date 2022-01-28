#pragma once

#include <internal/cpp_utils.hpp>

#include <iostream>
#include <string>


namespace sim {
using namespace std::string_literals;
template<typename T> struct configuration {
    static constexpr T m_i = 18;                            // Mass of a particle in some unit
    static constexpr T conversion_force = 0.0001 * 4.186;   //
    static constexpr T constante_R = 0.00199;               //
    T dt = 1;                                               // 0.1 fs, should be 1.
    static constexpr T T0 = 300;                            // 300 Kelvin

    // Berdensten thermostate
    bool use_berdensten_thermostate = true;
    static constexpr T gamma = 0.01;        // Gamma for the berdensten thermostate, should be 0.01
    static constexpr size_t m_step = 100;   // Should be 100

    // Lennard jones field config
    static constexpr T r_star = static_cast<T>(3);           // R* distance: 3A
    static constexpr T epsilon_star = static_cast<T>(0.2);   //
    static constexpr bool use_cutoff = true;                 //
    T r_cut = static_cast<T>(35);                            // Should be 10 Angstroms
    int n_symetries = 27;                                    //
    T L = static_cast<T>(35);                                // 30 in the subject

    // PDB Out settings
    int iter_per_frame = 100;                                      //
    std::string out_file = "unnamed_"s + config_hash() + ".pdb";   // Set an empty name to not save the result.

    [[nodiscard]] std::string config_hash() const {
        return "L="s + std::to_string(L)                                 //
             + "_sym=" + std::to_string(n_symetries)                     //
             + "_rcut=" + std::to_string(r_cut)                          //
             + "_usecut=" + std::to_string(use_cutoff)                   //
             + "_dt=" + std::to_string(dt)                               //
             + "_period=" + std::to_string(iter_per_frame)               //
             + "_thermo=" + std::to_string(use_berdensten_thermostate)   //
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
            internal::fail_to_compile<T>();
        }
    }


    friend std::ostream& operator<<(std::ostream& os, configuration config) {
        os << "Cutoff: " << config.use_cutoff            //
           << ", r_cut: " << config.r_cut                //
           << ", n_symetries_: " << config.n_symetries   //
           << ", box_width: " << config.L                //
           << ", type: " << type_to_string();
        return os;
    }
};
}   // namespace sim
