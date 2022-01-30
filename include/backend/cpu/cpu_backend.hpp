#pragma once
#include "cpu_backend.h"

namespace sim {
/**
 *
 * @tparam T Floating point type
 * @param coordinates std::vector of particules on the host
 * @param config Simulation configuration
 * @return tuple with the forces, summed_forces and the energy
 */
template<bool use_domain_decomposition, typename T, int n_sym>
static inline void update_lennard_jones_field_cpu_impl(   //
        std::vector<coordinate<T>>& coordinates, const configuration<T>& config, std::vector<coordinate<T>>& forces, std::vector<T>& energies,
        const domain_decomposer<T, use_domain_decomposition>& decomposer) {
    std::fill(energies.begin(), energies.end(), 0);
    std::fill(forces.begin(), forces.end(), coordinate<T>{});
    if constexpr (use_domain_decomposition) { decomposer.update_domains(coordinates); }
    decomposer.template run_kernel_on_domains<n_sym>(coordinates, [&](const auto i, const auto& this_particule, const auto& other_particule) mutable {
        const T squared_distance = compute_squared_distance(this_particule, other_particule) + 0.001;
        if (config.use_cutoff && squared_distance > integral_power<2>(config.r_cut)) return;
        internal::assume(squared_distance != T{});
        //if (squared_distance == T{}) { throw std::runtime_error("Got null distance"); }
        const T frac_pow_2 = config.r_star * config.r_star / squared_distance;
        const T frac_pow_6 = integral_power<3>(frac_pow_2);
        const T force_prefactor = (frac_pow_6 - 1) * frac_pow_6 * frac_pow_2;
        auto force = (this_particule - other_particule) * force_prefactor * config.epsilon_star * 48 / (config.r_star * config.r_star);
        if constexpr (use_domain_decomposition) {
            if (sycl::length(force) > config.max_force) return;   //{ printf("%f %f\n", squared_distance, sycl::length(force)); }
        }
        forces[i] += force;
        energies[i] += 2 * config.epsilon_star * (integral_power<2>(frac_pow_6) - 2 * frac_pow_6);   //We divided because the energies would be counted twice otherwise
    });
}

template<bool use_domain_decomposition, typename T> void cpu_backend<use_domain_decomposition, T>::init_lennard_jones_field(const configuration<T>& config) {
    if (config.n_symetries == 1) {
        update_lennard_jones_field_cpu_impl<use_domain_decomposition, T, 1>(coordinates_, config, forces_, energies_, decomposer_);
    } else if (config.n_symetries == 27) {
        update_lennard_jones_field_cpu_impl<use_domain_decomposition, T, 27>(coordinates_, config, forces_, energies_, decomposer_);
    } else if (config.n_symetries == 125) {
        update_lennard_jones_field_cpu_impl<use_domain_decomposition, T, 125>(coordinates_, config, forces_, energies_, decomposer_);
    } else {
        throw std::runtime_error("Unsupported");
    }
}


/**
 * Runs one iteration of the velocity verlet algorithm.
 * @tparam T float type
 * @tparam n_sym 1 or 27
 * @param coordinates inout
 * @param forces inout
 * @param momentums inout
 * @param config
 * @return std::tuple{sum, energy}
 */
template<bool use_domain_decomposition, typename T, int n_sym>
static inline void velocity_verlet_cpu_impl(       //
        std::vector<coordinate<T>>& coordinates,   //
        std::vector<coordinate<T>>& forces,        //
        std::vector<coordinate<T>>& momentums,     //
        std::vector<T>& energies,                  //
        const domain_decomposer<T, use_domain_decomposition>& decomposer, const configuration<T>& config) noexcept {

    internal::assume(coordinates.size() == forces.size() && forces.size() == momentums.size());
    const size_t N = coordinates.size();

    // First step: half step update of the momentums.
    for (size_t i = 0; i < N; ++i) { momentums[i] += config.conversion_force * forces[i] * config.dt / 2; }

    // Second step: update particules positions
    for (size_t i = 0; i < N; ++i) { coordinates[i] += config.dt * momentums[i] / config.m_i; }

    update_lennard_jones_field_cpu_impl<use_domain_decomposition, T, n_sym>(coordinates, config, forces, energies, decomposer);

    // Last step: update momentums given new forces
    for (size_t i = 0; i < N; ++i) { momentums[i] += config.conversion_force * forces[i] * config.dt / 2; }

    update_lennard_jones_field_cpu_impl<use_domain_decomposition, T, n_sym>(coordinates, config, forces, energies, decomposer);
}

template<bool use_domain_decomposition, typename T> void cpu_backend<use_domain_decomposition, T>::run_velocity_verlet(const configuration<T>& config) {
    if (config.n_symetries == 1) {
        velocity_verlet_cpu_impl<use_domain_decomposition, T, 1>(coordinates_, forces_, momentums_, energies_, decomposer_, config);
    } else if (config.n_symetries == 27) {
        velocity_verlet_cpu_impl<use_domain_decomposition, T, 27>(coordinates_, forces_, momentums_, energies_, decomposer_, config);
    } else if (config.n_symetries == 125) {
        velocity_verlet_cpu_impl<use_domain_decomposition, T, 125>(coordinates_, forces_, momentums_, energies_, decomposer_, config);
    } else {
        throw std::runtime_error("Unsupported");
    }
}

template<bool use_domain_decomposition, typename T> void cpu_backend<use_domain_decomposition, T>::randinit_momentums(T min, T max) {
    std::generate(momentums_.begin(), momentums_.end(), [=]() {   //
        return coordinate<T>(internal::generate_random_value<T>(min, max), internal::generate_random_value<T>(min, max), internal::generate_random_value<T>(min, max));
    });
}

template<bool use_domain_decomposition, typename T> void cpu_backend<use_domain_decomposition, T>::store_particules_coordinates(pdb_writer& writer, size_t i) const {
    writer.store_new_iter(coordinates_, i);
}

template<bool use_domain_decomposition, typename T> T cpu_backend<use_domain_decomposition, T>::get_momentums_squared_norm() const {
    T sum{};
    for (const auto& momentum: momentums_) { sum += sycl::dot(momentum, momentum); }
    return sum;
}

template<bool use_domain_decomposition, typename T> void cpu_backend<use_domain_decomposition, T>::apply_multiplicative_correction_to_momentums(T coeff) {
    for (auto& momentum: momentums_) { momentum *= coeff; }
}

template<bool use_domain_decomposition, typename T> void cpu_backend<use_domain_decomposition, T>::center_kinetic_momentums() {
    auto mean = mean_kinetic_momentums();
    for (auto& momentum: momentums_) { momentum -= mean; }
}

template<bool use_domain_decomposition, typename T> coordinate<T> cpu_backend<use_domain_decomposition, T>::mean_kinetic_momentums() const {
    coordinate<T> mean{};   // Sum of vi * mi;
    for (const auto& momentum: momentums_) { mean += momentum; }
    return mean / momentums_.size();
}

template<bool use_domain_decomposition, typename T>
void cpu_backend<use_domain_decomposition, T>::init_backend(const std::vector<coordinate<T>>& particules, const configuration<T>& config) {
    size_ = particules.size();
    coordinates_ = particules;
    momentums_ = std::vector<coordinate<T>>(size_);
    forces_ = std::vector<coordinate<T>>(size_);
    energies_ = std::vector<T>(size_);
    if constexpr (use_domain_decomposition) {
        decomposer_ = domain_decomposer<T, use_domain_decomposition>(config.domain_mins, config.domain_maxs, config.domain_widths);
    } else {
        decomposer_ = domain_decomposer<T, use_domain_decomposition>(config.L);
    }
}
template<bool use_domain_decomposition, typename T> std::tuple<coordinate<T>, T> cpu_backend<use_domain_decomposition, T>::get_last_lennard_jones_metrics() const {
    auto sum_forces = coordinate<T>{};
    auto sum_energies = T{};
    for (size_t i = 0; i < size_; ++i) {
        sum_forces += forces_[i];
        sum_energies += energies_[i];
    }
    return std::tuple{sum_forces, sum_energies};
}


}   // namespace sim
