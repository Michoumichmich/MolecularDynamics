#pragma once

#include "cpu_backend.h"

namespace sim {

/**
 *
 * @param particules
 * @param config
 */
template<bool use_domain_decomposition, typename T>
void cpu_backend<use_domain_decomposition, T>::set_particles(const std::vector<coordinate<T>>& particules, const configuration<T>& config) {
    size_ = particules.size();
    coordinates_ = particules;
    momentums_ = std::vector<coordinate<T>>(size_);
    forces_ = std::vector<coordinate<T>>(size_);
    energies_ = std::vector<T>(size_);
    if constexpr (use_domain_decomposition) {
        decomposer_ = domain_decomposer<T, true>(config.domain_mins, config.domain_maxs, config.domain_widths, config.decompose_periodicity);
    } else {
        decomposer_ = domain_decomposer<T, false>(config.L);
    }
}

template<bool use_domain_decomposition, typename T>
void cpu_backend<use_domain_decomposition, T>::set_speeds(const std::vector<coordinate<T>>& speeds, const configuration<T>& config) {
    momentums_ = speeds;
    for (auto& momentum: momentums_) { momentum *= config.m_i; }
}


/**
 *
 * @param min
 * @param max
 */
template<bool use_domain_decomposition, typename T> void cpu_backend<use_domain_decomposition, T>::randinit_momentums(T min, T max) {
    std::generate(momentums_.begin(), momentums_.end(), [=]() {   //
        return coordinate<T>(internal::generate_random_value<T>(min, max), internal::generate_random_value<T>(min, max), internal::generate_random_value<T>(min, max));
    });
}

/**
 *
 * @param writer
 * @param i
 */
template<bool use_domain_decomposition, typename T> void cpu_backend<use_domain_decomposition, T>::store_particles_coordinates(pdb_writer& writer, size_t i, T temp, T epot) const {
    writer.store_new_iter(i, coordinates_, temp, epot);
}

/**
 *
 * @return
 */
template<bool use_domain_decomposition, typename T> T cpu_backend<use_domain_decomposition, T>::get_momentums_squared_norm() const {
    T sum{};
    for (const auto& momentum: momentums_) { sum += sycl::dot(momentum, momentum); }
    return sum;
}

/**
 *
 * @param coeff
 */
template<bool use_domain_decomposition, typename T> void cpu_backend<use_domain_decomposition, T>::apply_multiplicative_correction_to_momentums(T coeff) {
    for (auto& momentum: momentums_) { momentum *= coeff; }
}

/**
 *
 */
template<bool use_domain_decomposition, typename T> void cpu_backend<use_domain_decomposition, T>::center_kinetic_momentums() {
    auto mean = mean_kinetic_momentum();
    for (auto& momentum: momentums_) { momentum -= mean; }
}

/**
 *
 * @return
 */
template<bool use_domain_decomposition, typename T> coordinate<T> cpu_backend<use_domain_decomposition, T>::mean_kinetic_momentum() const {
    coordinate<T> mean{};   // Sum of vi * mi;
    for (const auto& momentum: momentums_) { mean += momentum; }
    return mean / momentums_.size();
}

/**
 *
 * @return
 */
template<bool use_domain_decomposition, typename T> std::tuple<coordinate<T>, T> cpu_backend<use_domain_decomposition, T>::get_last_lennard_jones_metrics() const {
    auto sum_forces = coordinate<T>{};
    auto sum_energies = T{};
    for (size_t i = 0; i < size_; ++i) {
        sum_forces += forces_[i];
        sum_energies += energies_[i];
    }
    return std::tuple{sum_forces, sum_energies};
}


/**
 *
 * @param config
 */
template<bool use_domain_decomposition, typename T> void cpu_backend<use_domain_decomposition, T>::update_lennard_jones_field(const configuration<T>& config) {
    std::fill(energies_.begin(), energies_.end(), 0);
    std::fill(forces_.begin(), forces_.end(), coordinate<T>{});

    auto lennard_jones_field_impl = [&]<int n_sym>() mutable {
        decomposer_.template run_kernel_on_domains<n_sym>(coordinates_, [&](const auto i, const auto& this_particule, const auto& other_particule) mutable {
            const T squared_distance = compute_squared_distance(this_particule, other_particule);
            const bool commit_result = !(squared_distance < get_epsilon<T>() || (config.use_cutoff && squared_distance > integral_power<2>(config.r_cut)));

            if constexpr (!use_domain_decomposition) {
                if (!commit_result) return;
            }

            const T frac_pow_2 = config.r_star * config.r_star / (squared_distance + get_epsilon<T>());
            const T frac_pow_6 = integral_power<3>(frac_pow_2);
            const T force_prefactor = (frac_pow_6 - 1) * frac_pow_6 * frac_pow_2;
            forces_[i] += (this_particule - other_particule) * force_prefactor * config.epsilon_star * 48 * static_cast<T>(commit_result);   // / (config.r_star * config.r_star);
            energies_[i] += 2 * config.epsilon_star * (integral_power<2>(frac_pow_6) - 2 * frac_pow_6) * static_cast<T>(commit_result);      // Divided by two bc symetries
        });
    };

    if (config.n_symetries == 1) {
        lennard_jones_field_impl.template operator()<1>();
    } else if (config.n_symetries == 27) {
        lennard_jones_field_impl.template operator()<27>();
    } else if (config.n_symetries == 125) {
        lennard_jones_field_impl.template operator()<125>();
    } else {
        throw std::runtime_error("Unsupported");
    }
}

/**
 *
 * @param config
 */
template<bool use_domain_decomposition, typename T> void cpu_backend<use_domain_decomposition, T>::run_velocity_verlet(const configuration<T>& config) {
    internal::assume(coordinates_.size() == forces_.size() && forces_.size() == momentums_.size());
    const size_t N = coordinates_.size();

    // First step: half step update of the momentums.
    for (size_t i = 0; i < N; ++i) { momentums_[i] += config.conversion_force * forces_[i] * config.dt / 2; }

    // Second step: update particules positions
    for (size_t i = 0; i < N; ++i) { coordinates_[i] += config.dt * momentums_[i] / config.m_i; }

    update_lennard_jones_field(config);

    // Last step: update momentums given new forces
    for (size_t i = 0; i < N; ++i) { momentums_[i] += config.conversion_force * forces_[i] * config.dt / 2; }

    update_lennard_jones_field(config);
}


}   // namespace sim
