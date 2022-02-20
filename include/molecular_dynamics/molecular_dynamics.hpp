#pragma once

#include "molecular_dynamics.h"

namespace sim {


/**
 *
 * @param particules
 * @param config
 * @param be
 */
template<typename T, template<typename> class backend>
EXPORT molecular_dynamics<T, backend>::molecular_dynamics(const std::vector<coordinate<T>>& particules, configuration<T> config, backend<T>&& be)
    : configuration_(config),                                         //
      simulation_idx_(0),                                             //
      writer_(config.out_file, config.store_lennard_jones_metrics),   //
      backend_(std::move(be)) {

    backend_.set_particles(std::move(particules), config);

    // Initializes the forces field.
    backend_.update_lennard_jones_field(config);

    // Randinit momentums
    backend_.randinit_momentums(-1, 1);

    // Fixes the barycenter.
    backend_.center_kinetic_momentums();

    // Computes the temperature of the systme and scales the momentums to reach the target temperature.
    fixup_temperature(configuration_.T0);
    update_display_metrics();
    backend_.store_particles_coordinates(writer_, simulation_idx_, kinetic_temperature_, lennard_jones_energy_);
}


template<typename T, template<typename> class backend>
EXPORT molecular_dynamics<T, backend>::molecular_dynamics(const std::vector<coordinate<T>>& particules, const std::vector<coordinate<T>>& speeds, configuration<T> config,
                                                          backend<T>&& be)
    : configuration_(config),                                         //
      simulation_idx_(0),                                             //
      writer_(config.out_file, config.store_lennard_jones_metrics),   //
      backend_(std::move(be)) {

    backend_.set_particles(std::move(particules), config);
    backend_.set_speeds(speeds, config);
    // Initializes the forces field.
    backend_.update_lennard_jones_field(config);
    fixup_temperature(configuration_.T0);
    recompute_kinetic_energy_and_temp();
    backend_.store_particles_coordinates(writer_, simulation_idx_, kinetic_temperature_, lennard_jones_energy_);
}


/**
 *
 */
template<typename T, template<typename> class backend> EXPORT void molecular_dynamics<T, backend>::run_iter() {
    auto begin = std::chrono::high_resolution_clock::now();
    ++simulation_idx_;

    // Running the velocity verlet algorithm and getting back the potential energy as well as the sum of the forces.
    backend_.run_velocity_verlet(configuration_);

    // Eventyally and fixes up the temperature
    try_to_apply_berendsen_thermostate();

    // Recoputes the temperature and the kinetic energy
    recompute_kinetic_energy_and_temp();

    // Perf update
    auto end = std::chrono::high_resolution_clock::now();
    constexpr static double aging_coeff = 0.01;
    auto duration = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()) * 1e-9;
    avg_iter_duration_ = (avg_iter_duration_ + duration * aging_coeff) / (1 + aging_coeff);

    // Eventually stores the result


    if (simulation_idx_ % configuration_.iter_per_frame == 0) {
        if (configuration_.store_lennard_jones_metrics) { update_display_metrics(); }
        backend_.store_particles_coordinates(writer_, simulation_idx_, lennard_jones_energy_, kinetic_temperature_);
    }
}

/**
 *
 * @tparam T
 * @tparam backend
 */
template<typename T, template<typename> class backend> void molecular_dynamics<T, backend>::try_to_apply_berendsen_thermostate() noexcept {
    if (!configuration_.use_berdensten_thermostate) { return; }
    if (simulation_idx_ % configuration_.m_step != 0 || simulation_idx_ == 0) { return; }
    recompute_kinetic_energy_and_temp();
    //const T coeff = config_.gamma * sycl::sqrt(std::abs((config_.T0 / kinetic_temperature_) - 1));
    const T coeff = configuration_.gamma * ((configuration_.T0 / kinetic_temperature_) - 1);
    backend_.apply_multiplicative_correction_to_momentums(1 + coeff);   // momentum += momentum * coeff
    std::cout << "[Berendsen] Thermostate applied with coeff: " << coeff << std::endl;
}

/**
 *
 * @tparam T
 * @tparam backend
 * @param desired_temperature
 */
template<typename T, template<typename> class backend> void molecular_dynamics<T, backend>::fixup_temperature(T desired_temperature) noexcept {
    recompute_kinetic_energy_and_temp();
    const T coeff = sycl::sqrt(degrees_of_freedom() * configuration_.constante_R * desired_temperature / kinetic_energy_);
    backend_.apply_multiplicative_correction_to_momentums(coeff);
    recompute_kinetic_energy_and_temp();
}

/**
 *
 * @tparam T
 * @tparam backend
 */
template<typename T, template<typename> class backend> void molecular_dynamics<T, backend>::recompute_kinetic_energy_and_temp() noexcept {
    T sum = backend_.get_momentums_squared_norm();
    kinetic_energy_ = sum / (2 * configuration_.conversion_force * configuration_.m_i);
    kinetic_temperature_ = kinetic_energy_ / (configuration_.constante_R * degrees_of_freedom());
}

/**
 *
 * @tparam T
 * @tparam backend
 */
template<typename T, template<typename> class backend> EXPORT void molecular_dynamics<T, backend>::update_display_metrics() const noexcept {
    // recompute_kinetic_energy_and_temp();
    auto [sum, energy] = backend_.get_last_lennard_jones_metrics();
    lennard_jones_energy_ = energy;
    forces_sum_ = sum;

    constexpr static double aging_coeff = 0.01;
    double prev_energy = total_energy_;
    total_energy_ = static_cast<double>(kinetic_energy_ + lennard_jones_energy_);
    avg_delta_energy_ = ((total_energy_ - prev_energy) * aging_coeff + avg_delta_energy_) / (1 + aging_coeff);
}

template<typename T, template<typename> class backend> T EXPORT molecular_dynamics<T, backend>::compute_barycenter_speed_norm() const {
    return sycl::length(backend_.mean_kinetic_momentum()) / configuration_.m_i;
}

}   // namespace sim