#pragma once

#include <backend/backend_interface.h>

#include <iomanip>

namespace sim {

/**
 * @class molecular_dynamics
 * @tparam T
 */
template<typename T, template<typename> class backend> class molecular_dynamics {
public:
    /**
     * Constructor, initializes the system.
     * @param particules
     * @param config
     */
    inline molecular_dynamics(const std::vector<coordinate<T>>& particules, configuration<T> config, backend<T>&& be = {})
        : configuration_(config),     //
          simulation_idx_(0),         //
          writer_(config.out_file),   //
          backend_(std::move(be)) {

        backend_.init_backend(std::move(particules));

        // Initializes the forces field.
        backend_.init_lennard_jones_field(config);

        // Randinit momentums
        backend_.randinit_momentums(-1, 1);

        // Fixes the barycenter.
        backend_.center_kinetic_momentums();

        // Computes the temperature of the systme and scales the momentums to reach the target temperature.
        fixup_temperature(configuration_.T0);
        backend_.store_particules_coordinates(writer_, simulation_idx_);
    }

    /**
     *
     */
    inline void run_iter() {
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
        if (simulation_idx_ % configuration_.iter_per_frame == 0) { backend_.store_particules_coordinates(writer_, simulation_idx_); }
    }

private:
    /**
     * Updates the kinectic energy and temperature based on current momentums.
     * @tparam T
     */
    inline void recompute_kinetic_energy_and_temp() noexcept {
        T sum = backend_.get_momentums_squared_norm();
        kinetic_energy_ = sum / (2 * configuration_.conversion_force * configuration_.m_i);
        kinetic_temperature_ = kinetic_energy_ / (configuration_.constante_R * degrees_of_freedom());
    }

    [[nodiscard]] inline size_t degrees_of_freedom() const noexcept { return 3 * backend_.get_particules_count() - 3; }

    /**
     * Updates the momentums to match the desierd kinetic temperature.
     * @tparam T
     * @param desired_temperature
     */
    inline void fixup_temperature(T desired_temperature) noexcept {
        recompute_kinetic_energy_and_temp();
        const T coeff = sycl::sqrt(degrees_of_freedom() * configuration_.constante_R * desired_temperature / kinetic_energy_);
        backend_.apply_multiplicative_correction_to_momentums(coeff);
        recompute_kinetic_energy_and_temp();
    }

    /**
     * Applies the Berendsen thermostate on the current system using the current kinetic temperature.
     */
    inline void try_to_apply_berendsen_thermostate() noexcept {
        if (!configuration_.use_berdensten_thermostate) { return; }
        if (simulation_idx_ % configuration_.m_step != 0 || simulation_idx_ == 0) { return; }
        recompute_kinetic_energy_and_temp();
        //const T coeff = config_.gamma * sycl::sqrt(std::abs((config_.T0 / kinetic_temperature_) - 1));
        const T coeff = configuration_.gamma * ((configuration_.T0 / kinetic_temperature_) - 1);
        backend_.apply_multiplicative_correction_to_momentums(1 + coeff);   // momentum += momentum * coeff
        std::cout << "[Berendsen] Thermostate applied with coeff: " << coeff << std::endl;
    }

private:
    const configuration<T> configuration_;   //
    size_t simulation_idx_;                  //
    pdb_writer writer_;                      //
    backend<T> backend_;                     //
    T kinetic_temperature_{};                //
    T kinetic_energy_{};                     //


    // Metrics
private:
    // Mutable variables are the variables that are easily and often re-computed based on the simulation
    // they should be updated as often as possible, but don't really affect the simulation at all
    mutable coordinate<T> forces_sum_{};     //
    mutable T lennard_jones_energy_{};       //
    mutable double avg_iter_duration_ = 1;   //
    mutable double total_energy_ = 0;        //
    mutable double avg_delta_energy_ = 0;    //


    // Not very important stuff
public:
    inline void update_display_metrics() const noexcept {
        auto [sum, energy] = backend_.get_last_lennard_jones_metrics();
        lennard_jones_energy_ = energy;
        forces_sum_ = sum;

        constexpr static double aging_coeff = 0.01;
        double prev_energy = total_energy_;
        total_energy_ = kinetic_energy_ + lennard_jones_energy_;
        avg_delta_energy_ = ((total_energy_ - prev_energy) * aging_coeff + avg_delta_energy_) / (1 + aging_coeff);
    }


    friend std::ostream& operator<<(std::ostream& os, const molecular_dynamics& state) {
        state.update_display_metrics();
        os << std::setprecision(10) << "[" << state.simulation_idx_ << "] "                                                                                           //
           << "E_tot: " << std::setw(13) << std::setfill(' ') << state.total_energy_                                                                                  //
           << ", Temp: " << std::setw(13) << std::setfill(' ') << state.kinetic_temperature_                                                                          //
           << ", E_c: " << std::setw(13) << std::setfill(' ') << state.kinetic_energy_                                                                                //
           << ", E_pot: " << std::setw(13) << std::setfill(' ') << state.lennard_jones_energy_                                                                        //
           << ", Field_sum_norm: " << std::setw(13) << std::setfill(' ') << sycl::length(state.forces_sum_)                                                           //
           << ", Barycenter_speed_norm: " << std::setw(13) << std::setfill(' ') << sycl::length(state.backend_.mean_kinetic_momentums()) / state.configuration_.m_i   //
           << ", Avg_delta_energy: " << std::setprecision(5) << state.avg_delta_energy_                                                                               //
           << ", Time: " << state.configuration_.dt * state.simulation_idx_ << " fs"                                                                                  //
           << ", Speed: " << std::setprecision(3) << 1.0 / state.avg_iter_duration_ << " iter/s.";
        return os;
    }
};

}   // namespace sim
