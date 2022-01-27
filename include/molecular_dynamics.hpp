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
    inline molecular_dynamics(const std::vector<coordinate<T>>& particules, configuration<T> config, backend<T> be = {})
        : configuration_(config),     //
          simulation_idx_(0),         //
          forces_sum_(),              //
          lennard_jones_energy_(),    //
          writer_(config.out_file),   //
          backend_(be) {

        backend_.set_particules_coordinates(std::move(particules));

        // Initializes the forces field.
        auto [sums, energy] = backend_.init_lennard_jones_field(config);
        forces_sum_ = sums;
        lennard_jones_energy_ = energy;

        // Randinit momentums
        backend_.randinit_momentums(-1, 1);

        // Fixes the barycenter.
        backend_.center_kinetic_momentums();

        // Computes the temperature of the systme and scales the momentums to reach the target temperature.
        fixup_temperature(configuration_.T0);
        backend_.store_particules_coordinates(writer_, simulation_idx_);
        update_energy_metrics();
    }

    /**
     *
     */
    inline void run_iter() {
        auto begin = std::chrono::high_resolution_clock::now();
        ++simulation_idx_;

        // Running the velocity verlet algorithm and getting back the potential energy as well as the sum of the forces.
        auto [summed_forces, lennard_jones_energy] = backend_.run_velocity_verlet(configuration_);
        forces_sum_ = summed_forces;
        lennard_jones_energy_ = lennard_jones_energy;

        // Update and fixes up the temperature
        apply_berendsen_thermostate();
        update_kinetic_energy_and_temp();
        update_energy_metrics();

        if (simulation_idx_ % configuration_.iter_per_frame == 0) { backend_.store_particules_coordinates(writer_, simulation_idx_); }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()) * 1e-9;
        constexpr static double aging_coeff = 0.01;
        avg_iter_duration_ = (avg_iter_duration_ + duration * aging_coeff) / (1 + aging_coeff);
    }


    friend std::ostream& operator<<(std::ostream& os, const molecular_dynamics& state) {
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

private:
    /**
     * Updates the kinectic energy and temperature based on current momentums.
     * @tparam T
     */
    inline void update_kinetic_energy_and_temp() const noexcept {
        T sum = backend_.get_momentums_squared_norm();
        kinetic_energy_ = sum / (2 * configuration_.conversion_force * configuration_.m_i);
        kinetic_temperature_ = kinetic_energy_ / (configuration_.constante_R * degrees_of_freedom());
    }

    /**
     *
     * @return
     */
    [[nodiscard]] inline size_t degrees_of_freedom() const noexcept { return 3 * backend_.get_particules_count() - 3; }

    inline void update_energy_metrics() const noexcept {
        constexpr static double aging_coeff = 0.001;
        double prev_energy = total_energy_;
        total_energy_ = kinetic_energy_ + lennard_jones_energy_;
        avg_delta_energy_ = ((total_energy_ - prev_energy) * aging_coeff + avg_delta_energy_) / (1 + aging_coeff);
    }

    /**
     * Updates the momentums to match the desierd kinetic temperature.
     * @tparam T
     * @param desired_temperature
     */
    inline void fixup_temperature(T desired_temperature) noexcept {
        update_kinetic_energy_and_temp();
        const T coeff = sycl::sqrt(degrees_of_freedom() * configuration_.constante_R * desired_temperature / kinetic_energy_);
        backend_.apply_multiplicative_correction_to_momentums(coeff);
        update_kinetic_energy_and_temp();
    }

    /**
     * Applies the Berendsen thermostate on the current system using the current kinetic temperature.
     */
    inline void apply_berendsen_thermostate() noexcept {
        if (!configuration_.use_berdensten_thermostate) { return; }
        if (simulation_idx_ % configuration_.m_step != 0 || simulation_idx_ == 0) { return; }
        update_kinetic_energy_and_temp();
        //const T coeff = config_.gamma * sycl::sqrt(std::abs((config_.T0 / kinetic_temperature_) - 1));
        const T coeff = configuration_.gamma * ((configuration_.T0 / kinetic_temperature_) - 1);
        backend_.apply_multiplicative_correction_to_momentums(1 + coeff);   // momentum += momentum * coeff
        std::cout << "[Berendsen] Thermostate applied with coeff: " << coeff << std::endl;
    }

private:
    const configuration<T> configuration_;   //
    size_t simulation_idx_;                  //
    coordinate<T> forces_sum_;               //
    T lennard_jones_energy_;                 //
    pdb_writer writer_;                      //
    backend<T> backend_;                     //


    // Mutable variables are the variables that are easily and often re-computed based on the simulation
    // they should be updated as often as possible, but don't really affect the simulation at all
    mutable T kinetic_temperature_;          //
    mutable T kinetic_energy_;               //
    mutable double avg_iter_duration_ = 1;   //
    mutable double total_energy_ = 0;        //
    mutable double avg_delta_energy_ = 0;    //
};

}   // namespace sim