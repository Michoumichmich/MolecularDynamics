#pragma once

#include <backends/cpu_backend.hpp>
#include <internal/pdb_writer.hpp>
#include <internal/sim_common.hpp>
#include <iomanip>


/**
 *
 * @tparam T
 */
template<typename T> class simulation_runner {
private:
    const simulation_configuration<T> config_;   //
    size_t simulation_idx;                       //
    coordinate<T> forces_sum_;                   //
    T lennard_jones_energy_;                     //
    pdb_writer out;                              //
    cpu_backend<T> backend;


    // Mutable variables are the variables that are easily and often re-computed based on the simulation
    // they should be updated as often as possible, but don't really affect the simulation at all
    mutable T kinetic_temperature_;         //
    mutable T kinetic_energy_;              //
    mutable double avg_iter_duration = 1;   //
    mutable double total_energy = 0;        //
    mutable double avg_delta_energy = 0;    //

private:
    /**
     * Updates the kinectic energy and temperature based on current momentums.
     * @tparam T
     */
    void update_kinetic_energy_and_temp() const noexcept {
        T sum = backend.get_momentums_squared_norm();
        kinetic_energy_ = sum / (2 * config_.conversion_force * config_.m_i);
        kinetic_temperature_ = kinetic_energy_ / (config_.constante_R * degrees_of_freedom());
    }

    /**
     *
     * @return
     */
    [[nodiscard]] inline size_t degrees_of_freedom() const { return 3 * backend.get_particules_count() - 3; }

    void update_energy_metrics() const {
        constexpr static double aging_coeff = 0.01;
        double prev_energy = total_energy;
        total_energy = kinetic_energy_ + lennard_jones_energy_;
        avg_delta_energy = ((total_energy - prev_energy) * aging_coeff + avg_delta_energy) / (1 + aging_coeff);
    }

    /**
     * Updates the momentums to match the desierd kinetic temperature.
     * @tparam T
     * @param desired_temperature
     */
    void fixup_temperature(T desired_temperature) noexcept {
        update_kinetic_energy_and_temp();
        const T rapport = sycl::sqrt(degrees_of_freedom() * config_.constante_R * desired_temperature / kinetic_energy_);
        backend.apply_multiplicative_correction_to_momentums(rapport);
        update_kinetic_energy_and_temp();
    }

    /**
     * Fixes the kinetic momentums in a way that the barycenter does not move.
     * @tparam T
     */
    void fixup_kinetic_momentums() noexcept { backend.center_kinetic_momentums(); }

    /**
     * Applies the Berendsen thermostate on the current system using the current kinetic temperature.
     */
    void apply_berendsen_thermostate() noexcept {
        if constexpr (!simulation_configuration<T>::use_berdensten_thermostate) {
            return;
        } else {
            if (simulation_idx % config_.m_step != 0 || simulation_idx == 0) { return; }
            update_kinetic_energy_and_temp();
            //const T coeff = config_.gamma * sycl::sqrt(std::abs((config_.T0 / kinetic_temperature_) - 1));
            const T coeff = config_.gamma * ((config_.T0 / kinetic_temperature_) - 1);
            backend.apply_multiplicative_correction_to_momentums(1 + coeff);   // momentum += momentum * coeff
            std::cout << "[Berendsen] Thermostate applied with coeff: " << coeff << std::endl;
        }
    }

public:
    /**
     * Constructor, initializes the system.
     * @param particules
     * @param config
     */
    simulation_runner(const std::vector<coordinate<T>>& particules, simulation_configuration<T> config)
        : config_(config),     //
          simulation_idx(0),   //
          out(config.out_file) {

        backend.set_particules_coordinates(std::move(particules));

        // Initializes the forces field.
        auto [sums, energy] = backend.init_lennard_jones_field(config);
        forces_sum_ = sums;
        lennard_jones_energy_ = energy;

        // Randinit momentums
        backend.randinit_momentums(-1, 1);

        // Fixes the barycenter.
        fixup_kinetic_momentums();

        // Computes the temperature of the systme and scales the momentums to reach the target temperature.
        fixup_temperature(config_.T0);
        backend.store_particules_coordinates(out, simulation_idx);
        update_energy_metrics();
    }

    /**
     *
     */
    void run_iter() {
        auto begin = std::chrono::high_resolution_clock::now();
        ++simulation_idx;

        // Running the velocity verlet algorithm and getting back the potential energy as well as the sum of the forces.
        auto [summed_forces, lennard_jones_energy] = backend.run_velocity_verlet(config_);
        forces_sum_ = summed_forces;
        lennard_jones_energy_ = lennard_jones_energy;

        // Update and fixe up the temperature
        apply_berendsen_thermostate();
        update_kinetic_energy_and_temp();

        if (simulation_idx % config_.iter_per_frame == 0) { backend.store_particules_coordinates(out, simulation_idx); }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()) * 1e-9;
        avg_iter_duration = (avg_iter_duration + duration) / 2;
        update_energy_metrics();
    }


    friend std::ostream& operator<<(std::ostream& os, const simulation_runner& state) {
        auto mean_kinetic_momentum = std::sqrt(state.backend.get_momentums_squared_norm()) / state.backend.get_particules_count();
        os << std::setprecision(10) << "[" << state.simulation_idx << "] "                                                                   //
           << "E_tot: " << std::setw(13) << std::setfill(' ') << state.total_energy                                                          //
           << ", Temp: " << std::setw(13) << std::setfill(' ') << state.kinetic_temperature_                                                 //
           << ", E_c: " << std::setw(13) << std::setfill(' ') << state.kinetic_energy_                                                       //
           << ", E_pot: " << std::setw(13) << std::setfill(' ') << state.lennard_jones_energy_                                               //
           << ", Field_sum_norm: " << std::setw(13) << std::setfill(' ') << sycl::length(state.forces_sum_)                                  //
           << ", Barycenter_speed_norm: " << std::setw(13) << std::setfill(' ') << sycl::length(mean_kinetic_momentum) / state.config_.m_i   //
           << ", Avg_delta_energy: " << std::setprecision(5) << state.avg_delta_energy                                                       //
           << ", Time: " << state.config_.dt * state.simulation_idx << " fs"                                                                 //
           << ", Speed: " << std::setprecision(3) << 1.0 / state.avg_iter_duration << " iter/s."                                             //
           << '\n';
        return os;
    }
};
