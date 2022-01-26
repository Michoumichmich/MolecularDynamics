#pragma once

#include <internal/pdb_writer.hpp>
#include <internal/sim_common.hpp>
#include <iomanip>

/**
 *
 * @tparam T
 */
template<typename T> class simulation_state {
private:
    const simulation_configuration<T> config_;   //
    size_t simulation_idx;                       //
    std::vector<coordinate<T>> coordinates_;     //
    std::vector<coordinate<T>> momentums_;       // Vi * mi
    std::vector<coordinate<T>> forces_;          // Lennard Jones Field
    coordinate<T> forces_sum_;                   //
    T lennard_jones_energy_;                     //
    pdb_writer out;                              //

    // Mutable variables are the variables that are easily and often re-computed based on the simulation
    // they should be updated as often as possible, but don't really affect the simulation at all
    mutable T kinetic_temperature_;         //
    mutable T kinetic_energy_;              //
    mutable double avg_iter_duration = 1;   //
    mutable double total_energy = 0;        //
    mutable double avg_delta_energy = 0;    //

private:
    /**
     *
     */
    void update_kinetic_energy_and_temp() const noexcept;

    /**
     *
     * @return
     */
    [[nodiscard]] inline size_t degrees_of_freedom() const { return 3 * coordinates_.size() - 3; }

    void update_energy_metrics() const {
        constexpr static double aging_coeff = 0.01;
        double prev_energy = total_energy;
        total_energy = kinetic_energy_ + lennard_jones_energy_;
        avg_delta_energy = ((total_energy - prev_energy) * aging_coeff + avg_delta_energy) / (1 + aging_coeff);
    }

    /**
     *
     * @param desired_temp
     */
    void fixup_temperature(T desired_temp) noexcept;

    /**
     *
     */
    void fixup_kinetic_momentums() noexcept;

    void apply_berendsen_thermostate() noexcept;

public:
    /**
     *
     * @param particules
     * @param config
     */
    simulation_state(const std::vector<coordinate<T>>& particules, simulation_configuration<T> config);

    /**
     *
     */
    void run_iter();

    [[nodiscard]] coordinate<T> compute_mean_kinetic_momentum() const noexcept;

    friend std::ostream& operator<<(std::ostream& os, const simulation_state& state) {
        auto mean_kinetic_momentum = state.compute_mean_kinetic_momentum();
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

#ifdef BUILD_HALF
extern template class simulation_state<sycl::half>;
#endif


#ifdef BUILD_FLOAT
extern template class simulation_state<float>;
#endif


#ifdef BUILD_DOUBLE
extern template class simulation_state<double>;
#endif
