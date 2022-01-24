#pragma once

#include <internal/sim_common.hpp>
#include <iomanip>

/**
 *
 * @tparam T
 */
template<typename T> class simulation_state {
private:
    const simulation_configuration<T> config_;
    size_t simulation_idx;
    std::vector<coordinate<T>> coordinates_;
    std::vector<coordinate<T>> momentums_;   // Vi * mi
    std::vector<coordinate<T>> forces_;      // Lennard Jones Field
    coordinate<T> forces_sum_;
    T lennard_jones_energy_;
    T kinetic_temperature_;
    T kinetic_energy_;


private:
    /**
     *
     */
    void update_kinetic_energy_and_temp() noexcept;

    /**
     *
     * @return
     */
    [[nodiscard]] inline size_t degrees_of_freedom() const { return 3 * coordinates_.size() - 3; }

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

    coordinate<T> compute_mean_kinetic_momentum() const noexcept;

    friend std::ostream& operator<<(std::ostream& os, const simulation_state& state) {
        auto mean_kinetic_momentum = state.compute_mean_kinetic_momentum();
        os << std::setprecision(10) << "[" << state.simulation_idx << "] "                          //
           << "E_tot: " << state.kinetic_energy_ + state.lennard_jones_energy_                      //
           << ", Temp: " << state.kinetic_temperature_                                              //
           << ", E_c: " << state.kinetic_energy_                                                    //
           << ", E_pot: " << state.lennard_jones_energy_                                            //
           << ", field_sum_norm: " << sycl::sqrt(sycl::dot(state.forces_sum_, state.forces_sum_))   //
           << ", barycenter_speed_norm " << sycl::sqrt(sycl::dot(mean_kinetic_momentum, mean_kinetic_momentum)) / state.config_.m_i << '\n';
        // for (size_t i = 0; i <10; ++i) { os << state.coordinates_[i].x() << ' '; } os << '\n';
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
