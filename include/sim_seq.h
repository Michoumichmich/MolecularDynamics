#pragma once

#include <internal/sim_common.hpp>


template<typename T> class simulation_state {
private:
    const simulation_configuration<T> config_;
    size_t simulation_idx;
    std::vector<coordinate<T>> coordinates_;
    std::vector<coordinate<T>> momentums_;   // Vi * mi
    std::vector<coordinate<T>> forces_;      // Lennard Jones Field
    coordinate<T> summed_forces_;
    T lennard_jones_energy_;
    T temperature_;
    T kinetic_energy_;


private:
    void update_kinetic_energy_and_temp() noexcept;

public:
    simulation_state(const std::vector<coordinate<T>>& particules, simulation_configuration<T> config);

    void run_iter();
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
