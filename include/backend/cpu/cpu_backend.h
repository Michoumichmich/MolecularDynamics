#pragma once

#include "backend/backend_interface.h"

namespace sim {
template<typename T> class cpu_backend : backend_interface<T> {

public:
    void init_backend(const std::vector<coordinate<T>>& particules) override;

    void randinit_momentums(T min, T max) override;

    void store_particules_coordinates(pdb_writer& writer, size_t i) const override;

    [[nodiscard]] T get_momentums_squared_norm() const override;

    void apply_multiplicative_correction_to_momentums(T coeff) override;

    void center_kinetic_momentums() override;

    [[nodiscard]] coordinate<T> mean_kinetic_momentums() const override;

    [[nodiscard]] inline size_t get_particules_count() const override { return size_; }

    void run_velocity_verlet(const configuration<T>& config) override;

    void init_lennard_jones_field(const configuration<T>& config) override;

    /**
     * The CPU backends updates these values on each iteration so there's no need to recompute them.
     * @return
     */
    [[nodiscard]] inline std::tuple<coordinate<T>, T> get_last_lennard_jones_metrics() const override { return last_lennard_jones_metrics_; }

private:
    size_t size_{};
    std::tuple<coordinate<T>, T> last_lennard_jones_metrics_{};
    std::vector<coordinate<T>> coordinates_{};   //
    std::vector<coordinate<T>> momentums_{};     // Vi * mi
    std::vector<coordinate<T>> forces_{};        // Lennard Jones Field
};

#ifdef BUILD_DOUBLE
extern template class cpu_backend<double>;
#endif

#ifdef BUILD_FLOAT
extern template class cpu_backend<float>;
#endif

#ifdef BUILD_HALF
extern template class cpu_backend<sycl::half>;
#endif


}   // namespace sim
