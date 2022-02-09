#pragma once

#include <backend/backend_interface.h>

#include "domain_decomposition.hpp"

/**
 * For the moment only the Domain Decomposition kernel runs on OpenMP
 */
namespace sim {
template<bool use_domain_decomposition, typename T> class cpu_backend : backend_interface<T> {

public:
    void set_particles(const std::vector<coordinate<T>>& particules, const configuration<T>& config) override;

    void set_speeds(const std::vector<coordinate<T>>& speeds, const configuration<T>& config) override;

    void randinit_momentums(T min, T max) override;

    void store_particles_coordinates(pdb_writer& writer, size_t i, T temp, T epot) const override;

    [[nodiscard]] T get_momentums_squared_norm() const override;

    void apply_multiplicative_correction_to_momentums(T coeff) override;

    void center_kinetic_momentums() override;

    [[nodiscard]] coordinate<T> mean_kinetic_momentum() const override;

    [[nodiscard]] inline size_t get_particles_count() const override { return size_; }

    void run_velocity_verlet(const configuration<T>& config) override;

    void update_lennard_jones_field(const configuration<T>& config) override;

    /**
     * The CPU backends updates these values on each iteration so there's no need to recompute them.
     * @return
     */
    [[nodiscard]] std::tuple<coordinate<T>, T> get_last_lennard_jones_metrics() const override;

private:
    size_t size_{};
    std::vector<coordinate<T>> coordinates_{};   //
    std::vector<coordinate<T>> momentums_{};     // Vi * mi
    std::vector<coordinate<T>> forces_{};        // Lennard Jones Field
    std::vector<T> energies_{};                  // Lennard Jones Field
    domain_decomposer<T, use_domain_decomposition> decomposer_{};
};


template<typename T> using cpu_backend_regular = cpu_backend<false, T>;

template<typename T> using cpu_backend_decompose = cpu_backend<true, T>;


}   // namespace sim
