#pragma once

#include <internal/pdb_writer.hpp>
#include <internal/sim_config.hpp>

namespace sim {

template<typename T> class backend_interface {
public:
    virtual void set_particules_coordinates(const std::vector<coordinate<T>>& particules) = 0;

    virtual void randinit_momentums(T min, T max) = 0;

    virtual void store_particules_coordinates(pdb_writer& writer, size_t i) const = 0;

    virtual T get_momentums_squared_norm() const = 0;

    virtual void apply_multiplicative_correction_to_momentums(T coeff) = 0;

    virtual void center_kinetic_momentums() = 0;

    [[nodiscard]] virtual size_t get_particules_count() const = 0;

    // Returns sum of field and potential energy.
    virtual std::tuple<coordinate<T>, T> run_velocity_verlet(const configuration<T>& config) = 0;

    virtual std::tuple<coordinate<T>, T> init_lennard_jones_field(const configuration<T>& config) = 0;
};
}   // namespace sim