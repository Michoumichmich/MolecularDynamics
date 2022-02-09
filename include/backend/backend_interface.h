#pragma once

#include <internal/pdb_writer.hpp>
#include <sim_config.hpp>

namespace sim {

/**
 * Stateful backend.
 * Not ideal but makes ressource managmeent easier (especially for GPUs).
 *
 * This clases is used to enforce an interface. Later we want not to use virtual inheritence as it prevents regular
 * C++ the compiler to inline (and optimize) "kernels" (in this code at least...). So we will be doing template metaprogramming instead.
 * @tparam T
 */
template<typename T> class backend_interface {
public:
#ifdef DEBUG_INTERFACE
    virtual void set_particles(const std::vector<coordinate<T>>& particles, const configuration<T>& config) = 0;

    virtual void set_speeds(const std::vector<coordinate<T>>& speeds, const configuration<T>& config) = 0;


    /**
     * Initializes the momentums randomly
     * @param min minimum value for each component
     * @param max maximum value for each component
     */
    virtual void randinit_momentums(T min, T max) = 0;

    /**
     * Stores the particles in the writer.
     * @param writer
     * @param i
     */
    //virtual void store_particules_coordinates(pdb_writer& writer, size_t i) const = 0;
    virtual void store_particles_coordinates(pdb_writer& writer, size_t i, T temp, T epot) const = 0;

    /**
     *
     * @return
     */
    virtual T get_momentums_squared_norm() const = 0;

    /**
     *
     * @param coeff
     */
    virtual void apply_multiplicative_correction_to_momentums(T coeff) = 0;

    /**
     * Fixes the kinetic momentums in a way that the barycenter does not move.
     */
    virtual void center_kinetic_momentums() = 0;

    virtual std::tuple<coordinate<T>, T> get_last_lennard_jones_metrics() const = 0;

    virtual coordinate<T> mean_kinetic_momentum() const = 0;

    /**
     *
     */
    [[nodiscard]] virtual size_t get_particles_count() const = 0;

    /**
     * Returns sum of field and potential energy.
     */
    virtual void run_velocity_verlet(const configuration<T>& config) = 0;

    /**
     * Returns sum of field and potential energy.
     * @param config
     * @return
     */
    virtual void update_lennard_jones_field(const configuration<T>& config) = 0;
#endif
};
}   // namespace sim
