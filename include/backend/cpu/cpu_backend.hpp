#pragma once
#include "cpu_backend.h"

namespace sim {
/**
 *
 * @tparam T Floating point type
 * @param coordinates std::vector of particules on the host
 * @param config Simulation configuration
 * @return tuple with the forces, summed_forces and the energy
 */
template<typename T, int n_sym>
static inline std::tuple<coordinate<T>, T> update_lennard_jones_field_cpu_impl(   //
        const std::vector<coordinate<T>>& coordinates,                            //
        const configuration<T>& config,                                           //
        std::vector<coordinate<T>>& forces) {
    auto summed_forces = coordinate<T>{};
    auto energy = T{};

#pragma omp declare reduction(+ : coordinate <T> : omp_out += omp_in)   // initializer(omp_priv = coordinate <double>{}))

#pragma omp parallel for default(none) shared(coordinates, config, forces) reduction(+ : energy, summed_forces)
    for (auto i = 0U; i < coordinates.size(); ++i) {
        forces[i] = {};
        auto this_particule_energy = T{};
        const auto this_particule = coordinates[i];
        for (auto j = 0U; j < coordinates.size(); ++j) {

#pragma unroll
            for (const auto& sym: get_symetries<n_sym>()) {
                if (i == j && sym.x() == 0 && sym.y() == 0 && sym.z() == 0) continue;
                const coordinate<T> delta{sym.x() * config.L, sym.y() * config.L, sym.z() * config.L};
                const auto other_particule = delta + coordinates[j];
                T squared_distance = compute_squared_distance(this_particule, other_particule);
                if (config.use_cutoff && squared_distance > integral_power<2>(config.r_cut)) continue;

                internal::assume(squared_distance != T{});
                //if (squared_distance == T{}) { throw std::runtime_error("Got null distance"); }

                const T frac_pow_2 = config.r_star * config.r_star / squared_distance;
                const T frac_pow_6 = integral_power<3>(frac_pow_2);
                this_particule_energy += integral_power<2>(frac_pow_6) - 2 * frac_pow_6;
                const T force_prefactor = (frac_pow_6 - 1.) * frac_pow_6 * frac_pow_2;
                forces[i] += (this_particule - other_particule) * force_prefactor;
            }
        }
        energy += 2 * config.epsilon_star * this_particule_energy;   //We divided because the energies would be counted twice otherwise
        summed_forces += forces[i] * (-48) * config.epsilon_star;
    }
    //std::cout << "Energy: " << energy << std::endl;
    return std::tuple(summed_forces, energy);
}

template<typename T> void cpu_backend<T>::init_lennard_jones_field(const configuration<T>& config) {
    if (config.n_symetries == 1) {
        last_lennard_jones_metrics_ = update_lennard_jones_field_cpu_impl<T, 1>(coordinates_, config, forces_);
    } else if (config.n_symetries == 27) {
        last_lennard_jones_metrics_ = update_lennard_jones_field_cpu_impl<T, 27>(coordinates_, config, forces_);
    } else if (config.n_symetries == 125) {
        last_lennard_jones_metrics_ = update_lennard_jones_field_cpu_impl<T, 125>(coordinates_, config, forces_);
    } else {
        throw std::runtime_error("Unsupported");
    }
}


/**
 * Runs one iteration of the velocity verlet algorithm.
 * @tparam T float type
 * @tparam n_sym 1 or 27
 * @param coordinates inout
 * @param forces inout
 * @param momentums inout
 * @param config
 * @return std::tuple{sum, energy}
 */
template<typename T, int n_sym>
static inline std::tuple<coordinate<T>, T> velocity_verlet_cpu_impl(   //
        std::vector<coordinate<T>>& coordinates,                       //
        std::vector<coordinate<T>>& forces,                            //
        std::vector<coordinate<T>>& momentums,                         //
        const configuration<T>& config) noexcept {

    internal::assume(coordinates.size() == forces.size() && forces.size() == momentums.size());
    const size_t N = coordinates.size();

    // First step: half step update of the momentums.
    for (size_t i = 0; i < N; ++i) { momentums[i] += config.conversion_force * forces[i] * config.dt / 2; }

    // Second step: update particules positions
    for (size_t i = 0; i < N; ++i) { coordinates[i] += config.dt * momentums[i] / config.m_i; }

    update_lennard_jones_field_cpu_impl<T, n_sym>(coordinates, config, forces);

    // Last step: update momentums given new forces
    for (size_t i = 0; i < N; ++i) { momentums[i] += config.conversion_force * forces[i] * config.dt / 2; }

    auto [sum, energy] = update_lennard_jones_field_cpu_impl<T, n_sym>(coordinates, config, forces);
    return std::tuple{sum, energy};
}


template<typename T> void cpu_backend<T>::run_velocity_verlet(const configuration<T>& config) {
    if (config.n_symetries == 1) {
        last_lennard_jones_metrics_ = velocity_verlet_cpu_impl<T, 1>(coordinates_, forces_, momentums_, config);
    } else if (config.n_symetries == 27) {
        last_lennard_jones_metrics_ = velocity_verlet_cpu_impl<T, 27>(coordinates_, forces_, momentums_, config);
    } else if (config.n_symetries == 125) {
        last_lennard_jones_metrics_ = velocity_verlet_cpu_impl<T, 125>(coordinates_, forces_, momentums_, config);
    } else {
        throw std::runtime_error("Unsupported");
    }
}


template<typename T> void cpu_backend<T>::randinit_momentums(T min, T max) {
    std::generate(momentums_.begin(), momentums_.end(), [=]() {   //
        return coordinate<T>(internal::generate_random_value<T>(min, max), internal::generate_random_value<T>(min, max), internal::generate_random_value<T>(min, max));
    });
}

template<typename T> void cpu_backend<T>::store_particules_coordinates(pdb_writer& writer, size_t i) const { writer.store_new_iter(coordinates_, i); }

template<typename T> T cpu_backend<T>::get_momentums_squared_norm() const {
    T sum{};
    for (const auto& momentum: momentums_) { sum += sycl::dot(momentum, momentum); }
    return sum;
}

template<typename T> void cpu_backend<T>::apply_multiplicative_correction_to_momentums(T coeff) {
    for (auto& momentum: momentums_) { momentum *= coeff; }
}

template<typename T> void cpu_backend<T>::center_kinetic_momentums() {
    auto mean = mean_kinetic_momentums();
    for (auto& momentum: momentums_) { momentum -= mean; }
}

template<typename T> coordinate<T> cpu_backend<T>::mean_kinetic_momentums() const {
    coordinate<T> mean{};   // Sum of vi * mi;
    for (const auto& momentum: momentums_) { mean += momentum; }
    return mean / momentums_.size();
}

template<typename T> void cpu_backend<T>::init_backend(const std::vector<coordinate<T>>& particules) {
    size_ = particules.size();
    coordinates_ = particules;
    momentums_ = std::vector<coordinate<T>>(size_);
    forces_ = std::vector<coordinate<T>>(size_);
}


}   // namespace sim