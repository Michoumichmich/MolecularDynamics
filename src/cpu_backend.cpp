#include <backend/cpu_backend.hpp>

namespace sim {
/**
 *
 * @tparam T Floating point type
 * @param particules std::vector of particules on the host
 * @param config Simulation configuration
 * @return tuple with the forces, summed_forces and the energy
 */
template<typename T, int n_sym>
static inline std::tuple<coordinate<T>, T> compute_lennard_jones_field_inplace_sequential(   //
        const std::vector<coordinate<T>>& particules,                                        //
        const configuration<T>& config,                                                      //
        std::vector<coordinate<T>>& forces) {
    auto summed_forces = coordinate<T>{};
    auto energy = T{};

#pragma omp declare reduction(+ : coordinate <T> : omp_out += omp_in)   // initializer(omp_priv = coordinate <double>{}))

#pragma omp parallel for default(none) shared(particules, config, forces) reduction(+ : energy, summed_forces)
    for (auto i = 0U; i < particules.size(); ++i) {
        forces[i] = {};
        auto this_particule_energy = T{};
        const auto this_particule = particules[i];
        for (auto j = 0U; j < particules.size(); ++j) {

#pragma unroll
            for (const auto& sym: get_symetries<n_sym>()) {
                if (i == j && sym.x() == 0 && sym.y() == 0 && sym.z() == 0) continue;
                const coordinate<T> delta{sym.x() * config.L_, sym.y() * config.L_, sym.z() * config.L_};
                const auto other_particule = delta + particules[j];
                T squared_distance = compute_squared_distance(this_particule, other_particule);
                if (config.use_cutoff && squared_distance > integral_power<2>(config.r_cut_)) continue;

                internal::assume(squared_distance != T{});
                //if (squared_distance == T{}) { throw std::runtime_error("Got null distance"); }

                const T frac_pow_2 = config.r_star_ * config.r_star_ / squared_distance;
                const T frac_pow_6 = integral_power<3>(frac_pow_2);
                this_particule_energy += integral_power<2>(frac_pow_6) - 2 * frac_pow_6;
                const T force_prefactor = (frac_pow_6 - 1.) * frac_pow_6 * frac_pow_2;
                forces[i] += (this_particule - other_particule) * force_prefactor;
            }
        }
        energy += 2 * config.epsilon_star_ * this_particule_energy;   //We divided because the energies would be counted twice otherwise
        summed_forces += forces[i] * (-48) * config.epsilon_star_;
    }
    //std::cout << "Energy: " << energy << std::endl;
    return std::tuple(summed_forces, energy);
}


/**
 * Runs one iteration of the velocity verlet algorithm.
 * @tparam T float type
 * @tparam n_sym 1 or 27
 * @param particules inout
 * @param forces inout
 * @param momentums inout
 * @param config
 * @return std::tuple{sum, energy}
 */
template<typename T, int n_sym>
static inline std::tuple<coordinate<T>, T> velocity_verlet_sequential(   //
        std::vector<coordinate<T>>& particules,                          //
        std::vector<coordinate<T>>& forces,                              //
        std::vector<coordinate<T>>& momentums,                           //
        const configuration<T>& config) noexcept {

    internal::assume(particules.size() == forces.size() && forces.size() == momentums.size());
    const size_t N = particules.size();

    // First step: half step update of the momentums.
    for (size_t i = 0; i < N; ++i) { momentums[i] += config.conversion_force * forces[i] * config.dt / 2; }

    // Second step: update particules positions
    for (size_t i = 0; i < N; ++i) { particules[i] += config.dt * momentums[i] / config.m_i; }

    compute_lennard_jones_field_inplace_sequential<T, n_sym>(particules, config, forces);

    // Last step: update momentums given new forces
    for (size_t i = 0; i < N; ++i) { momentums[i] += config.conversion_force * forces[i] * config.dt / 2; }

    auto [sum, energy] = compute_lennard_jones_field_inplace_sequential<T, n_sym>(particules, config, forces);
    return std::tuple{sum, energy};
}


/**
 * Runs an iteration of the Velocity Verlet algorithm.
 * @tparam T
 * @param config
 * @return
 */
template<typename T> std::tuple<coordinate<T>, T> cpu_backend<T>::run_velocity_verlet(const configuration<T>& config) {
    if (config.n_symetries == 1) {
        return velocity_verlet_sequential<T, 1>(coordinates_, forces_, momentums_, config);
    } else if (config.n_symetries == 27) {
        return velocity_verlet_sequential<T, 27>(coordinates_, forces_, momentums_, config);
    } else {
        throw std::runtime_error("Unsupported");
    }
}


/**
 *
 * @tparam T
 * @param config
 * @return
 */
template<typename T> std::tuple<coordinate<T>, T> cpu_backend<T>::init_lennard_jones_field(const configuration<T>& config) {
    if (config.n_symetries == 1) {
        return compute_lennard_jones_field_inplace_sequential<T, 1>(coordinates_, config, forces_);
    } else if (config.n_symetries == 27) {
        return compute_lennard_jones_field_inplace_sequential<T, 27>(coordinates_, config, forces_);
    } else {
        throw std::runtime_error("Unsupported");
    }
}

template<typename T> void cpu_backend<T>::randinit_momentums(T min, T max) {
    std::generate(momentums_.begin(), momentums_.end(), [=]() {   //
        return coordinate<T>(internal::generate_random_value<T>(min, max), internal::generate_random_value<T>(min, max), internal::generate_random_value<T>(min, max));
    });
}


#ifdef BUILD_HALF
template class cpu_backend<sycl::half>;
#endif

#ifdef BUILD_FLOAT
template class cpu_backend<float>;
#endif

#ifdef BUILD_DOUBLE
template class cpu_backend<double>;
#endif

}   // namespace sim
