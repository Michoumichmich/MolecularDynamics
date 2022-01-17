#include <sim_seq.h>


/**
 *
 * @tparam T Floating point type
 * @param particules std::vector of particules on the host
 * @param config Simulation configuration
 * @return tuple with the forces, summed_forces and the energy
 */
template<typename T, int n_sym>
static inline auto compute_lennard_jones_field_inplace_sequential(const std::vector<coordinate<T>>& particules,   //
                                                                  simulation_configuration<T> config,             //
                                                                  std::vector<coordinate<T>>& forces) noexcept {
    auto summed_forces = coordinate<T>{};
    auto energy = T{};

    for (auto i = 0U; i < particules.size(); ++i) {
        auto this_particule_energy = T{};
        const auto this_particule = particules[i];
        for (auto j = 0U; j < particules.size(); ++j) {
            for (const auto& sym: get_symetries<n_sym>()) {
                if (i == j && sym.x() == 0 && sym.y() == 0 && sym.z() == 0) continue;
                const coordinate<T> delta{sym.x() * config.L_, sym.y() * config.L_, sym.z() * config.L_};
                const auto other_particule = delta + particules[j];
                T squared_distance = compute_squared_distance(this_particule, other_particule);
                if (config.use_cutoff && squared_distance > integral_power<2>(config.r_cut_)) continue;

                if constexpr (std::is_same_v<T, sycl::half>) {
                    if (squared_distance == T{}) continue;
                }

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
    return std::tuple(summed_forces, energy);
}

template<typename T>
static inline auto compute_lennard_jones_field(const std::vector<coordinate<T>>& particules, simulation_configuration<T> config, std::vector<coordinate<T>>& forces) {
    if (config.n_symetries == 1) {
        return compute_lennard_jones_field_inplace_sequential<T, 1>(particules, config, forces);
    } else if (config.n_symetries == 27) {
        return compute_lennard_jones_field_inplace_sequential<T, 27>(particules, config, forces);
    } else {
        throw std::runtime_error("Unsupported");
    }
}

template<typename T, int n_sym>
static inline auto velocity_verlet_sequential(std::vector<coordinate<T>>& particules,   //
                                              std::vector<coordinate<T>>& forces,       //
                                              std::vector<coordinate<T>>& momentums,    //
                                              simulation_configuration<T> config) noexcept {

    const size_t N = particules.size();

    // First step: half step update of the momentums.
    for (size_t i = 0; i < N; ++i) { momentums[i] += forces[i] * config.dt / 2; }

    // Second step: update particules positions
    for (size_t i = 0; i < N; ++i) { particules[i] += config.dt * (momentums[i] / config.m_i); }

    auto [sum, energy] = compute_lennard_jones_field_inplace_sequential<T, n_sym>(particules, config, forces);

    // Last step: update momentums given new forces
    for (size_t i = 0; i < N; ++i) { momentums[i] += forces[i] * config.dt / 2; }
    return std::tuple{sum, energy};
}

/**
 * Runs an iteration of the Velocity Verlet algorithm.
 * @tparam T
 * @param particules
 * @param forces
 * @param momentums
 * @param config
 */
template<typename T>
static inline auto run_velocity_verlet_sequential(std::vector<coordinate<T>>& particules,   //
                                                  std::vector<coordinate<T>>& forces,       //
                                                  std::vector<coordinate<T>>& momentums,    //
                                                  simulation_configuration<T> config) {
    if (config.n_symetries == 1) {
        return velocity_verlet_sequential<T, 1>(particules, forces, momentums, config);
    } else if (config.n_symetries == 27) {
        return velocity_verlet_sequential<T, 27>(particules, forces, momentums, config);
    } else {
        throw std::runtime_error("Unsupported");
    }
}


template<typename T>
simulation_state<T>::simulation_state(const std::vector<coordinate<T>>& particules, simulation_configuration<T> config)
    : config_(config),                                             //
      simulation_idx(0),                                           //
      coordinates_(particules),                                    //
      momentums_(std::vector<coordinate<T>>(particules.size())),   //
      forces_(std::vector<coordinate<T>>(particules.size())) {

    // Initializes the forces field.
    auto [sums, energy] = compute_lennard_jones_field(coordinates_, config_, forces_);
    forces_sum_ = sums;
    lennard_jones_energy_ = energy;

    // Randinit momentums
    auto gen = []() { return coordinate<T>(generate_random_value<T>(-1, 1), generate_random_value<T>(-1, 1), generate_random_value<T>(-1, 1)); };
    std::generate(momentums_.begin(), momentums_.end(), gen);
    fixup_kinetic_momentums();
    fixup_temperature(config_.T0);
}

template<typename T> void simulation_state<T>::update_kinetic_energy_and_temp() const noexcept {
    T sum = {};
    for (const auto& momentum: momentums_) { sum += sycl::dot(momentum, momentum); }
    sum /= config_.m_i;
    kinetic_energy_ = sum / (2 * config_.conversion_force);
    temperature_ = kinetic_energy_ / (config_.constante_R * degrees_of_freedom());
}

template<typename T> void simulation_state<T>::run_iter() {
    ++simulation_idx;
    auto [summed_forces, lennard_jones_energy] = run_velocity_verlet_sequential(coordinates_, forces_, momentums_, config_);
    forces_sum_ = summed_forces;
    lennard_jones_energy_ = lennard_jones_energy;
    update_kinetic_energy_and_temp();
    apply_berendsen_termostate();
}

template<typename T> void simulation_state<T>::fixup_temperature(T desired_temperature) {
    update_kinetic_energy_and_temp();
    const T rapport = degrees_of_freedom() * config_.constante_R * desired_temperature / kinetic_energy_;
    for (auto& momentum: momentums_) { momentum *= sycl::sqrt(rapport); }   // TODO sqrt missing in the paper.
    update_kinetic_energy_and_temp();
}

template<typename T> void simulation_state<T>::fixup_kinetic_momentums() {
    coordinate<T> sum{};
    for (const auto& momentum: momentums_) { sum += momentum; }
    for (auto& momentum: momentums_) { momentum -= sum; }
}

template<typename T> void simulation_state<T>::apply_berendsen_termostate() {
    if (simulation_idx % config_.m_step != 0 || simulation_idx == 0) { return; }
    const T coeff = config_.gamma * ((config_.T0 / temperature_) - 1);
    for (auto& momentum: momentums_) { momentum += momentum * coeff; }
    update_kinetic_energy_and_temp();
}

template<typename T> coordinate<T> simulation_state<T>::compute_barycenter() const noexcept {
    coordinate<T> sum{};   // Sum of vi * mi;
    for (const auto& momentum: momentums_) { sum += momentum; }
    return sum / (momentums_.size() * config_.m_i);
}

#ifdef BUILD_HALF
template class simulation_state<sycl::half>;
#endif

#ifdef BUILD_FLOAT
template class simulation_state<float>;
#endif

#ifdef BUILD_DOUBLE
template class simulation_state<double>;
#endif
