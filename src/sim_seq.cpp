#include <sim_seq.h>

template<typename T, int n_sym>
static inline auto internal_simulator_sequential(const std::vector<coordinate<T>> &particules, simulation_configuration<T> config) {
    auto forces = std::vector<coordinate<T>>(particules.size());
    auto summed_forces = coordinate<T>{};
    auto energy = T{};

    for (auto i = 0U; i < particules.size(); ++i) {
        auto this_particule_energy = T{};
        const auto this_particule = particules[i];
        for (auto j = 0U; j < particules.size(); ++j) {
            for (const auto &sym: get_symetries<n_sym, T>()) {
                if (i == j && sym.x() == 0 && sym.y() == 0 && sym.z() == 0) continue;
                const auto other_particule = config.L_ * sym + particules[j];
                T squared_distance = compute_squared_distance(this_particule, other_particule);
                if (config.use_cutoff && squared_distance > integral_power<2>(config.r_cut_)) continue;
                T frac_pow_2 = config.r_star_ * config.r_star_ / squared_distance;
                auto frac_pow_6 = integral_power<3>(frac_pow_2);
                this_particule_energy += integral_power<2>(frac_pow_6) - 2 * frac_pow_6;
                auto force_prefactor = (frac_pow_6 - 1.) * frac_pow_6 * frac_pow_2;
                forces[i] += (this_particule - other_particule) * force_prefactor;
            }
        }
        energy += 2 * config.epsilon_star_ * this_particule_energy; //We divided because the energies would be counted twice otherwise
        summed_forces += forces[i] * (-48) * config.epsilon_star_;
    }
    return std::tuple(forces, summed_forces, energy);
}

template<typename T>
std::tuple<std::vector<coordinate<T>>, coordinate<T>, T> run_simulation_sequential(const std::vector<coordinate<T>> &particules, simulation_configuration<T> config) {
    if (config.n_symetries == 1) {
        return internal_simulator_sequential<T, 1>(particules, config);
    } else if (config.n_symetries == 27) {
        return internal_simulator_sequential<T, 27>(particules, config);
    } else if (config.n_symetries == 125) {
        return internal_simulator_sequential<T, 125>(particules, config);
    } else {
        throw std::runtime_error("Unsupported");
    }
}

template std::tuple<std::vector<coordinate<sycl::half>>, coordinate<sycl::half>, sycl::half>
run_simulation_sequential(const std::vector<coordinate<sycl::half>> &particules, simulation_configuration<sycl::half> config);

template std::tuple<std::vector<coordinate<float>>, coordinate<float>, float>
run_simulation_sequential(const std::vector<coordinate<float>> &particules, simulation_configuration<float> config);

template std::tuple<std::vector<coordinate<double>>, coordinate<double>, double>
run_simulation_sequential(const std::vector<coordinate<double>> &particules, simulation_configuration<double> config);
