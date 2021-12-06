#pragma once

#include "sim_common.hpp"
#include <tuple>
#include <vector>

/**
 *
 * @tparam T Floating point type
 * @param particules std::vector of particules on the host
 * @param config Simulation configuration
 * @return tuple with the forces, summed_forces and the energy
 */
template<typename T>
std::tuple<std::vector<coordinate<T>>, coordinate<T>, T> run_simulation_sequential(const std::vector<coordinate<T>> &particules, simulation_configuration<T> config);

extern template std::tuple<std::vector<coordinate<sycl::half>>, coordinate<sycl::half>, sycl::half>
run_simulation_sequential(const std::vector<coordinate<sycl::half>> &particules, simulation_configuration<sycl::half> config);

extern template std::tuple<std::vector<coordinate<float>>, coordinate<float>, float>
run_simulation_sequential(const std::vector<coordinate<float>> &particules, simulation_configuration<float> config);

extern template std::tuple<std::vector<coordinate<double>>, coordinate<double>, double>
run_simulation_sequential(const std::vector<coordinate<double>> &particules, simulation_configuration<double> config);
