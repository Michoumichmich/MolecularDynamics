#pragma once

#include "sim_config.hpp"

/**
 *
 * @tparam T Floating point type
 * @param q sycl queue to run the kernel
 * @param particules_device_in vector of particles that are allocated on the device
 * @param forces_device_out empty vector of forces allocated on the device
 * @param config simulation configuration
 * @param evt input sycl event on which the kernel will depends
 * @return tuple with the forces, summed_forces and the energy
 */
template<typename T>
std::tuple<coordinate<T>, T> run_simulation_sycl_device_memory(                                      //
        sycl::queue& q,                                                                              //
        std::span<coordinate<T>> particules_device_in, std::span<coordinate<T>> forces_device_out,   //
        configuration<T> config,                                                                     //
        sycl::event evt = {});

#ifdef BUILD_HALF
extern template std::tuple<coordinate<sycl::half>, sycl::half> run_simulation_sycl_device_memory<sycl::half>(   //
        sycl::queue& q,                                                                                         //
        const std::span<coordinate<sycl::half>> particules, std::span<coordinate<sycl::half>> forces,           //
        simulation_configuration<sycl::half> config,                                                            //
        sycl::event evt);
#endif

#ifdef BUILD_FLOAT
extern template std::tuple<coordinate<float>, float> run_simulation_sycl_device_memory<float>(                     //
        sycl::queue& q,                                                                                            //
        const std::span<coordinate<float>> particules_device_in, std::span<coordinate<float>> forces_device_out,   //
        configuration<float> config,                                                                               //
        sycl::event evt);
#endif

#ifdef BUILD_DOUBLE
extern template std::tuple<coordinate<double>, double> run_simulation_sycl_device_memory<double>(                    //
        sycl::queue& q,                                                                                              //
        const std::span<coordinate<double>> particules_device_in, std::span<coordinate<double>> forces_device_out,   //
        configuration<double> config,                                                                                //
        sycl::event evt);
#endif

/**
 *
 * @tparam T Floating point type
 * @param q sycl queue to run the kernel on
 * @param config simulation configuration
 * @param particules_host std::vector of particules allocaed on the host
 * @return tuple with the forces, summed_forces and the energy
 */
template<typename T>
std::tuple<std::vector<coordinate<T>>, coordinate<T>, T>   //
run_simulation_sycl(sycl::queue& q, configuration<T> config, const std::vector<coordinate<T>>& particules_host);

#ifdef BUILD_HALF
extern template std::tuple<std::vector<coordinate<sycl::half>>, coordinate<sycl::half>, sycl::half>   //
run_simulation_sycl<sycl::half>(sycl::queue& q, simulation_configuration<sycl::half> config, const std::vector<coordinate<sycl::half>>& particules_host);
#endif

#ifdef BUILD_FLOAT
extern template std::tuple<std::vector<coordinate<float>>, coordinate<float>, float>   //
run_simulation_sycl<float>(sycl::queue& q, configuration<float> config, const std::vector<coordinate<float>>& particules_host);
#endif

#ifdef BUILD_DOUBLE
extern template std::tuple<std::vector<coordinate<double>>, coordinate<double>, double>   //
run_simulation_sycl<double>(sycl::queue& q, configuration<double> config, const std::vector<coordinate<double>>& particules_host);
#endif