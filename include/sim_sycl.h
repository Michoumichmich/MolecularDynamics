#pragma once

#include <sim_common.hpp>
#include <span_helper.hpp>
#include <tuple>
#include <vector>

#ifdef SYCL_IMPLEMENTATION_ONEAPI

#include <sycl/ext/intel/fpga_device_selector.hpp>

#endif


/**
 *
 * @tparam T Floating point type
 * @param q sycl queue to run the kernel
 * @param particules vector of particles that are allocated on the device
 * @param forces empty vector of forces allocated on the device
 * @param config simulation configuration
 * @param evt input sycl event on which the kernel will depends
 * @return tuple with the forces, summed_forces and the energy
 */
template<typename T>
std::tuple<coordinate<T>, T> run_simulation_sycl_device_memory(   //
        sycl::queue& q,                                           //
        std::span<coordinate<T>> particules,                      //
        std::span<coordinate<T>> forces,                          //
        simulation_configuration<T> config,                       //
        sycl::event evt = {});

#ifdef BUILD_HALF
extern template std::tuple<coordinate<sycl::half>, sycl::half> run_simulation_sycl_device_memory<sycl::half>(   //
        sycl::queue& q,                                                                                         //
        const std::span<coordinate<sycl::half>> particules,                                                     //
        std::span<coordinate<sycl::half>> forces,                                                               //
        simulation_configuration<sycl::half> config,                                                            //
        sycl::event evt);
#endif

#ifdef BUILD_FLOAT
extern template std::tuple<coordinate<float>, float> run_simulation_sycl_device_memory<float>(   //
        sycl::queue& q,                                                                          //
        const std::span<coordinate<float>> particules,                                           //
        std::span<coordinate<float>> forces,                                                     //
        simulation_configuration<float> config,                                                  //
        sycl::event evt);
#endif

extern template std::tuple<coordinate<double>, double> run_simulation_sycl_device_memory<double>(   //
        sycl::queue& q,                                                                             //
        const std::span<coordinate<double>> particules,                                             //
        std::span<coordinate<double>> forces,                                                       //
        simulation_configuration<double> config,                                                    //
        sycl::event evt);


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
run_simulation_sycl(sycl::queue& q, simulation_configuration<T> config, const std::vector<coordinate<T>>& particules_host);

#ifdef BUILD_HALF
extern template std::tuple<std::vector<coordinate<sycl::half>>, coordinate<sycl::half>, sycl::half>   //
run_simulation_sycl<sycl::half>(sycl::queue& q, simulation_configuration<sycl::half> config, const std::vector<coordinate<sycl::half>>& particules_host);
#endif

#ifdef BUILD_FLOAT
extern template std::tuple<std::vector<coordinate<float>>, coordinate<float>, float>   //
run_simulation_sycl<float>(sycl::queue& q, simulation_configuration<float> config, const std::vector<coordinate<float>>& particules_host);
#endif

extern template std::tuple<std::vector<coordinate<double>>, coordinate<double>, double>   //
run_simulation_sycl<double>(sycl::queue& q, simulation_configuration<double> config, const std::vector<coordinate<double>>& particules_host);
