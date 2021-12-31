#include <sim_sycl.h>
#include <utility>
namespace internal {

static inline auto compute_range_size(size_t size, size_t work_group_size) {
    return sycl::nd_range<1>(work_group_size * ((size + work_group_size - 1) / work_group_size), work_group_size);
}

template<typename T> static inline void prefetch_constant(const T* ptr) {
#if defined(__NVPTX__) && defined(__SYCL_DEVICE_ONLY__)
    if constexpr (sizeof(ptr) == 8) {
        asm("prefetchu.L1 [%0];" : : "l"(ptr));
    } else {
        asm("prefetchu.L1 [%0];" : : "r"(ptr));
    }
#else
    (void) ptr;
#endif
}

template<typename T, bool multiple_size, int n_sym> class leenard_jones_kernel;

template<typename T, bool multiple_size, int n_sym>
static inline auto internal_simulator_on_sycl(                                        //
        sycl::queue& q, size_t work_group_size,                                       //
        const std::span<coordinate<T>> particules, std::span<coordinate<T>> forces,   //
        simulation_configuration<T> config, sycl::event evt = {}) {
    auto summed_forces = coordinate<T>{};
    auto energy = T{};
    {
        auto x_reduction_buffer = sycl::buffer<T>(&summed_forces.x(), 1U);
        auto y_reduction_buffer = sycl::buffer<T>(&summed_forces.y(), 1U);
        auto z_reduction_buffer = sycl::buffer<T>(&summed_forces.z(), 1U);
        auto energy_reduction_buffer = sycl::buffer<T>(&energy, 1);
        q.submit([&](sycl::handler& cgh) {
             cgh.depends_on(std::move(evt));
#ifdef SYCL_IMPLEMENTATION_HIPSYCL
             auto reduction_x = sycl::reduction(x_reduction_buffer.get_access(cgh), sycl::plus<T>{});
             auto reduction_y = sycl::reduction(y_reduction_buffer.get_access(cgh), sycl::plus<T>{});
             auto reduction_z = sycl::reduction(z_reduction_buffer.get_access(cgh), sycl::plus<T>{});
             auto reduction_energy = sycl::reduction(energy_reduction_buffer.get_access(cgh), sycl::plus<T>{});
#else
             auto reduction_x = sycl::reduction(x_reduction_buffer, cgh, sycl::plus<>{});
             auto reduction_y = sycl::reduction(y_reduction_buffer, cgh, sycl::plus<>{});
             auto reduction_z = sycl::reduction(z_reduction_buffer, cgh, sycl::plus<>{});
             auto reduction_energy = sycl::reduction(energy_reduction_buffer, cgh, sycl::plus<>{});
#endif
             auto particules_tile = sycl::accessor<coordinate<T>, 1, sycl::access_mode::read_write, sycl::access::target::local>(sycl::range<1>(work_group_size), cgh);
             cgh.parallel_for<leenard_jones_kernel<T, multiple_size, n_sym>>(
                     compute_range_size(particules.size(), work_group_size), reduction_x, reduction_y, reduction_z, reduction_energy,
                     [=](const sycl::nd_item<1>& item, auto& reducer_x, auto& reducer_y, auto& reducer_z, auto& reducer_energy) {
                         /* Getting space coordinates */
                         const uint32_t global_id = item.get_global_linear_id();
                         const uint32_t local_id = item.get_local_linear_id();
                         const uint32_t group_count = item.get_group_range().size();
                         const uint32_t group_size = item.get_local_range().size();

                         /* Whether the current work item takes part in the computation or not.
                         * We cannot return as it needs to be present for further barriers. */
                         const bool is_active_work_item = [&]() {
                             if constexpr (multiple_size) return true;
                             else {
                                 return global_id < particules.size();
                             }
                         }();

                         /* Setting up local variables */
                         const auto this_work_item_particule = is_active_work_item ? particules[global_id] : coordinate<T>{};
                         static const auto symetries = get_symetries<n_sym>();
                         /* Local reducers */
                         auto this_particule_energy = T{};
                         auto this_particule_force = coordinate<T>{0, 0, 0};

                         /* Loop over 'how many tiles we need'. Each tile being a sequence of particles loaded into local memory */
                         for (uint32_t tile_id = 0U; tile_id < group_count; ++tile_id) {
                             const uint32_t global_particule_idx = tile_id * group_size + local_id;
                             const bool is_active_tile = [&]() {
                                 if constexpr (multiple_size) return true;
                                 else {
                                     return global_id < particules.size();
                                 }
                             }();
                             const uint32_t this_tile_size = [&]() {
                                 if constexpr (multiple_size) {
                                     return group_size;
                                 } else {
                                     return std::min<uint32_t>(group_size, particules.size() - tile_id * group_size);
                                 }
                             }();

                             /* Loading data (as tiles) into local_memory */
                             const auto new_particule = is_active_tile ? particules[global_particule_idx] : coordinate<T>{};
                             sycl::group_barrier(item.get_group());
                             particules_tile[local_id] = new_particule;
                             sycl::group_barrier(item.get_group());

                             if (!is_active_work_item) continue; /* Current not considered as we're ouf of range */
                             prefetch_constant(particules.data() + global_particule_idx + group_size);

                             /* Doing the computation between our own particule and the ones from the tile */
                             for (uint32_t j = 0U; j < this_tile_size; ++j) {
#if defined(__NVPTX__) && defined(__SYCL_DEVICE_ONLY__)
#    pragma unroll n_sym
#endif
                                 for (const auto& sym: symetries) {
                                     if (global_id == j + tile_id * group_size && sym.x() == 0 && sym.y() == 0 && sym.z() == 0)
                                         continue; /* We eliminate the case where the two particles are the same */

                                     /* Getting the other particle 'j' and it's perturbation */
                                     const coordinate<T> delta{sym.x() * config.L_, sym.y() * config.L_, sym.z() * config.L_};

                                     const auto other_particule = delta + particules_tile[j];
                                     //const T squared_distance = sycl::dot(this_work_item_particule, other_particule);//
                                     const T squared_distance = compute_squared_distance(this_work_item_particule, other_particule);
                                     /* If kernel uses radius cutoff, known at compile-time */
                                     if (config.use_cutoff && squared_distance > integral_power<2>(config.r_cut_)) continue;

                                     if constexpr (std::is_same_v<T, sycl::half>) {
                                         if (squared_distance == T{}) continue;
                                     }

                                     const T frac_pow_2 = config.r_star_ * config.r_star_ / squared_distance;
                                     const T frac_pow_6 = integral_power<3>(frac_pow_2);
                                     this_particule_energy += integral_power<2>(frac_pow_6) - 2 * frac_pow_6;
                                     const T force_prefactor = (frac_pow_6 - 1) * frac_pow_6 * frac_pow_2;
                                     this_particule_force += (this_work_item_particule - other_particule) * force_prefactor;
                                 }
                             }
                         }

                         if (!is_active_work_item) return;
                         forces[global_id] = this_particule_force * (-48) * config.epsilon_star_;
                         reducer_energy.combine(2 * config.epsilon_star_ * this_particule_energy);   //We divided because the energies would be counted twice otherwise
                         reducer_x.combine(forces[global_id].x());
                         reducer_y.combine(forces[global_id].y());
                         reducer_z.combine(forces[global_id].z());
                     });
         }).wait_and_throw();
    }
    return std::tuple(summed_forces, energy);
}

}   // namespace internal

/**
 * Dispatches the computation to the proper kernel
 * @tparam T
 * @param q
 * @param particules_device_in
 * @param forces_device_out
 * @param config
 * @param evt
 * @return
 */
template<typename T>
std::tuple<coordinate<T>, T> run_simulation_sycl_device_memory(                                            //
        sycl::queue& q,                                                                                    //
        const std::span<coordinate<T>> particules_device_in, std::span<coordinate<T>> forces_device_out,   //
        simulation_configuration<T> config,                                                                //
        sycl::event evt) {
    auto max_compute_units = q.get_device().get_info<sycl::info::device::max_compute_units>();
    auto work_group_size = std::min(particules_device_in.size() / max_compute_units, q.get_device().get_info<sycl::info::device::max_work_group_size>());

#ifdef SYCL_IMPLEMENTATION_ONEAPI
    if (q.get_device().is_cpu() || q.get_device().is_gpu()) work_group_size = std::min(512UL, work_group_size);
#endif

    auto kernel_on_multiple_size = [&]<int multiple_size>() {
        if (config.n_symetries == 1) {
            return internal::internal_simulator_on_sycl<T, multiple_size, 1>(q, work_group_size, particules_device_in, forces_device_out, config, evt);
        } else if (config.n_symetries == 27) {
            return internal::internal_simulator_on_sycl<T, multiple_size, 27>(q, work_group_size, particules_device_in, forces_device_out, config, evt);
            //        } else if (config.n_symetries == 125) {
            //            return internal_simulator_on_sycl<T, multiple_size, 125>(q, work_group_size, particules, forces, config, evt);
        } else {
            throw std::runtime_error("Unsupported");
        }
    };

    if (particules_device_in.size() % work_group_size == 0) {
        return kernel_on_multiple_size.template operator()<true>();
    } else {
        return kernel_on_multiple_size.template operator()<false>();
    }
}
#ifdef BUILD_HALF
template std::tuple<coordinate<sycl::half>, sycl::half> run_simulation_sycl_device_memory<sycl::half>(   //
        sycl::queue& q,                                                                                  //
        const std::span<coordinate<sycl::half>> particules, std::span<coordinate<sycl::half>> forces,    //
        simulation_configuration<sycl::half> config, sycl::event evt);
#endif
#ifdef BUILD_FLOAT
template std::tuple<coordinate<float>, float> run_simulation_sycl_device_memory<float>(                            //
        sycl::queue& q,                                                                                            //
        const std::span<coordinate<float>> particules_device_in, std::span<coordinate<float>> forces_device_out,   //
        simulation_configuration<float> config, sycl::event evt);
#endif

#ifdef BUILD_DOUBLE
template std::tuple<coordinate<double>, double> run_simulation_sycl_device_memory<double>(                           //
        sycl::queue& q,                                                                                              //
        const std::span<coordinate<double>> particules_device_in, std::span<coordinate<double>> forces_device_out,   //
        simulation_configuration<double> config, sycl::event evt);
#endif

/**
 * Launches computation on host memory
 * @tparam T
 * @param q
 * @param config
 * @param particules_host
 * @return
 */
template<typename T>
std::tuple<std::vector<coordinate<T>>, coordinate<T>, T> run_simulation_sycl(   //
        sycl::queue& q,                                                         //
        simulation_configuration<T> config, const std::vector<coordinate<T>>& particules_host) {
    auto particules_device = std::span(sycl::malloc_device<coordinate<T>>(particules_host.size(), q), particules_host.size());
    auto copy_evt = q.copy(particules_host.data(), particules_device.data(), particules_device.size());
    auto forces_device = std::span(sycl::malloc_device<coordinate<T>>(particules_host.size(), q), particules_host.size());
    auto [summed_forces, energy] = run_simulation_sycl_device_memory(q, particules_device, forces_device, config, copy_evt);
    auto forces_out = std::vector<coordinate<T>>(particules_host.size());
    q.copy(forces_device.data(), forces_out.data(), forces_device.size()).wait();
    sycl::free(particules_device.data(), q);
    sycl::free(forces_device.data(), q);
    return std::tuple(forces_out, summed_forces, energy);
}

#ifdef BUILD_HALF
template std::tuple<std::vector<coordinate<sycl::half>>, coordinate<sycl::half>, sycl::half>   //
run_simulation_sycl<sycl::half>(sycl::queue& q, simulation_configuration<sycl::half> config, const std::vector<coordinate<sycl::half>>& particules_host);
#endif

#ifdef BUILD_FLOAT
template std::tuple<std::vector<coordinate<float>>, coordinate<float>, float>   //
run_simulation_sycl<float>(sycl::queue& q, simulation_configuration<float> config, const std::vector<coordinate<float>>& particules_host);
#endif

#ifdef BUILD_DOUBLE
template std::tuple<std::vector<coordinate<double>>, coordinate<double>, double>   //
run_simulation_sycl<double>(sycl::queue& q, simulation_configuration<double> config, const std::vector<coordinate<double>>& particules_host);
#endif