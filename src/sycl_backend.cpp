#include <backend/sycl_backend.hpp>
#include <utility>

namespace sim {


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


template<typename T, int n_sym>
static inline auto internal_simulator_on_sycl(                                          //
        sycl::queue& q, size_t size, size_t work_group_size,                            //
        const coordinate<T>* __restrict particules, coordinate<T>* __restrict forces,   //
        const configuration<T>& config, sycl::event in_evt) {

    return q.submit([&](sycl::handler& cgh) {
        auto particules_tile_ = sycl::accessor<coordinate<T>, 1, sycl::access_mode::read_write, sycl::access::target::local>(work_group_size, cgh);
        cgh.depends_on(in_evt);
        cgh.parallel_for(compute_range_size(size, work_group_size),
                         [size = size, L = config.L_, particules = particules, particules_tile_ = particules_tile_, r_star = config.r_star_, r_cut = config.r_cut_,
                          epsilon_star = config.epsilon_star_, forces = forces, use_cutoff = config.use_cutoff](sycl::nd_item<1> item) {
                             /* Getting space coordinates */
                             const uint32_t global_id = item.get_global_linear_id();
                             const uint32_t local_id = item.get_local_linear_id();
                             const uint32_t group_count = item.get_group_range().size();
                             const uint32_t group_size = item.get_local_range().size();

                             /* Whether the current work item takes part in the computation or not. We cannot return as it needs to be present for further barriers. */
                             const bool is_active_work_item = global_id < size;

                             /* Setting up local variables */
                             const coordinate<T> this_work_item_particule = is_active_work_item ? particules[global_id] : coordinate<T>{};

                             /* Local reducers */
                             auto this_particule_energy = T{};
                             auto this_particule_force = coordinate<T>{0, 0, 0};

                             /* Loop over 'how many tiles we need'. Each tile being a sequence of particles loaded into local memory */
                             for (uint32_t tile_id = 0U; tile_id < group_count; ++tile_id) {
                                 const uint32_t global_particule_idx = tile_id * group_size + local_id;
                                 const bool is_active_tile = global_particule_idx < size;
                                 const uint32_t this_tile_size = std::min<uint32_t>(group_size, size - tile_id * group_size);

                                 /* Loading data (as tiles) into local_memory */
                                 const coordinate<T> new_particule = is_active_tile ? particules[global_particule_idx] : coordinate<T>{};
                                 sycl::group_barrier(item.get_group());
                                 particules_tile_[local_id] = new_particule;
                                 sycl::group_barrier(item.get_group());

                                 if (!is_active_work_item) continue; /* Current not considered as we're ouf of range */
                                 prefetch_constant(particules + global_particule_idx + group_size);

                                 /* Doing the computation between our own particule and the ones from the tile */
                                 for (uint32_t j = 0U; j < this_tile_size; ++j) {
#pragma unroll
                                     for (const auto& sym: get_symetries<n_sym>()) {
                                         if (global_id == j + tile_id * group_size && sym.x() == 0 && sym.y() == 0 && sym.z() == 0) {
                                             continue; /* We eliminate the case where the two particles are the same */
                                         }

                                         /* Getting the other particle 'j' and it's perturbation */
                                         const coordinate<T> delta{sym.x() * L, sym.y() * L, sym.z() * L};

                                         const coordinate<T> other_particule = delta + particules_tile_[j];
                                         const T squared_distance = compute_squared_distance(this_work_item_particule, other_particule);
                                         /* If kernel uses radius cutoff, known at compile-time */
                                         if (use_cutoff && squared_distance > integral_power<2>(r_cut)) continue;

                                         //if constexpr (std::is_same_v<T, sycl::half>) {if (squared_distance == T{}) continue;}

                                         const T frac_pow_2 = r_star * r_star / squared_distance;
                                         const T frac_pow_6 = integral_power<3>(frac_pow_2);
                                         this_particule_energy += integral_power<2>(frac_pow_6) - 2 * frac_pow_6;
                                         const T force_prefactor = (frac_pow_6 - 1) * frac_pow_6 * frac_pow_2;
                                         this_particule_force += (this_work_item_particule - other_particule) * force_prefactor;
                                     }
                                 }
                             }

                             if (!is_active_work_item) return;
                             forces[global_id] = this_particule_force * (-48) * epsilon_star;
                         });
    });
}


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
static inline auto run_simulation_sycl_device_memory(                                   //
        sycl::queue& q,                                                                 //
        size_t size, size_t max_work_group_size_,                                       //
        const coordinate<T>* __restrict particules, coordinate<T>* __restrict forces,   //
        const configuration<T>& config, sycl::event in_evt) {

    //auto subgroup_sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
    //for (auto size: subgroup_sizes) { std::cout << size << std::endl; }
#ifdef SYCL_IMPLEMENTATION_ONEAPI
    if (q.get_device().is_cpu()) {
        max_work_group_size_ = std::min(32UL, max_work_group_size_);
    } else if (q.get_device().is_gpu()) {
        max_work_group_size_ = std::min(512UL, max_work_group_size_);
    }
#endif

    if (config.n_symetries == 1) {
        return internal_simulator_on_sycl<T, 1>(q, size, max_work_group_size_, particules, forces, config, in_evt);
    } else if (config.n_symetries == 27) {
        return internal_simulator_on_sycl<T, 27>(q, size, max_work_group_size_, particules, forces, config, in_evt);
        //        } else if (config.n_symetries == 125) {
        //            return internal_simulator_on_sycl<T, multiple_size, 125>(q, work_group_size, particules, forces, config, evt);
    } else {
        throw std::runtime_error("Unsupported");
    }
}


template<typename T> std::tuple<coordinate<T>, T> sycl_backend<T>::init_lennard_jones_field(const configuration<T>& config) {
    run_simulation_sycl_device_memory(q, size_, max_work_group_size_, coordinates_.get(), forces_.get(), config, sycl::event{}).wait();
    return {};
}


template<typename T> T sycl_backend<T>::get_momentums_squared_norm() const {
    T sum{};
    {
        auto sum_buffer = sycl::buffer<T>(&sum, 1U);
        q.submit([&](sycl::handler& cgh) {
#ifdef SYCL_IMPLEMENTATION_HIPSYCL
             auto reduction_sum = sycl::reduction(sum_buffer.get_access(cgh), sycl::plus<T>{});
#else
             auto reduction_sum = sycl::reduction(sum_buffer, cgh, sycl::plus<>{});
#endif
             cgh.parallel_for(compute_range_size(size_, max_work_group_size_), reduction_sum, [=, momentums = momentums_.get(), size_ = size_](sycl::nd_item<1> it, auto& red) {
                 auto i = it.get_global_linear_id();
                 if (i < size_) { red.combine(sycl::dot(momentums[i], momentums[i])); }
             });
         }).wait_and_throw();
    }
    return sum;
}

template<typename T> coordinate<T> sycl_backend<T>::mean_kinetic_momentums() const {
    coordinate<T> mean{};   // Sum of vi * mi;
    {
        auto x_reduction_buffer = sycl::buffer<T>(&mean.x(), 1U);
        auto y_reduction_buffer = sycl::buffer<T>(&mean.y(), 1U);
        auto z_reduction_buffer = sycl::buffer<T>(&mean.z(), 1U);
        q.submit([&](sycl::handler& cgh) {
#ifdef SYCL_IMPLEMENTATION_HIPSYCL
             auto reduction_x = sycl::reduction(x_reduction_buffer.get_access(cgh), sycl::plus<T>{});
             auto reduction_y = sycl::reduction(y_reduction_buffer.get_access(cgh), sycl::plus<T>{});
             auto reduction_z = sycl::reduction(z_reduction_buffer.get_access(cgh), sycl::plus<T>{});
#else
             auto reduction_x = sycl::reduction(x_reduction_buffer, cgh, sycl::plus<>{});
             auto reduction_y = sycl::reduction(y_reduction_buffer, cgh, sycl::plus<>{});
             auto reduction_z = sycl::reduction(z_reduction_buffer, cgh, sycl::plus<>{});
#endif
             cgh.parallel_for(compute_range_size(size_, max_work_group_size_), reduction_x, reduction_y, reduction_z,   //
                              [=, size_ = size_, momentums = momentums_.get()](sycl::nd_item<1> it, auto& x, auto& y, auto& z) {
                                  auto i = it.get_global_linear_id();
                                  if (i < size_) {
                                      auto momentum = momentums[i];
                                      x.combine(momentum.x());
                                      y.combine(momentum.y());
                                      z.combine(momentum.z());
                                  }
                              });
         }).wait_and_throw();
    }
    return mean / momentums_.size();
}

template<typename T> void sycl_backend<T>::randinit_momentums(T min, T max) {
    std::generate(tmp_buf_.begin(), tmp_buf_.end(), [=]() {   //
        return coordinate<T>(internal::generate_random_value<T>(min, max), internal::generate_random_value<T>(min, max), internal::generate_random_value<T>(min, max));
    });
    q.copy(tmp_buf_.data(), momentums_.get(), size_).wait();
}

template<typename T> void sycl_backend<T>::center_kinetic_momentums() {
    auto mean = mean_kinetic_momentums();
    q.parallel_for(compute_range_size(size_, max_work_group_size_), [=, size_ = size_, momentums = momentums_.get()](sycl::nd_item<1> it) {
         auto i = it.get_global_linear_id();
         if (i < size_) { momentums[i] -= mean; }
     }).wait();
}

template<typename T> void sycl_backend<T>::apply_multiplicative_correction_to_momentums(T coeff) {
    q.parallel_for(compute_range_size(size_, max_work_group_size_), [=, size_ = size_, momentums = momentums_.get()](sycl::nd_item<1> it) {
         auto i = it.get_global_linear_id();
         if (i < size_) { momentums[i] *= coeff; }
     }).wait();
}

template<typename T> void sycl_backend<T>::store_particules_coordinates(pdb_writer& writer, size_t i) const {
    q.copy(coordinates_.get(), tmp_buf_.data(), size_).wait();
    writer.store_new_iter(tmp_buf_, i);
}


template<typename T> std::tuple<coordinate<T>, T> sycl_backend<T>::run_velocity_verlet(const configuration<T>& config) {
    // First step: half step update of the momentums.
    auto evt = q.parallel_for(compute_range_size(size_, max_work_group_size_), [size = size_, momentums = momentums_.get(), forces = forces_.get(),
                                                                                conversion_force = config.conversion_force, dt = config.dt](sycl::nd_item<1> it) {
        auto i = it.get_global_linear_id();
        if (i < size) { momentums[i] += conversion_force * forces[i] * dt / 2; }
    });

    // Second step: update particules positions
    auto evt2 = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(evt);
        cgh.parallel_for(compute_range_size(size_, max_work_group_size_),   //
                         [size = size_, particules = coordinates_.get(), momentums = momentums_.get(), m_i = config.m_i, dt = config.dt](sycl::nd_item<1> it) {
                             auto i = it.get_global_linear_id();
                             if (i < size) { particules[i] += dt * momentums[i] / m_i; }
                         });
    });

    auto evt3 = run_simulation_sycl_device_memory(q, size_, max_work_group_size_, coordinates_.get(), forces_.get(), config, evt2);

    // Last step: update momentums given new forces
    auto evt4 = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(evt3);
        cgh.parallel_for(compute_range_size(size_, max_work_group_size_),   //
                         [size = size_, momentums = momentums_.get(), forces = forces_.get(), conversion_force = config.conversion_force, dt = config.dt](sycl::nd_item<1> it) {
                             auto i = it.get_global_linear_id();
                             if (i < size) { momentums[i] += conversion_force * forces[i] * dt / 2; }
                         });
    });

    auto evt5 = run_simulation_sycl_device_memory(q, size_, max_work_group_size_, coordinates_.get(), forces_.get(), config, evt4);
    evt5.wait_and_throw();
    return {};
}

template<typename T>
sycl_backend<T>::sycl_backend(sycl::queue queue, size_t size) : q(std::move(queue)), size_(size), coordinates_(size, q), momentums_(size, q), forces_(size, q), tmp_buf_(size) {
    auto max_compute_units = q.get_device().get_info<sycl::info::device::max_compute_units>();
    max_work_group_size_ = std::min(size / max_compute_units, q.get_device().get_info<sycl::info::device::max_work_group_size>());
}


#ifdef BUILD_HALF
template class sycl_backend<sycl::half>;
#endif

#ifdef BUILD_FLOAT
template class sycl_backend<float>;
#endif

#ifdef BUILD_DOUBLE
template class sycl_backend<double>;
#endif


}   // namespace sim
