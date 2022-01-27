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

template<typename T, bool multiple_size, int n_sym> class leenard_jones_kernel {
private:
    const sim::configuration<T> config_;
    const std::span<coordinate<T>> particules_;
    std::span<coordinate<T>> forces_field_;
    sycl::accessor<coordinate<T>, 1, sycl::access_mode::read_write, sycl::access::target::local> particules_tile_;

public:
    leenard_jones_kernel(                                 //
            const configuration<T>& config,               //
            const std::span<coordinate<T>>& particules,   //
            const std::span<coordinate<T>>& forces,       //
            const sycl::accessor<coordinate<T>, 1, sycl::access_mode::read_write, sycl::access::target::local>& acc)
        : config_(config), particules_(particules), forces_field_(forces), particules_tile_(acc) {}


    inline void operator()(const sycl::nd_item<1>& item, auto& reducer_x, auto& reducer_y, auto& reducer_z, auto& reducer_energy) const noexcept {
        /* Getting space coordinates */
        const uint32_t global_id = item.get_global_linear_id();
        const uint32_t local_id = item.get_local_linear_id();
        const uint32_t group_count = item.get_group_range().size();
        const uint32_t group_size = item.get_local_range().size();

        /* Whether the current work item takes part in the computation or not. We cannot return as it needs to be present for further barriers. */
        const bool is_active_work_item = [&]() {
            if constexpr (multiple_size) return true;
            else {
                return global_id < particules_.size();
            }
        }();

        /* Setting up local variables */
        const coordinate<T> this_work_item_particule = is_active_work_item ? particules_[global_id] : coordinate<T>{};

        /* Local reducers */
        auto this_particule_energy = T{};
        auto this_particule_force = coordinate<T>{0, 0, 0};

        /* Loop over 'how many tiles we need'. Each tile being a sequence of particles loaded into local memory */
        for (uint32_t tile_id = 0U; tile_id < group_count; ++tile_id) {
            const uint32_t global_particule_idx = tile_id * group_size + local_id;
            const bool is_active_tile = [&]() {
                if constexpr (multiple_size) return true;
                else {
                    return global_particule_idx < particules_.size();
                }
            }();
            const uint32_t this_tile_size = [&]() {
                if constexpr (multiple_size) {
                    return group_size;
                } else {
                    return std::min<uint32_t>(group_size, particules_.size() - tile_id * group_size);
                }
            }();

            /* Loading data (as tiles) into local_memory */
            const coordinate<T> new_particule = is_active_tile ? particules_[global_particule_idx] : coordinate<T>{};
            sycl::group_barrier(item.get_group());
            particules_tile_[local_id] = new_particule;
            sycl::group_barrier(item.get_group());

            if (!is_active_work_item) continue; /* Current not considered as we're ouf of range */
            prefetch_constant(particules_.data() + global_particule_idx + group_size);

            /* Doing the computation between our own particule and the ones from the tile */
            for (uint32_t j = 0U; j < this_tile_size; ++j) {
#if defined(__NVPTX__) && defined(__SYCL_DEVICE_ONLY__)
#    pragma unroll n_sym
#endif
                for (const auto& sym: get_symetries<n_sym>()) {
                    if (global_id == j + tile_id * group_size && sym.x() == 0 && sym.y() == 0 && sym.z() == 0) {
                        continue; /* We eliminate the case where the two particles are the same */
                    }

                    /* Getting the other particle 'j' and it's perturbation */
                    const coordinate<T> delta{sym.x() * config_.L_, sym.y() * config_.L_, sym.z() * config_.L_};

                    const coordinate<T> other_particule = delta + particules_tile_[j];
                    const T squared_distance = compute_squared_distance(this_work_item_particule, other_particule);
                    /* If kernel uses radius cutoff, known at compile-time */
                    if (config_.use_cutoff && squared_distance > integral_power<2>(config_.r_cut_)) continue;

                    if constexpr (std::is_same_v<T, sycl::half>) {
                        if (squared_distance == T{}) continue;
                    }

                    const T frac_pow_2 = config_.r_star_ * config_.r_star_ / squared_distance;
                    const T frac_pow_6 = integral_power<3>(frac_pow_2);
                    this_particule_energy += integral_power<2>(frac_pow_6) - 2 * frac_pow_6;
                    const T force_prefactor = (frac_pow_6 - 1) * frac_pow_6 * frac_pow_2;
                    this_particule_force += (this_work_item_particule - other_particule) * force_prefactor;
                }
            }
        }

        if (!is_active_work_item) return;
        forces_field_[global_id] = this_particule_force * (-48) * config_.epsilon_star_;
        reducer_energy.combine(2 * config_.epsilon_star_ * this_particule_energy);   //We divided because the energies would be counted twice otherwise
        reducer_x.combine(forces_field_[global_id].x());
        reducer_y.combine(forces_field_[global_id].y());
        reducer_z.combine(forces_field_[global_id].z());
    }
};

template<typename T, bool multiple_size, int n_sym>
static inline auto internal_simulator_on_sycl(                                        //
        sycl::queue& q, size_t work_group_size,                                       //
        const std::span<coordinate<T>> particules, std::span<coordinate<T>> forces,   //
        configuration<T> config, sycl::event evt = {}) {
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
             auto kernel = leenard_jones_kernel<T, multiple_size, n_sym>(config, particules, forces, particules_tile);
             cgh.parallel_for(compute_range_size(particules.size(), work_group_size), reduction_x, reduction_y, reduction_z, reduction_energy, kernel);
         }).wait_and_throw();
    }
    return std::tuple(summed_forces, energy);
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
std::tuple<coordinate<T>, T> run_simulation_sycl_device_memory(                                            //
        sycl::queue& q,                                                                                    //
        const std::span<coordinate<T>> particules_device_in, std::span<coordinate<T>> forces_device_out,   //
        configuration<T> config,                                                                           //
        sycl::event evt) {
    auto max_compute_units = q.get_device().get_info<sycl::info::device::max_compute_units>();
    auto work_group_size = std::min(particules_device_in.size() / max_compute_units, q.get_device().get_info<sycl::info::device::max_work_group_size>());
    //auto subgroup_sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
    //for (auto size: subgroup_sizes) { std::cout << size << std::endl; }
#ifdef SYCL_IMPLEMENTATION_ONEAPI
    if (q.get_device().is_cpu()) {
        work_group_size = std::min(32UL, work_group_size);
    } else if (q.get_device().is_gpu()) {
        work_group_size = std::min(512UL, work_group_size);
    }
#endif

    auto kernel_on_multiple_size = [&]<int multiple_size>() {
        if (config.n_symetries == 1) {
            return internal_simulator_on_sycl<T, multiple_size, 1>(q, work_group_size, particules_device_in, forces_device_out, config, evt);
        } else if (config.n_symetries == 27) {
            return internal_simulator_on_sycl<T, multiple_size, 27>(q, work_group_size, particules_device_in, forces_device_out, config, evt);
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
             cgh.parallel_for(compute_range_size(size_, 32), reduction_sum, [=, momentums = momentums_.get(), size_ = size_](sycl::nd_item<1> it, auto& red) {
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
             cgh.parallel_for(compute_range_size(size_, 32), reduction_x, reduction_y, reduction_z,   //
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
    q.parallel_for(compute_range_size(size_, 32), [=, size_ = size_, momentums = momentums_.get()](sycl::nd_item<1> it) {
         auto i = it.get_global_linear_id();
         if (i < size_) { momentums[i] -= mean; }
     }).wait();
}

template<typename T> void sycl_backend<T>::apply_multiplicative_correction_to_momentums(T coeff) {
    q.parallel_for(compute_range_size(size_, 32), [=, size_ = size_, momentums = momentums_.get()](sycl::nd_item<1> it) {
         auto i = it.get_global_linear_id();
         if (i < size_) { momentums[i] *= coeff; }
     }).wait();
}

template<typename T> void sycl_backend<T>::store_particules_coordinates(pdb_writer& writer, size_t i) const {
    q.copy(coordinates_.get(), tmp_buf_.data(), size_).wait();
    writer.store_new_iter(tmp_buf_, i);
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