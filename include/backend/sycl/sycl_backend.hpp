#pragma once
#include "sycl_backend.h"
#include <utility>

namespace sim {


template<typename KernelName> static inline size_t max_work_groups_for_kernel(sycl::queue& q) {
    size_t max_items = std::max<size_t>(1U, std::min<size_t>(4096U, static_cast<uint32_t>(q.get_device().get_info<sycl::info::device::max_work_group_size>())));
#if defined(SYCL_IMPLEMENTATION_INTEL) || defined(SYCL_IMPLEMENTATION_ONEAPI)
    try {
        sycl::kernel_id id = sycl::get_kernel_id<KernelName>();
        auto kernel = sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context()).get_kernel(id);
        //size_t register_count = kernel.get_info<sycl::info::kernel_device_specific::ext_codeplay_num_regs>(q.get_device());
        max_items = std::min(max_items, kernel.get_info<sycl::info::kernel_device_specific::work_group_size>(q.get_device()));
    } catch (std::exception& e) {
        std::cout << "Couldn't read kernel properties for device: " << q.get_device().get_info<sycl::info::device::name>() << " got exception: " << e.what() << std::endl;
    }
#endif
    return max_items;
}


template<typename T> static inline void prefetch_constant(const T* ptr) {
#if defined(__NVPTX__) && defined(__SYCL_DEVICE_ONLY__)
    if constexpr (sizeof(T*) == 8) {
        asm("prefetchu.L1 [%0];" : : "l"(ptr));
    } else {
        asm("prefetchu.L1 [%0];" : : "r"(ptr));
    }
#else
    (void) ptr;
#endif
}

static inline auto compute_range_size(size_t size, size_t work_group_size) {
    return sycl::nd_range<1>(work_group_size * ((size + work_group_size - 1) / work_group_size), work_group_size);
}

#include "sycl_backend_impl_reductions.hpp"


template<typename T, int n_sym> struct lennard_jones_kernel;

template<typename T, int n_sym>
static inline auto update_lennard_jones_field_impl_sycl(                                                         //
        sycl::queue& q, size_t size, size_t work_group_size,                                                     //
        const coordinate<T>* __restrict coordinates, coordinate<T>* __restrict forces, T* __restrict energies,   //
        const configuration<T>& config, sycl::event in_evt) {

    return q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(std::move(in_evt));
        cgh.parallel_for<lennard_jones_kernel<T, n_sym>>(   //
                compute_range_size(size, work_group_size), [size = size, L = config.L, coordinates = coordinates, r_star = config.r_star, r_cut = config.r_cut, forces = forces,
                                                            energies = energies, use_cutoff = config.use_cutoff, epsilon_star = config.epsilon_star](sycl::nd_item<1> item) {
                    const auto i = item.get_global_linear_id();
                    if (i >= size) return;
                    auto this_particule_energy = T{};
                    auto this_particule_force = coordinate<T>{};
                    const auto this_particule = coordinates[i];
                    for (auto j = 0U; j < size; ++j) {

#pragma unroll
                        for (const auto& sym: get_symetries<n_sym>()) {
                            if (i == j && sym.x() == 0 && sym.y() == 0 && sym.z() == 0) continue;
                            const coordinate<T> delta{sym.x() * L, sym.y() * L, sym.z() * L};
                            const auto other_particule = delta + coordinates[j];
                            T squared_distance = compute_squared_distance(this_particule, other_particule);
                            if (use_cutoff && squared_distance > integral_power<2>(r_cut)) continue;

                            internal::assume(squared_distance != T{});
                            //if (squared_distance == T{}) { throw std::runtime_error("Got null distance"); }

                            const T frac_pow_2 = r_star * r_star / squared_distance;
                            const T frac_pow_6 = integral_power<3>(frac_pow_2);
                            this_particule_energy += integral_power<2>(frac_pow_6) - 2 * frac_pow_6;
                            const T force_prefactor = (frac_pow_6 - 1.) * frac_pow_6 * frac_pow_2;
                            this_particule_force += (this_particule - other_particule) * force_prefactor;
                        }
                    }
                    forces[i] = this_particule_force;
                    energies[i] = 2 * epsilon_star * this_particule_energy;
                });
    });
}


template<typename T>
static inline auto update_lennard_jones_field_dispatch_impl(                                                     //
        sycl::queue& q,                                                                                          //
        size_t size, const typename sycl_backend<T>::kernel_configs& configs,                                    //
        const coordinate<T>* __restrict coordinates, coordinate<T>* __restrict forces, T* __restrict energies,   //
        const configuration<T>& config, sycl::event in_evt) {

    if (config.n_symetries == 1) {
        return update_lennard_jones_field_impl_sycl<T, 1>(q, size, configs.max_work_groups_lennard_1, coordinates, forces, energies, config, in_evt);
    } else if (config.n_symetries == 27) {
        return update_lennard_jones_field_impl_sycl<T, 27>(q, size, configs.max_work_groups_lennard_27, coordinates, forces, energies, config, in_evt);
    } else if (config.n_symetries == 125) {
        return update_lennard_jones_field_impl_sycl<T, 125>(q, size, configs.max_work_groups_lennard_125, coordinates, forces, energies, config, in_evt);
    } else {
        throw std::runtime_error("Unsupported");
    }
}


template<typename T> void sycl_backend<T>::init_lennard_jones_field(const configuration<T>& config) {
    update_lennard_jones_field_dispatch_impl(q, size_, configs_, coordinates_.get(), forces_.get(), particule_energy_.get(), config, sycl::event{}).wait();
}


template<typename T> void sycl_backend<T>::randinit_momentums(T min, T max) {
    std::generate(tmp_buf_.begin(), tmp_buf_.end(), [=]() {   //
        return coordinate<T>(internal::generate_random_value<T>(min, max), internal::generate_random_value<T>(min, max), internal::generate_random_value<T>(min, max));
    });
    q.copy(tmp_buf_.data(), momentums_.get(), size_).wait();
}

template<typename T> void sycl_backend<T>::center_kinetic_momentums() {
    auto mean = mean_kinetic_momentums();
    q.parallel_for(compute_range_size(size_, configs_.max_work_group_size), [mean = mean, size_ = size_, momentums = momentums_.get()](sycl::nd_item<1> it) {
         const auto i = it.get_global_linear_id();
         if (i >= size_) return;
         momentums[i] -= mean;
     }).wait();
}

template<typename T> void sycl_backend<T>::apply_multiplicative_correction_to_momentums(T coeff) {
    q.parallel_for(compute_range_size(size_, configs_.max_work_group_size), [coeff = coeff, size_ = size_, momentums = momentums_.get()](sycl::nd_item<1> it) {
         const auto i = it.get_global_linear_id();
         if (i >= size_) return;
         momentums[i] *= coeff;
     }).wait();
}

template<typename T> void sycl_backend<T>::store_particules_coordinates(pdb_writer& writer, size_t i) const {
    q.copy(coordinates_.get(), tmp_buf_.data(), size_).wait();
    writer.store_new_iter(tmp_buf_, i);
}


template<typename T> void sycl_backend<T>::run_velocity_verlet(const configuration<T>& config) {
    // First step: half step update of the momentums.
    auto evt = q.parallel_for(   //
            compute_range_size(size_, configs_.max_work_group_size),
            [size = size_, momentums = momentums_.get(), forces = forces_.get(), conversion_force = config.conversion_force, dt = config.dt](sycl::nd_item<1> it) {
                const auto i = it.get_global_linear_id();
                if (i >= size) return;
                momentums[i] += conversion_force * forces[i] * dt / 2;
            });

    // Second step: update particules positions
    auto evt2 = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(evt);
        cgh.parallel_for(compute_range_size(size_, configs_.max_work_group_size),
                         [size = size_, coordinates = coordinates_.get(), momentums = momentums_.get(), m_i = config.m_i, dt = config.dt](sycl::nd_item<1> it) {
                             const auto i = it.get_global_linear_id();
                             if (i >= size) return;
                             coordinates[i] += dt * momentums[i] / m_i;
                         });
    });

    auto evt3 = update_lennard_jones_field_dispatch_impl(q, size_, configs_, coordinates_.get(), forces_.get(), particule_energy_.get(), config, evt2);

    // Last step: update momentums given new forces
    auto evt4 = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(evt3);
        cgh.parallel_for(compute_range_size(size_, configs_.max_work_group_size),
                         [size = size_, momentums = momentums_.get(), forces = forces_.get(), conversion_force = config.conversion_force, dt = config.dt](sycl::nd_item<1> it) {
                             const auto i = it.get_global_linear_id();
                             if (i >= size) return;
                             momentums[i] += conversion_force * forces[i] * dt / 2;
                         });
    });

    auto evt5 = update_lennard_jones_field_dispatch_impl(q, size_, configs_, coordinates_.get(), forces_.get(), particule_energy_.get(), config, evt4);
    evt5.wait_and_throw();
}

template<typename T>
sycl_backend<T>::sycl_backend(size_t size, const sycl::queue& queue, bool maximise_occupancy)
    : q(queue), size_(size), coordinates_(size, q), momentums_(size, q), forces_(size, q), particule_energy_(size, q), tmp_buf_(size) {

    const auto max_compute_units = std::max(1UL, std::min<size_t>(size, q.get_device().template get_info<sycl::info::device::max_compute_units>()));
    const auto max_work_group_size = std::max(1UL, std::min<size_t>(size, q.get_device().template get_info<sycl::info::device::max_work_group_size>()));

    if (maximise_occupancy && size_ < max_compute_units * max_work_group_size) {
        configs_.max_work_group_size = std::min(max_work_group_size, std::max(1UL, (size + max_compute_units - 1) / max_compute_units));
    } else {
        configs_.max_work_group_size = max_work_group_size;
    }

    configs_.max_work_groups_lennard_1 = std::min(configs_.max_work_group_size, max_work_groups_for_kernel<lennard_jones_kernel<T, 1>>(q));
    configs_.max_work_groups_lennard_27 = std::min(configs_.max_work_group_size, max_work_groups_for_kernel<lennard_jones_kernel<T, 27>>(q));
    configs_.max_work_groups_lennard_125 = std::min(configs_.max_work_group_size, max_work_groups_for_kernel<lennard_jones_kernel<T, 125>>(q));

    configs_.max_reduction_size = configs_.max_work_group_size;
#ifdef SYCL_IMPLEMENTATION_ONEAPI
    if (q.get_device().is_cpu()) {
        configs_.max_reduction_size = std::min(64UL, configs_.max_reduction_size);
    } else if (q.get_device().is_gpu()) {
        configs_.max_reduction_size = std::min(512UL, configs_.max_reduction_size);
    }
#endif
}

template<typename T> void sycl_backend<T>::init_backend(const std::vector<coordinate<T>>& particules) { q.copy(particules.data(), coordinates_.get(), size_).wait(); }

template<typename T> std::tuple<coordinate<T>, T> sycl_backend<T>::get_last_lennard_jones_metrics() const { return {compute_error_lennard_jones(), reduce_energies()}; }

}   // namespace sim
