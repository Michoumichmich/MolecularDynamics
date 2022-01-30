#pragma once
#include <internal/cpp_utils.hpp>

template<typename KernelName> static inline size_t max_work_groups_for_kernel(sycl::queue& q) {
    size_t max_items = std::max(1U, std::min(4096U, static_cast<uint32_t>(q.get_device().get_info<sycl::info::device::max_work_group_size>())));
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


template<typename KernelName> static inline std::pair<size_t, size_t> query_kernel_group_sizes(const sycl::queue& q) {
    size_t max_items = std::max(1U, std::min(4096U, static_cast<uint32_t>(q.get_device().get_info<sycl::info::device::max_work_group_size>())));
    auto preferred = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
    auto preferred_multiple = std::min(32UL, *std::max_element(preferred.begin(), preferred.end()));
    std::cout << "Preferred multiple " << std::endl;
#if defined(SYCL_IMPLEMENTATION_INTEL) || defined(SYCL_IMPLEMENTATION_ONEAPI)
    try {
        sycl::kernel_id id = sycl::get_kernel_id<KernelName>();
        auto kernel = sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context()).get_kernel(id);
        //size_t register_count = kernel.get_info<sycl::info::kernel_device_specific::ext_codeplay_num_regs>(q.get_device());
        max_items = std::min(max_items, kernel.get_info<sycl::info::kernel_device_specific::work_group_size>(q.get_device()));
        preferred_multiple = std::min(preferred_multiple, kernel.get_info<sycl::info::kernel_device_specific::preferred_work_group_size_multiple>(q.get_device()));
    } catch (std::exception& e) {
        std::cout << "Couldn't read kernel properties for device: " << q.get_device().get_info<sycl::info::device::name>() << " got exception: " << e.what() << std::endl;
    }
#endif
    //std::cout << "Max items: " << max_items << ", Preferred_Multiple: " << preferred_multiple << std::endl;
    return {max_items, preferred_multiple};
}

template<typename kernel> static inline size_t restrict_work_group_size(size_t size, const sycl::queue& q) noexcept {
    const auto max_compute_units = std::max(1UL, std::min<size_t>(size, q.get_device().template get_info<sycl::info::device::max_compute_units>()));
    auto [max_group_size, preferred_multiple] = query_kernel_group_sizes<kernel>(q);
    auto rqd_work_per_cu = (size + max_compute_units - 1) / max_compute_units;
    while (rqd_work_per_cu > max_group_size) { rqd_work_per_cu /= 2; }
    auto per_work_item = preferred_multiple * ((rqd_work_per_cu + preferred_multiple + 1) / preferred_multiple);
    //std::cout << "per_work_item: " << per_work_item << std::endl;
    return per_work_item;
}

static inline size_t restrict_work_group_size(size_t size, const sycl::queue& q) noexcept {
    const auto max_compute_units = std::max(1UL, std::min<size_t>(size, q.get_device().template get_info<sycl::info::device::max_compute_units>()));
    const auto max_work_group_size = std::max(1UL, std::min<size_t>(size, q.get_device().template get_info<sycl::info::device::max_work_group_size>()));
    auto rqd_work_per_cu = (size + max_compute_units - 1) / max_compute_units;
    while (rqd_work_per_cu > max_work_group_size) { rqd_work_per_cu /= 2; }
    //std::cout << "rqd_work_per_cu: " << rqd_work_per_cu << std::endl;
    return rqd_work_per_cu;
}