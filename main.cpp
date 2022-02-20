/**
 * @author Michel Migdal
 * Sample file for the MD simulation (missng a real front-end sorry).
 */

#include <sim>


using namespace std::string_literals;

template<typename KernelName> static inline size_t max_work_groups_for_kernel(sycl::queue q) {
    size_t max_items = std::max(1U, std::min(4096U, static_cast<uint32_t>(q.get_device().get_info<sycl::info::device::max_work_group_size>())));
#if defined(SYCL_IMPLEMENTATION_INTEL) || defined(SYCL_IMPLEMENTATION_ONEAPI)
    try {
        sycl::kernel_id id = sycl::get_kernel_id<KernelName>();
        auto kernel = sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context()).get_kernel(id);
        max_items = std::min(max_items, kernel.get_info<sycl::info::kernel_device_specific::work_group_size>(q.get_device()));
    } catch (std::exception& e) {
        std::cout << "Couldn't read kernel properties for device: " << q.get_device().get_info<sycl::info::device::name>() << " got exception: " << e.what() << std::endl;
    }
#endif
    return max_items;
}

class my_kernel;

int main() {
    sycl::queue{}.parallel_for<my_kernel>(sycl::range(100), [=](sycl::item<1> it) { (void) it.get_id(); });
    std::cout << max_work_groups_for_kernel<my_kernel>(sycl::queue{}) << std::endl;
}
