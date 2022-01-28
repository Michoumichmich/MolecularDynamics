#pragma once

#include <internal/cpp_utils.hpp>

namespace sim {
template<typename T> struct usm_deleter {

    const sycl::queue q_;

    explicit usm_deleter(sycl::queue q) : q_(std::move(q)) {}

    void operator()(T* ptr) const noexcept {
        if (ptr) { sycl::free(ptr, q_); }
    }
};


/**
     * Wrapper for a std::unique_ptr that calls the SYCL deleter (sycl::free).
     * Also holds the number of elements allocated.
     * @tparam T
     * @tparam location
     */
template<typename T, sycl::usm::alloc location> class usm_unique_ptr : public std::unique_ptr<T, usm_deleter<T>> {
private:
    const size_t count_{};

public:
    [[nodiscard]] usm_unique_ptr(size_t count, sycl::queue q) : std::unique_ptr<T, usm_deleter<T>>(sycl::malloc<T>(count, q, location), usm_deleter<T>{q}), count_(count) {}

    [[nodiscard]] explicit usm_unique_ptr(sycl::queue q) : usm_unique_ptr(1, q) {}

    [[nodiscard]] inline size_t size_bytes() const noexcept { return count_ * sizeof(T); }

    [[nodiscard]] inline size_t size() const noexcept { return count_; }

    [[nodiscard]] inline sycl::multi_ptr<T, sycl::access::address_space::global_space> get_multi() const noexcept { return {std::unique_ptr<T, usm_deleter<T>>::get()}; }
};

template<typename T> using sycl_unique_device_ptr = usm_unique_ptr<T, sycl::usm::alloc::device>;


}   // namespace sim
