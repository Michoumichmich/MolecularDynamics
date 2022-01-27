#pragma once
#include <backend/backend_interface.h>
#include <internal/sycl_usm_smart_ptr.hpp>
#include <utility>

namespace sim {

static inline auto compute_range_size(size_t size, size_t work_group_size) {
    return sycl::nd_range<1>(work_group_size * ((size + work_group_size - 1) / work_group_size), work_group_size);
}


template<typename T> class sycl_backend : backend_interface<T> {

public:
    explicit sycl_backend(sycl::queue queue, size_t size) : q(std::move(queue)), size_(size), coordinates_(size, q), momentums_(size, q), forces_(size, q), tmp_buf_(size) {}

    inline void init_backend(const std::vector<coordinate<T>>& particules) override { q.copy(particules.data(), coordinates_.get(), size_).wait(); }

    void randinit_momentums(T min, T max) override;

    inline void store_particules_coordinates(pdb_writer& writer, size_t i) const override;

    [[nodiscard]] inline T get_momentums_squared_norm() const override;

    inline void apply_multiplicative_correction_to_momentums(T coeff) override;

    inline void center_kinetic_momentums() override;

    [[nodiscard]] inline coordinate<T> mean_kinetic_momentums() const override;


    [[nodiscard]] inline size_t get_particules_count() const override { return size_; }

    std::tuple<coordinate<T>, T> run_velocity_verlet(const configuration<T>& config) override;

    std::tuple<coordinate<T>, T> init_lennard_jones_field(const configuration<T>& config) override;

private:
    mutable sycl::queue q;                                //
    size_t size_;                                         //
    sycl_unique_device_ptr<coordinate<T>> coordinates_;   //
    sycl_unique_device_ptr<coordinate<T>> momentums_;     // Vi * mi
    sycl_unique_device_ptr<coordinate<T>> forces_;        // Lennard Jones Field
    mutable std::vector<coordinate<T>> tmp_buf_;          //
};


#ifdef BUILD_HALF
extern template class sycl_backend<sycl::half>;
#endif

#ifdef BUILD_FLOAT
extern template class sycl_backend<float>;
#endif

#ifdef BUILD_DOUBLE
extern template class sycl_backend<double>;
#endif
}   // namespace sim