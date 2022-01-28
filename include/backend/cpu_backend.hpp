#pragma once

#include <backend/backend_interface.h>

namespace sim {
template<typename T> class cpu_backend : backend_interface<T> {

public:
    inline void init_backend(const std::vector<coordinate<T>>& particules) override {
        size_ = particules.size();
        coordinates_ = particules;
        momentums_ = std::vector<coordinate<T>>(size_);
        forces_ = std::vector<coordinate<T>>(size_);
    }


    void randinit_momentums(T min, T max) override;

    inline void store_particules_coordinates(pdb_writer& writer, size_t i) const override { writer.store_new_iter(coordinates_, i); };

    [[nodiscard]] inline T get_momentums_squared_norm() const override {
        T sum{};
        for (const auto& momentum: momentums_) { sum += sycl::dot(momentum, momentum); }
        return sum;
    }

    inline void apply_multiplicative_correction_to_momentums(T coeff) override {
        for (auto& momentum: momentums_) { momentum *= coeff; }
    }

    inline void center_kinetic_momentums() override {
        auto mean = mean_kinetic_momentums();
        for (auto& momentum: momentums_) { momentum -= mean; }
    }

    [[nodiscard]] inline coordinate<T> mean_kinetic_momentums() const override {
        coordinate<T> mean{};   // Sum of vi * mi;
        for (const auto& momentum: momentums_) { mean += momentum; }
        return mean / momentums_.size();
    }


    [[nodiscard]] inline size_t get_particules_count() const override { return size_; }

    std::tuple<coordinate<T>, T> run_velocity_verlet(const configuration<T>& config) override;

    std::tuple<coordinate<T>, T> init_lennard_jones_field(const configuration<T>& config) override;

private:
    size_t size_{};
    std::vector<coordinate<T>> coordinates_{};   //
    std::vector<coordinate<T>> momentums_{};     // Vi * mi
    std::vector<coordinate<T>> forces_{};        // Lennard Jones Field
};

#ifdef BUILD_HALF
extern template class cpu_backend<sycl::half>;
#endif

#ifdef BUILD_FLOAT
extern template class cpu_backend<float>;
#endif

#ifdef BUILD_DOUBLE
extern template class cpu_backend<double>;
#endif
}   // namespace sim
