#pragma once
#include <backend/backend_interface.h>
#include <internal/sycl_usm_smart_ptr.hpp>
#include <utility>

namespace sim {

template<typename T> class sycl_backend : backend_interface<T> {

public:
    EXPORT explicit sycl_backend(size_t size, const sycl::queue& queue = sycl::queue{}, bool maximise_occupancy = true);

    void set_speeds(const std::vector<coordinate<T>>& speeds, const configuration<T>& config) OPT_OVERRIDE;

    void set_particles(const std::vector<coordinate<T>>& particules, const configuration<T>& config) OPT_OVERRIDE;

    [[nodiscard]] inline size_t get_particles_count() const OPT_OVERRIDE { return size_; }

    void randinit_momentums(T min, T max) OPT_OVERRIDE;

    void store_particles_coordinates(pdb_writer& writer, size_t i, T temp, T epot) const OPT_OVERRIDE;

    [[nodiscard]] T get_momentums_squared_norm() const OPT_OVERRIDE;

    void apply_multiplicative_correction_to_momentums(T coeff) OPT_OVERRIDE;

    void center_kinetic_momentums() OPT_OVERRIDE;

    [[nodiscard]] coordinate<T> mean_kinetic_momentum() const OPT_OVERRIDE;

    void run_velocity_verlet(const configuration<T>& config) OPT_OVERRIDE;

    void update_lennard_jones_field(const configuration<T>& config) OPT_OVERRIDE;

    /**
     * The SYCL backend does not recompute these values systematically, so we'll have to do it. Reductions in SYCL are messy.
     * @return
     */
    [[nodiscard]] std::tuple<coordinate<T>, T> get_last_lennard_jones_metrics() const OPT_OVERRIDE;

private:
    T reduce_energies() const;
    coordinate<T> compute_error_lennard_jones() const;

    mutable sycl::queue q;                                                      //
    [[no_unique_address]] size_t size_;                                         //
    [[no_unique_address]] sycl_unique_device_ptr<coordinate<T>> coordinates_;   //
    [[no_unique_address]] sycl_unique_device_ptr<coordinate<T>> momentums_;     // Vi * mi
    [[no_unique_address]] sycl_unique_device_ptr<coordinate<T>> forces_;        // Lennard Jones Field
    [[no_unique_address]] sycl_unique_device_ptr<T> particule_energy_;          // Lennard Jones Field
    [[no_unique_address]] mutable std::vector<coordinate<T>> tmp_buf_;          //

public:
    /**
     * To avoid calling the SYCL runtime too much in a hot path
     */
    struct kernel_configs {
        size_t max_work_group_size;   //
        size_t max_reduction_size;    //
    };

private:
    [[no_unique_address]] kernel_configs configs_;
};


}   // namespace sim
