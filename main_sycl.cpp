#include <sim_sycl.h>

#ifdef SYCL_IMPLEMENTATION_ONEAPI
#    include <sycl/ext/intel/fpga_device_selector.hpp>   //sycl::queue{sycl::ext::intel::fpga_emulator_selector{}};
#endif

template<typename T> void run_example(sycl::queue& q, const std::vector<coordinate<T>>& coordinates, configuration<T> config) {
    std::cout << config << std::endl;
    auto [field, sum, energy] = run_simulation_sycl(q, config, coordinates);
    std::cout << "Lennard Jones Energy: " << energy << std::endl;
    std::cout << "sum x: " << sum[0] << ", y: " << sum[1] << ", z: " << sum[2] << std::endl << std::endl;   // Should be 0 0 0.
}

int main(int argc, char** argv) {
    auto q = sycl::queue{sycl::default_selector{}};
    std::cout << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    if (argc < 2) {
        std::cerr << "Usage: ./" << argv[0] << "particle_file.xyz" << std::endl;
        return 1;
    }
    auto coordinates_double = parse_particule_file(argv[1]);

#ifdef BUILD_DOUBLE
    run_example(q, coordinates_double, {.use_cutoff = false, .n_symetries = 1});
    run_example(q, coordinates_double, {.use_cutoff = false, .n_symetries = 27});
    run_example(q, coordinates_double, {.use_cutoff = true, .n_symetries = 1});
    run_example(q, coordinates_double, {.use_cutoff = true, .n_symetries = 27});
    run_example(q, coordinates_double, {.use_cutoff = true, .n_symetries = 27, .L_ = 50});
    run_example(q, coordinates_double, {.use_cutoff = false, .n_symetries = 27, .L_ = 50});
    //run_example(q, coordinates, {.use_cutoff = false, .n_symetries = 125, .L_ = 50});
#endif

#ifdef BUILD_FLOAT
    auto coordinates_float = coordinate_vector_cast<float>(coordinates_double);
    run_example(q, coordinates_float, {.use_cutoff = false, .n_symetries = 1});
    run_example(q, coordinates_float, {.use_cutoff = false, .n_symetries = 27});
    run_example(q, coordinates_float, {.use_cutoff = true, .n_symetries = 1});
    run_example(q, coordinates_float, {.use_cutoff = true, .n_symetries = 27});
    run_example(q, coordinates_float, {.use_cutoff = true, .n_symetries = 27, .L_ = 50});
    run_example(q, coordinates_float, {.use_cutoff = false, .n_symetries = 27, .L_ = 50});
    //run_example(q, coordinates_float, {.use_cutoff = false, .n_symetries = 125, .L_ = 50});
#endif

#ifdef BUILD_HALF
    auto coordinates_half = coordinate_vector_cast<sycl::half>(coordinates_double);
    run_example(q, coordinates_half, {.use_cutoff = false, .n_symetries = 1});
    run_example(q, coordinates_half, {.use_cutoff = false, .n_symetries = 27});
    run_example(q, coordinates_half, {.use_cutoff = true, .n_symetries = 1});
    run_example(q, coordinates_half, {.use_cutoff = true, .n_symetries = 27});
    run_example(q, coordinates_half, {.use_cutoff = true, .n_symetries = 27, .L_ = 50});
    run_example(q, coordinates_half, {.use_cutoff = false, .n_symetries = 27, .L_ = 50});
    //run_example(q, coordinates_float, {.use_cutoff = false, .n_symetries = 125, .L_ = 50});
#endif
}