#include <sim>

template<typename T> void run_example(const std::vector<sim::coordinate<T>>& coordinates, sim::configuration<T> config) {
    std::cout << config << std::endl;
    auto simulation = sim::molecular_dynamics<T, sim::cpu_backend>(coordinates, config);
    for (int i = 0; i < 100000; ++i) {
        std::cout << simulation << std::endl;
        simulation.run_iter();
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./" << argv[0] << "particle_file.xyz" << std::endl;
        return 1;
    }
    auto coordinates_double = sim::parse_particule_file(argv[1]);


#ifdef BUILD_DOUBLE
    run_example(coordinates_double, {.iter_per_frame = 1});
//    run_example(coordinates_double, {.use_cutoff = false, .n_symetries = 1});
//    run_example(coordinates_double, {.use_cutoff = true, .n_symetries = 1});
//    run_example(coordinates_double, {.use_cutoff = true, .n_symetries = 27});
//    run_example(coordinates_double, {.use_cutoff = true, .n_symetries = 27, .L_ = 50});
//    run_example(coordinates_double, {.use_cutoff = false, .n_symetries = 27, .L_ = 50});
#endif

#ifdef BUILD_FLOAT
    auto coordinates_float = sim::coordinate_vector_cast<float>(coordinates_double);
    run_example(coordinates_float, {});
//    run_example(coordinates_float, {.use_cutoff = true, .r_cut_ = 40, .n_symetries = 27, .L_ = 50});
//    run_example(coordinates_float, {.use_cutoff = false, .n_symetries = 1});
//    run_example(coordinates_float, {.use_cutoff = false, .n_symetries = 27});
//    run_example(coordinates_float, {.use_cutoff = true, .n_symetries = 1});
//    run_example(coordinates_float, {.use_cutoff = true, .n_symetries = 27});
//    run_example(coordinates_float, {.use_cutoff = true, .n_symetries = 27, .L_ = 50});
//    run_example(coordinates_float, {.use_cutoff = false, .n_symetries = 27, .L_ = 50});
#endif

#ifdef BUILD_HALF
    auto coordinates_half = sim::coordinate_vector_cast<sycl::half>(coordinates_double);
    run_example(coordinates_half, {.use_cutoff = false, .n_symetries = 1});
    run_example(coordinates_half, {.use_cutoff = false, .n_symetries = 27});
    run_example(coordinates_half, {.use_cutoff = true, .n_symetries = 1});
    run_example(coordinates_half, {.use_cutoff = true, .n_symetries = 27});
    run_example(coordinates_half, {.use_cutoff = true, .n_symetries = 27, .L_ = 50});
    run_example(coordinates_half, {.use_cutoff = false, .n_symetries = 27, .L_ = 50});
#endif
}