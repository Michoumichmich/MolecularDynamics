#include <helpers.hpp>
#include <sim_seq.h>

template<typename T> void run_example(const std::vector<coordinate<T>>& coordinates, simulation_configuration<T> config) {
    std::cout << config << std::endl;
    auto [field, sum, energy] = run_simulation_sequential(coordinates, config);
    std::cout << "Lennard Jones Energy: " << energy << std::endl;
    std::cout << "sum x: " << sum[0] << ", y: " << sum[1] << ", z: " << sum[2] << std::endl << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    auto coordinates = parse_particule_file(argv[1]);

    run_example(coordinates, {.use_cutoff = false, .n_symetries = 1});
    run_example(coordinates, {.use_cutoff = false, .n_symetries = 27});
    run_example(coordinates, {.use_cutoff = true, .n_symetries = 1});
    run_example(coordinates, {.use_cutoff = true, .n_symetries = 27});
    run_example(coordinates, {.use_cutoff = true, .n_symetries = 27, .L_ = 50});
    run_example(coordinates, {.use_cutoff = false, .n_symetries = 27, .L_ = 50});
    //run_example(coordinates, {.use_cutoff = false, .n_symetries = 125, .L_ = 50});

#ifdef BUILD_FLOAT
    auto coordinates_float = coordinate_vector_cast<float>(coordinates);
    run_example(coordinates_float, {.use_cutoff = false, .n_symetries = 1});
    run_example(coordinates_float, {.use_cutoff = false, .n_symetries = 27});
    run_example(coordinates_float, {.use_cutoff = true, .n_symetries = 1});
    run_example(coordinates_float, {.use_cutoff = true, .n_symetries = 27});
    run_example(coordinates_float, {.use_cutoff = true, .n_symetries = 27, .L_ = 50});
    run_example(coordinates_float, {.use_cutoff = false, .n_symetries = 27, .L_ = 50});
    //run_example(coordinates_float, {.use_cutoff = false, .n_symetries = 125, .L_ = 50});
#endif
}