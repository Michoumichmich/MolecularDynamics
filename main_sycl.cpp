#include <sim_sycl.h>
#include <helpers.hpp>

template<typename T>
void run_example(sycl::queue &q, const std::vector<coordinate<T>> &coordinates, simulation_configuration<T> config) {
    std::cout << config << std::endl;
    auto[field, sum, energy] = run_simulation_sycl(q, config, coordinates);
    std::cout << "Lennard Jones Energy: " << energy << std::endl;
    std::cout << "sum x: " << sum[0] << ", y: " << sum[1] << ", z: " << sum[2] << std::endl << std::endl;
}

int main(int argc, char **argv) {
    auto q = sycl::queue{sycl::default_selector{}}; //sycl::queue{sycl::ext::intel::fpga_emulator_selector{}};
    std::cout << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    if (argc < 2) return 1;
    auto coordinates = parse_particule_file(argv[1]);

    run_example(q, coordinates, {.use_cutoff = false, .n_symetries = 1});
    run_example(q, coordinates, {.use_cutoff = false, .n_symetries = 27});
    run_example(q, coordinates, {.use_cutoff = true, .n_symetries = 1});
    run_example(q, coordinates, {.use_cutoff = true, .n_symetries = 27});
    run_example(q, coordinates, {.use_cutoff = true, .n_symetries = 27, .L_=50});
    run_example(q, coordinates, {.use_cutoff = false, .n_symetries = 27, .L_=50});
    run_example(q, coordinates, {.use_cutoff = false, .n_symetries = 125, .L_=30});

    auto coordinates_float = coordinate_vector_cast<float>(coordinates);
    run_example(q, coordinates_float, {.use_cutoff = false, .n_symetries = 1});
    run_example(q, coordinates_float, {.use_cutoff = false, .n_symetries = 27});
    run_example(q, coordinates_float, {.use_cutoff = true, .n_symetries = 1});
    run_example(q, coordinates_float, {.use_cutoff = true, .n_symetries = 27});
    run_example(q, coordinates_float, {.use_cutoff = true, .n_symetries = 27, .L_=50});
    run_example(q, coordinates_float, {.use_cutoff = false, .n_symetries = 27, .L_=50});
    run_example(q, coordinates_float, {.use_cutoff = false, .n_symetries = 125, .L_=30});

}