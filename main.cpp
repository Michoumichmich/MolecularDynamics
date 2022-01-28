#include <sim>

template<typename T> void run_example(size_t n, const std::vector<sim::coordinate<T>>& coordinates, sim::configuration<T> config = {}) {
    std::cout << config << std::endl;
    //auto simulation = sim::molecular_dynamics<T, sim::cpu_backend>(coordinates, config);
    auto simulation = sim::molecular_dynamics<T, sim::sycl_backend>(coordinates, config, sim::sycl_backend<T>{coordinates.size()});
    for (size_t i = 0; i < n; ++i) {
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
    /* Default simulation */
    run_example(100'000, coordinates_double);
    //run_example(1'000'000, coordinates_double, {.dt = 0.1, .iter_per_frame = 1'000});

    /* Without the thermostate */
    run_example(100'000, coordinates_double, {.dt = 0.1, .use_berdensten_thermostate = false});

    /* One that explodes! */
    run_example(1'000, coordinates_double, {.dt = 16, .iter_per_frame = 1});
#endif

#ifdef BUILD_FLOAT
    auto coordinates_float = sim::coordinate_vector_cast<float>(coordinates_double);
    /* Default simulation */
    run_example(100'000, coordinates_float);

    /* Without the thermostate */
    run_example(100'000, coordinates_float, {.dt = 0.1, .use_berdensten_thermostate = false});

    /* One that explodes! */
    run_example(1'000, coordinates_float, {.dt = 16, .iter_per_frame = 1});
#endif

#ifdef BUILD_HALF
    auto coordinates_half = sim::coordinate_vector_cast<sycl::half>(coordinates_double);
    /* Default simulation */
    run_example(100'000, coordinates_half);

    /* Without the thermostate */
    run_example(100'000, coordinates_half, {.dt = 0.1, .use_berdensten_thermostate = false});

    /* One that explodes! */
    run_example(1'000, coordinates_half, {.dt = 16, .iter_per_frame = 1});
#endif
}