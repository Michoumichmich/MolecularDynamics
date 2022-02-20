/**
 * @author Michel Migdal
 * Sample file for the MD simulation (missng a real front-end sorry).
 */

#include <sim>

static bool use_sycl = false;
static bool use_domain_decomposition = false;

using namespace std::string_literals;

template<typename T> void run_example(size_t n, const std::vector<sim::coordinate<T>>& coordinates, sim::configuration<T> config = {}) {
    std::cout << config << std::endl;
    //static_assert(sizeof(sim::coordinate<T>) == 4 * sizeof(T));
    auto run = [&](auto simulation) mutable {
        constexpr int print_periodicity = 100;
        for (size_t i = 0; i < n; ++i) {
            if (i % print_periodicity == 0) std::cout << simulation << std::endl;
            simulation.run_iter();
        }
    };

    if (use_sycl) {
#if BUILD_SYCL
        auto q = sycl::queue{sycl::default_selector{}};
        std::cout << "Running on SYCL with " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
        run(sim::molecular_dynamics<T, sim::sycl_backend>(coordinates, config, sim::sycl_backend<T>{coordinates.size(), q}));
#endif
    } else {
        if (use_domain_decomposition) {
            std::cout << "Running on CPU/openMP with domain decomposition." << std::endl;
            run(sim::molecular_dynamics<T, sim::cpu_backend_decompose>(coordinates, config));
        } else {
            std::cout << "Running on CPU/openMP WITHOUT domain decomposition." << std::endl;
            run(sim::molecular_dynamics<T, sim::cpu_backend_regular>(coordinates, config));
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./" << argv[0] << "particle_file.xyz" << std::endl;
        return 1;
    }

    if (argc >= 3 && "-sycl"s == argv[2]) { use_sycl = true; }
    if (argc >= 3 && "-decompose"s == argv[2]) { use_domain_decomposition = true; }
    if (argc > 3) { std::cerr << "Extra arguments ignored" << std::endl; }

    const auto coordinates_double = sim::parse_particule_file(argv[1]);

#ifdef BUILD_DOUBLE

    /* The good one? */
    run_example(100'000, coordinates_double,
                {
                        .dt = 0.05,
                        .use_berdensten_thermostate = true,
                        .r_cut = 8,
                        .n_symetries = 125,
                        .domain_mins{-20, -20, -20},
                        .domain_maxs{+20, +20, +20},
                        .iter_per_frame = 100,
                });

    /* Default simulation */
    run_example(100'000, coordinates_double);

    /* For the domain decomposition */
    run_example(100'000, coordinates_double, {.dt = 0.1, .use_berdensten_thermostate = true, .r_cut = 8, .n_symetries = 125});

    /* Explodes! */
    run_example(1'000, coordinates_double, {.dt = 16, .iter_per_frame = 1});


#endif

#ifdef BUILD_FLOAT
    {
        const auto coordinates_float = sim::coordinate_vector_cast<float>(coordinates_double);
        run_example(100'000, coordinates_float);
        run_example(100'000, coordinates_float, {.dt = 0.1, .use_berdensten_thermostate = false});
        run_example(1'000, coordinates_float, {.dt = 16, .iter_per_frame = 1});
        run_example(10'000'000, coordinates_float, {.dt = 1, .use_berdensten_thermostate = false, .n_symetries = 1, .iter_per_frame = 10'000});
        run_example(100'000, coordinates_float, {.dt = 0.1, .r_cut = 70, .n_symetries = 125, .L = 35, .iter_per_frame = 100});
    };
#endif

#ifdef BUILD_HALF
    {
        const auto coordinates_half = sim::coordinate_vector_cast<sycl::half>(coordinates_double);
        run_example(100'000, coordinates_half);
        run_example(100'000, coordinates_half, {.dt = 0.1, .use_berdensten_thermostate = false});
        run_example(100'000, coordinates_half, {.dt = 16, .iter_per_frame = 1});
        run_example(1'000, coordinates_half, {.dt = 1, .use_berdensten_thermostate = false, .n_symetries = 1});
        run_example(100'000, coordinates_half, {.dt = 0.1, .r_cut = 70, .n_symetries = 125, .L = 35, .iter_per_frame = 100});
    };
#endif
}
