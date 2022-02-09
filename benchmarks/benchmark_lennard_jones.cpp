#include <benchmark/benchmark.h>
#include <random>
#include <sim>

template<typename T> static inline std::vector<sim::coordinate<T>> generate_particules(size_t size, T spacing) {
    std::vector<sim::coordinate<T>> out(size);
    auto max = std::pow(size, 0.33) * spacing;
    std::generate(out.begin(), out.end(), [&]() -> sim::coordinate<T> {
        return {sim::internal::generate_random_value<T>(-max, max), sim::internal::generate_random_value<T>(-max, max), sim::internal::generate_random_value<T>(-max, max)};
    });
    return out;
}

template<typename T> void cpu_backend_benchmark_impl(benchmark::State& state) {
    const sim::configuration<T> config{
            .dt = 0.0001,
            .use_berdensten_thermostate = false,
            .n_symetries = 27,
            .out_file = "",
    };
    const auto size = static_cast<size_t>(state.range(0));
    const auto coordinates = generate_particules<T>(size, 10 * config.r_star);
    auto simulation = sim::molecular_dynamics<T, sim::cpu_backend_regular>(coordinates, config);
    simulation.run_iter();

    for (auto _: state) {
        auto start = std::chrono::high_resolution_clock::now();
        simulation.run_iter();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) / 1e9;
        state.SetIterationTime(duration);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * size * size * config.n_symetries));
}

template<typename T> void sycl_backend_benchmark_impl(benchmark::State& state) {
    const sim::configuration<T> config{
            .dt = 0.0001,
            .use_berdensten_thermostate = false,
            .n_symetries = 27,
            .out_file = "",
    };
    const auto size = static_cast<size_t>(state.range(0));
    const auto coordinates = generate_particules<T>(size, 10 * config.r_star);

    auto simulation = sim::molecular_dynamics<T, sim::sycl_backend>(coordinates, config, sim::sycl_backend<T>{coordinates.size()});
    simulation.run_iter();

    for (auto _: state) {
        auto start = std::chrono::high_resolution_clock::now();
        simulation.run_iter();
        auto end = std::chrono::high_resolution_clock::now();
        state.SetIterationTime(static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) / 1e9);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * size * size * config.n_symetries));
}

#ifdef BUILD_FLOAT
static inline auto lennard_jones_sycl_float(auto& state) { sycl_backend_benchmark_impl<float>(state); }
static inline auto lennard_jones_sequential_float(auto& state) { cpu_backend_benchmark_impl<float>(state); }
BENCHMARK(lennard_jones_sycl_float)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(256, 262144)->UseManualTime();
BENCHMARK(lennard_jones_sequential_float)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(256, 4096)->UseManualTime();
#endif

#ifdef BUILD_DOUBLE
static inline auto lennard_jones_sycl_double(auto& state) { sycl_backend_benchmark_impl<double>(state); }
static inline auto lennard_jones_sequential_double(auto& state) { cpu_backend_benchmark_impl<double>(state); }
BENCHMARK(lennard_jones_sycl_double)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(256, 262144)->UseManualTime();
BENCHMARK(lennard_jones_sequential_double)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(256, 4096)->UseManualTime();
#endif


#ifdef BUILD_HALF
static inline auto lennard_jones_sycl_half(auto& state) { sycl_backend_benchmark_impl<sycl::half>(state); }
static inline auto lennard_jones_sequential_half(auto& state) { cpu_backend_benchmark_impl<sycl::half>(state); }
BENCHMARK(lennard_jones_sycl_half)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(256, 262144)->UseManualTime();
BENCHMARK(lennard_jones_sequential_half)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(256, 4096)->UseManualTime();
#endif


BENCHMARK_MAIN();