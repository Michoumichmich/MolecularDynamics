#include <benchmark/benchmark.h>
#include <random>
#include <sim>

template<typename T> static inline std::vector<sim::coordinate<T>> generate_particules(size_t size, T spacing) {
    std::vector<sim::coordinate<T>> out(size);
    auto max = spacing / 2;
    std::generate(out.begin(), out.end(), [&]() -> sim::coordinate<T> {
        return {sim::internal::generate_random_value<T>(-max, max), sim::internal::generate_random_value<T>(-max, max), sim::internal::generate_random_value<T>(-max, max)};
    });
    return out;
}


template<typename T> void benchmark_backend_decompose(benchmark::State& state) {
    int range = 3000;
    const sim::configuration<T> config{
            .dt = 0.05,
            .use_berdensten_thermostate = false,
            .r_cut = 100,
            .n_symetries = 125,
            .domain_mins{-range, -range, -range},
            .domain_maxs{+range, +range, +range},
            .domain_widths{50, 50, 50},
            .out_file = "",
    };
    const auto size = static_cast<size_t>(state.range(0));
    const auto coordinates = generate_particules<T>(size, range);
    auto simulation = sim::molecular_dynamics<T, sim::cpu_backend_decompose>(coordinates, config);
    simulation.run_iter();

    for (auto _: state) {
        auto start = std::chrono::high_resolution_clock::now();
        simulation.run_iter();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) / 1e9;
        state.SetIterationTime(duration);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * size * size * config.n_symetries));
    //state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * size));
    state.SetLabel("virtual interaction simulation speed");   // Speed if we were not using Domain Decomposition
}


#ifdef BUILD_DOUBLE
static inline auto backend_decompose_double(auto& state) { benchmark_backend_decompose<double>(state); }
BENCHMARK(backend_decompose_double)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(256, 2621440)->UseManualTime();
#endif

#ifdef BUILD_FLOAT
static inline auto backend_decompose_float(auto& state) { benchmark_backend_decompose<float>(state); }
BENCHMARK(backend_decompose_float)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(256, 2621440)->UseManualTime();
#endif

#ifdef BUILD_HALF
static inline auto backend_decompose_half(auto& state) { benchmark_backend_decompose<sycl::half>(state); }
BENCHMARK(backend_decompose_half)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(256, 26214)->UseManualTime();
#endif


BENCHMARK_MAIN();