#include <benchmark/benchmark.h>

#include <sim_seq.h>
#include <sim_sycl.h>

#include <random>

template<typename T> static inline std::vector<coordinate<T>> generate_particules(size_t size, T spacing) {
    std::vector<coordinate<T>> out(size);
    for (auto& particule: out) { particule = generate_random_value<T>(0., std::pow(size, 0.33) * spacing); }
    return out;
}


template<typename T> void sequential_simulation_benchmark(benchmark::State& state) {
    const simulation_configuration<T> config{.use_cutoff = false, .n_symetries = 27};
    const auto size = static_cast<size_t>(state.range(0));
    const auto vec = generate_particules<T>(size, 10 * config.r_star_);

    auto simulation = simulation_state(vec, config);   // preheat
    for (auto _: state) {
        auto start = std::chrono::high_resolution_clock::now();
        simulation.run_iter();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) / 1e9;
        state.SetIterationTime(duration);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * size * size * config.n_symetries));
}

template<typename T> void lennard_jones_sycl(benchmark::State& state) {
    const simulation_configuration<T> config{.use_cutoff = false, .n_symetries = 27};
    const auto size = static_cast<size_t>(state.range(0));
    const auto particules_host = generate_particules<T>(size, 10 * config.r_star_);

    sycl::queue q{sycl::default_selector{}};
    auto particules_device = std::span(sycl::malloc_device<coordinate<T>>(particules_host.size(), q), particules_host.size());
    auto forces_device = std::span(sycl::malloc_device<coordinate<T>>(particules_host.size(), q), particules_host.size());
    auto in_evt = q.memcpy(particules_device.data(), particules_host.data(), particules_device.size_bytes());

    run_simulation_sycl_device_memory(q, particules_device, forces_device, config, in_evt);   // Preheat
    for (auto _: state) {
        auto start = std::chrono::high_resolution_clock::now();
        run_simulation_sycl_device_memory(q, particules_device, forces_device, config);
        auto end = std::chrono::high_resolution_clock::now();
        state.SetIterationTime(static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) / 1e9);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * size * size * config.n_symetries));

    sycl::free(particules_device.data(), q);
    sycl::free(forces_device.data(), q);
}

#ifdef BUILD_HALF
static inline auto lennard_jones_sycl_half(auto& state) { lennard_jones_sycl<sycl::half>(state); }
static inline auto lennard_jones_sequential_half(auto& state) { lennard_jones_sequential<sycl::half>(state); }
BENCHMARK(lennard_jones_sycl_half)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(256, 262144)->UseManualTime();
BENCHMARK(lennard_jones_sequential_half)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(256, 4096)->UseManualTime();
#endif

#ifdef BUILD_FLOAT
static inline auto lennard_jones_sycl_float(auto& state) { lennard_jones_sycl<float>(state); }
static inline auto lennard_jones_sequential_float(auto& state) { sequential_simulation_benchmark<float>(state); }
BENCHMARK(lennard_jones_sycl_float)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(256, 262144)->UseManualTime();
BENCHMARK(lennard_jones_sequential_float)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(256, 4096)->UseManualTime();
#endif

#ifdef BUILD_DOUBLE
static inline auto lennard_jones_sycl_double(auto& state) { lennard_jones_sycl<double>(state); }
static inline auto lennard_jones_sequential_double(auto& state) { sequential_simulation_benchmark<double>(state); }
BENCHMARK(lennard_jones_sycl_double)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(256, 262144)->UseManualTime();
BENCHMARK(lennard_jones_sequential_double)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(256, 4096)->UseManualTime();
#endif

BENCHMARK_MAIN();