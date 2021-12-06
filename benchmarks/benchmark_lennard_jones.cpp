#include <benchmark/benchmark.h>

#include <sim_sycl.h>
#include <sim_seq.h>

#include <random>

using U = double;

static const simulation_configuration<U> config{.use_cutoff=false, .n_symetries=1};


template<typename T, class ForwardIt>
static inline void rand_fill_on_host(ForwardIt first, ForwardIt last, T min, T max) {
    std::mt19937 engine(0);
    auto generator = [&]() {
        std::uniform_real_distribution<T> distribution(min, max);
        return coordinate<T>{distribution(engine), distribution(engine), distribution(engine)};
    };
    std::generate(first, last, generator);
}

template<typename T>
static inline std::vector<coordinate<T>> generate_particules(size_t size, T spacing) {
    std::vector<coordinate<T>> out(size);
    rand_fill_on_host<T>(out.begin(), out.end(), 0., std::pow(size, 0.33) * spacing);
    return out;
}


void lennard_jones_sequential(benchmark::State &state) {
    auto size = static_cast<size_t>(state.range(0));
    auto vec = generate_particules<U>(size, 10 * config.r_star_);
    size_t processed_items = 0;
    U last_energy = 0;
    run_simulation_sequential(vec, config); // preheat ?
    for (auto _: state) {
        auto[field, sums, energy] = run_simulation_sequential(vec, config);
        processed_items += size * size * config.n_symetries;
        last_energy = energy;
    }
    state.SetLabel(std::string(std::to_string(last_energy)));
    state.SetItemsProcessed(static_cast<int64_t>(processed_items));
}

void lennard_jones_sycl(benchmark::State &state) {
    auto size = static_cast<size_t>(state.range(0));
    sycl::queue q{sycl::default_selector{}};

    auto particules_host = generate_particules<U>(size, 10 * config.r_star_);
    auto particules_device = std::span(sycl::malloc_device<coordinate<U>>(particules_host.size(), q), particules_host.size());
    q.memcpy(particules_device.data(), particules_host.data(), particules_device.size_bytes()).wait();
    auto forces_device = std::span(sycl::malloc_device<coordinate<U>>(particules_host.size(), q), particules_host.size());

    size_t processed_items = 0;
    auto[summed_forces, energy] =  run_simulation_sycl_device_memory(q, particules_device, forces_device, config);
    for (auto _: state) {
        run_simulation_sycl_device_memory(q, particules_device, forces_device, config);
        processed_items += size * size * config.n_symetries;
    }

    state.SetLabel(std::string(std::to_string(energy)));
    state.SetItemsProcessed(static_cast<int64_t>(processed_items));

    sycl::free(particules_device.data(), q);
    sycl::free(forces_device.data(), q);
}


BENCHMARK(lennard_jones_sycl)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(256, 1024 * 1048576);
BENCHMARK(lennard_jones_sequential)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(256, 1048576);

BENCHMARK_MAIN();