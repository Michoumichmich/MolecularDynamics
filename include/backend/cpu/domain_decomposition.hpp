#pragma once
#include <omp.h>

namespace sim {

/**
 * @class domain_decomposer
 *
 * This class implements the domain decomposition.
 * This class holds the domains info.
 *
 * @tparam T fp type of the coordinates
 * @tparam is_domain_decomposer if set to false, the domain_decomposer will not perform domain decmposition and will just
 * run the dummy O(n**2) algorithm.
 */
template<typename T, bool is_domain_decomposer = true> struct domain_decomposer {
public:
    static constexpr bool is_domain_decomposer_ = true;   // To know whether it's the real on or the dummy version in a constexpr environment
    using index_t = int32_t;                              // Type used to compute the indices. 32 bits should be enough & better for GPUs.

    domain_decomposer() = default;

    /**
     * Constructor to be used. It does sanity checking on the inputs and that's all
     * @param min Domain boundaries, vector of min_x, min_y and min_z
     * @param max Domain boundaries, vector of max_x, max_y and max_z
     * @param width Size of each domain size_x, size_y, size_z
     * @param decompose_period Every "decompose_period" use of the domain decomposer, the domains will be updated, if the used didn't do it in between (lazy!)
     */
    inline explicit domain_decomposer(coordinate<T> min, coordinate<T> max, coordinate<T> width, int decompose_period)
        : min_(min), max_(max), width_(width),                                                                                                                //
          domain_count(std::ceil((max.x() - min.x()) / width.x()), std::ceil((max.y() - min.y()) / width.y()), std::ceil((max.z() - min.z()) / width.z())),   //
          decompose_period_(decompose_period)                                                                                                                 //
    {
        internal::assume(width.x() > 0 && width.y() > 0 && width.z() > 0);
        internal::assume(min.x() < max.x());
        internal::assume(min.y() < max.y());
        internal::assume(min.z() < max.z());
        internal::assume(domain_count.x() > 0);
        internal::assume(domain_count.y() > 0);
        internal::assume(domain_count.z() > 0);

        particles_buffer = std::vector<std::vector<coordinate<T>>>(omp_get_max_threads());
    }


    /**
     * Takes a vector of coordinates and re-computes the domains accordingly.
     * The method is const as the changes made are not invalidating the object. The user
     * should be able to call it as often as needed.
     *
     * @warning The domain decomposer wraps the particles that are ouf of bounds which means that the coordinates CAN BE MODIFIED.
     * @param coordinates the coordinates
     */
    inline void update_domains(std::vector<coordinate<T>>& coordinates) const noexcept {
        /* Boilerplate setup */
        max_domain_size_cached = max_domain_size(coordinates);
        for (auto& domain: domains) { domain.reserve(max_domain_size_cached); }
        const auto coordinates_size = static_cast<index_t>(coordinates.size());
        domains = std::vector<std::vector<index_t>>(get_domains_count());

        /* Updatind the domains */
        for (index_t i = 0; i < coordinates_size; ++i) {
            auto c = coordinates[i];
            const auto domain_id = linearize(bind_coordinate_to_domain(c));
            domains[domain_id].push_back(i);
        }

        /* Reset counter and debug info */
        iters_since_last_update = 0;
        //std::cout << "[Domain decomposer] Domains were updated, max size is: " << max_domain_size_cached << std::endl;
    }

    /**
     * This is the main method used to launch computations using the domain decomposition. Basically the user provides a lambda function which takes three arguments:
     * particle id, particle corresponding to the id, and another particle. Filling these arguments will be performed by this method.
     * This function might also dispatch the kernels in parallel. We guarantee that no two threads will have the same particle id (first param) and particle associated (2nd param).
     * @tparam n_syms Number of neighbors to consider. (1, 27 or 125)
     * @tparam func the kernel type
     * @param particles the particles to run the kernel on. They must be the same as used in the update method.
     * @param kernel  (void)kernel(int id, particle_id, particle_other)
     */
    template<int n_syms, typename func> inline void run_kernel_on_domains(std::vector<coordinate<T>>& particles, func&& kernel) const noexcept {
        /* Ensuring the particle buffers are large enough to be used without realloc */
        for (auto& buf: particles_buffer) { buf.reserve(n_syms * max_domain_size_cached); }

        /* Eventually update the domains if that was not done in a while */
        if (iters_since_last_update > decompose_period_ || domains.empty()) { update_domains(particles); }
        iters_since_last_update++;

        const auto num_domains = static_cast<index_t>(domains.size());
        internal::assume(num_domains == get_domains_count());

        /* Runs the domains in parallel. Each OMP thread has it's owe particle buffer */
#pragma omp parallel for default(none) shared(particles, kernel, num_domains, particles_buffer, domain_count, max_, min_) schedule(dynamic)   // Dynamic bc unequal work
        for (index_t current_domain_id = 0; current_domain_id < num_domains; ++current_domain_id) {
            auto& tls_buf = particles_buffer[omp_get_thread_num()];
            const auto domain_pos = delinearize(current_domain_id);

            index_t n_neighbors = 0;
            std::array<std::pair<index_t, coordinate<T>>, n_syms> neighbors_domains_config{};
#pragma unroll   // unrollable
            for (const auto& verlet_sym: get_symetries<n_syms>()) {
                coordinate<T> neighbor_delta{};
                auto neighbor_pos = domain_pos + verlet_sym;
#pragma unroll
                /* Apply perodic conditions so the particles do not leave the domain. */
                for (int dim = 0; dim < 3; ++dim) {
                    if (neighbor_pos[dim] >= domain_count[dim]) {
                        neighbor_pos[dim] -= domain_count[dim];
                        neighbor_delta[dim] = max_[dim] - min_[dim];
                    } else if (neighbor_pos[dim] < 0) {
                        neighbor_pos[dim] += domain_count[dim];
                        neighbor_delta[dim] = min_[dim] - max_[dim];
                    }
                }

                const index_t neighbor_linear_id = linearize(neighbor_pos);
                internal::assume(neighbor_linear_id >= 0 && neighbor_linear_id < num_domains);
                neighbors_domains_config[n_neighbors].first = neighbor_linear_id;
                neighbors_domains_config[n_neighbors].second = neighbor_delta;
                ++n_neighbors;
            }


            tls_buf.clear();
            /* Loads all the neighbors in the buffer. "This" domain is also part of the neighbor domains */
            for (const auto& neighbor: neighbors_domains_config) {
                const coordinate<T> neighbor_delta = neighbor.second;
                const index_t neighbor_domain_id = neighbor.first;
                for (const index_t& neighbor_particle_id: domains[neighbor_domain_id]) { tls_buf.template emplace_back(particles[neighbor_particle_id] + neighbor_delta); }
            }

            /* Loop over current domain */
            for (const index_t& current_particle_id: domains[current_domain_id]) {
                const auto current_particle = particles[current_particle_id];
                //#pragma force vectorize ivdep?
                for (const auto& other_particle: tls_buf) { kernel(current_particle_id, current_particle, other_particle); }
            }
        }
    }

private:
    /**
     * Returns the total number of domains (ie. across all the dimensions)
     * @return
     */
    [[nodiscard]] inline constexpr index_t get_domains_count() const { return domain_count.x() * domain_count.y() * domain_count.z(); }

    /**
     * Takes a 3D coordinate and linearizes it to a 1d identifier.
     * @param id
     * @return
     */
    [[nodiscard]] inline constexpr index_t linearize(const sycl::vec<index_t, 3>& id) const {
        return id.x() * domain_count.y() * domain_count.z() + id.y() * domain_count.z() + id.z();
    }

    /**
     * Takes a 1D linear identifier and computes the the 3D coordinates associated.
     * (linearize)o(delinearize) = identity
     * @param id
     * @return
     */
    [[nodiscard]] inline sycl::vec<index_t, 3> delinearize(index_t id) const {
        index_t z = id % domain_count.z();
        index_t y = (id / domain_count.z()) % domain_count.y();
        index_t x = id / (domain_count.y() * domain_count.z());
        return {x, y, z};
    }

    /**
     * Takes a particle coordinates, applies wrapping to put it into our domains and returns the 3D coordinate of the domain where the particle falls.
     * THE PARTICLE CAN BE MODIFIED IF OUT OF BOUNDS!
     * @param coord
     * @return
     */
    [[nodiscard]] inline sycl::vec<index_t, 3> bind_coordinate_to_domain(coordinate<T>& coord) const {
#pragma unroll
        /* Wrap in range */
        for (int dim = 0; dim < 3; ++dim) {
            if (coord[dim] < min_[dim] || coord[dim] >= max_[dim]) { coord[dim] = float_wrap(coord[dim], min_[dim], max_[dim]); }
        }

        const auto tmp = (coord - min_) / width_;
        return sycl::vec<index_t, 3>{std::floor(tmp.x()), std::floor(tmp.y()), std::floor(tmp.z())};
    }

    /**
     * Potentially expensive. It counts the number of particles that fall into each domain and returns the maximum domain occupancy.
     * @param coordinates
     * @return
     */
    [[nodiscard]] inline index_t max_domain_size(std::vector<coordinate<T>>& coordinates) const noexcept {
        std::vector<index_t> counts(get_domains_count(), 0);
        for (auto& c: coordinates) { ++counts[linearize(bind_coordinate_to_domain(c))]; }
        return *std::max_element(counts.begin(), counts.end());
    }

private:
    coordinate<T> min_{}, max_{}, width_{};                               // Domain configuration
    sycl::vec<index_t, 3> domain_count{};                                 // Deduced number of domains in each dimension
    mutable std::vector<std::vector<coordinate<T>>> particles_buffer{};   // Buffer used by the openMP implementation. We have OMP_NUM_THREADS buffers.
    mutable std::vector<std::vector<index_t>> domains{};                  // domain[i] contains the particle IDs that fall into this domain
    size_t decompose_period_ = 1;                                         //
    mutable size_t iters_since_last_update = 0;                           //
    mutable size_t max_domain_size_cached = 0;                            // Caching the max domain size to avoid expensive computation.
};


/**
 * Dummy
 * @tparam T
 */
template<typename T> struct domain_decomposer<T, false> {
public:
    static constexpr bool is_domain_decomposer_ = false;

    /**
     *
     */
    domain_decomposer() = default;

    /**
     *
     * @param box_width
     */
    explicit domain_decomposer(T box_width) : L(box_width) {}

    /**
     *
     * @tparam n_syms
     * @tparam func
     * @param particles
     * @param kernel
     */
    template<int n_syms, typename func> inline void run_kernel_on_domains(const std::vector<coordinate<T>>& particles, func&& kernel) const noexcept {

#pragma omp parallel for default(none) shared(particles, kernel, L) schedule(dynamic)
        for (auto i = 0U; i < particles.size(); ++i) {
            for (auto j = 0U; j < particles.size(); ++j) {
#pragma unroll
                for (const auto& sym: get_symetries<n_syms>()) {
                    const coordinate<T> delta{sym.x() * L, sym.y() * L, sym.z() * L};
                    kernel(i, particles[i], delta + particles[j]);
                }
            }
        }
    }

private:
    T L{};
};
}   // namespace sim