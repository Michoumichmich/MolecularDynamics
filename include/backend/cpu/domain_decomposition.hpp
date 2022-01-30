#pragma once

template<typename T, bool is_domain_decomposer = true> struct domain_decomposer {
public:
    static constexpr bool is_domain_decomposer_ = true;
    using index_t = int32_t;

    domain_decomposer() = default;

    explicit domain_decomposer(sim::coordinate<T> min, sim::coordinate<T> max, sim::coordinate<T> width)
        : min_(min), max_(max), width_(width),   //
          domains_sizes(std::ceil((max.x() - min.x()) / width.x()), std::ceil((max.y() - min.y()) / width.y()), std::ceil((max.z() - min.z()) / width.z())) {
        sim::internal::assume(width.x() > 0 && width.y() > 0 && width.z() > 0);
        sim::internal::assume(min.x() < max.x());
        sim::internal::assume(min.y() < max.y());
        sim::internal::assume(min.z() < max.z());
        sim::internal::assume(domains_sizes.x() > 0);
        sim::internal::assume(domains_sizes.y() > 0);
        sim::internal::assume(domains_sizes.z() > 0);
    }


    inline void update_domains(const std::vector<sim::coordinate<T>>& coordinates) const noexcept {
        const auto max_size = max_domain_size(coordinates);
        particles_buffer.clear();
        particles_buffer.reserve(max_size);
        const auto coordinates_size = static_cast<index_t>(coordinates.size());
        domains = std::vector<std::vector<index_t>>(get_domains_count());
        for (auto& domain: domains) { domain.reserve(max_size); }

        for (index_t i = 0; i < coordinates_size; ++i) {
            auto c = coordinates[i];
            try {
                const auto domain_id = linearize(bind_coordinate_to_domain(c));
                domains[domain_id].push_back(i);
            } catch (...) {}
        }
    }

    template<int n_syms = 125, typename func> inline void run_kernel_on_domains(const std::vector<sim::coordinate<T>>& particles, func&& kernel) const noexcept {
        const auto num_domains = static_cast<index_t>(domains.size());
        sim::internal::assume(num_domains == get_domains_count());
        for (index_t current_domain_id = 0; current_domain_id < num_domains; ++current_domain_id) {

            const auto domain_coordinates = delinearize(current_domain_id);
            index_t n_neighbors = 0;

            std::array<std::pair<index_t, sycl::vec<T, 3>>, n_syms> neighbors{};
            for (const auto& verlet_sym: sim::get_symetries<n_syms>()) {
                sycl::vec<T, 3> deltas{};
                auto neighbor_domain_coordinates = domain_coordinates + verlet_sym;
                for (int dim = 0; dim < 3; ++dim) {
                    if (neighbor_domain_coordinates[dim] >= domains_sizes[dim]) {
                        neighbor_domain_coordinates[dim] = domains_sizes[dim] - verlet_sym[dim];   //   (verlet_sym[dim] - domains_sizes[dim]) % domains_sizes[dim];
                        deltas[dim] = width_[dim];
                    }

                    if (neighbor_domain_coordinates[dim] < 0) {
                        neighbor_domain_coordinates[dim] = verlet_sym[dim] + domains_sizes[dim];
                        deltas[dim] = -width_[dim];
                    }
                }
                auto neighbor_linear_id = linearize(neighbor_domain_coordinates);
                sim::internal::assume(neighbor_linear_id < num_domains);
                neighbors[n_neighbors].first = neighbor_linear_id;
                neighbors[n_neighbors].second = deltas;
                ++n_neighbors;
            }

            for (const index_t& current_domain_particle_id: domains[current_domain_id]) {
                const auto current_particle = particles[current_domain_particle_id];
                for (const auto& neighbor: neighbors) {
                    const auto delta = neighbor.second;
                    std::fill(particles_buffer.begin(), particles_buffer.end(), sim::coordinate<T>{});
                    for (const index_t& other_particle_id: domains[neighbor.first]) {
                        /**
                         * If the two particles are different, its always ok, else we have the same particle id twice, so we must ensure that the
                         * delta is not null.
                         */
                        if (current_domain_particle_id != other_particle_id || (delta.x() != 0 && delta.y() != 0 && delta.z() != 0)) {
                            particles_buffer.template emplace_back(particles[other_particle_id] + delta);
                        }
                    }

                    for (const auto& other_particle: particles_buffer) { kernel(current_domain_particle_id, current_particle, other_particle); }
                }
            }
        }
    }

private:
    [[nodiscard]] inline constexpr index_t linearize(const sycl::vec<index_t, 3>& domain_ids) const {
        return domain_ids.x() * domains_sizes.y() * domains_sizes.z() + domain_ids.y() * domains_sizes.z() + domain_ids.z();
    }

    [[nodiscard]] inline sycl::vec<index_t, 3> delinearize(index_t id) const {
        index_t z = id % domains_sizes.z();
        index_t y = (id - z) % domains_sizes.y();
        index_t x = (id - z - y * domains_sizes.z()) / (domains_sizes.y() * domains_sizes.z());
        return sycl::vec<index_t, 3>{x, y, z};
    }
    [[nodiscard]] inline constexpr index_t get_domains_count() const { return domains_sizes.x() * domains_sizes.y() * domains_sizes.z(); }
    [[nodiscard]] inline sycl::vec<index_t, 3> bind_coordinate_to_domain(sim::coordinate<T> coord) const {
        if (coord.x() < min_.x() || coord.x() > max_.x() || coord.y() < min_.y() || coord.y() > max_.y() || coord.z() < min_.z() || coord.z() > max_.z()) {
            throw std::runtime_error("Coordinate does not fall in range");
        }

        const auto tmp = (coord - min_) / width_;
        return sycl::vec<index_t, 3>{std::floor(tmp.x()), std::floor(tmp.y()), std::floor(tmp.z())};
    }
    [[nodiscard]] inline index_t max_domain_size(const std::vector<sim::coordinate<T>>& coordinates) const noexcept {
        std::vector<index_t> counts(get_domains_count(), 0);
        for (const auto& c: coordinates) {
            try {
                ++counts[linearize(bind_coordinate_to_domain(c))];
            } catch (...) {}
        }
        return *std::max_element(counts.begin(), counts.end());
    }

private:
    sim::coordinate<T> min_{}, max_{}, width_{};
    sycl::vec<index_t, 3> domains_sizes{};
    mutable std::vector<sim::coordinate<T>> particles_buffer{};
    mutable std::vector<std::vector<index_t>> domains{};
};


/**
 * Dummy
 * @tparam T
 */
template<typename T> struct domain_decomposer<T, false> {
public:
    static constexpr bool is_domain_decomposer_ = false;
    domain_decomposer() = default;
    explicit domain_decomposer(T box_width) : L(box_width) {}
    template<int n_syms = 125, typename func> inline void run_kernel_on_domains(const std::vector<sim::coordinate<T>>& particles, func&& kernel) const noexcept {

#pragma omp parallel for default(none) shared(particles, kernel, L)
        for (auto i = 0U; i < particles.size(); ++i) {
            for (auto j = 0U; j < particles.size(); ++j) {

#pragma unroll
                for (const auto& sym: sim::get_symetries<n_syms>()) {
                    if (i == j && sym.x() == 0 && sym.y() == 0 && sym.z() == 0) continue;
                    const sim::coordinate<T> delta{sym.x() * L, sym.y() * L, sym.z() * L};
                    const auto other_particule = delta + particles[j];
                    const auto this_particule = particles[i];
                    kernel(i, this_particule, other_particule);
                }
            }
        }
    }

private:
    T L{};
};
