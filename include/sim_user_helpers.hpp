#pragma once

#include <fstream>
#include <internal/cpp_utils.hpp>
#include <internal/sim_math.hpp>

namespace sim {
/**
 * Converts a vector of coordinates<Src_T> to a vector of coordinates<Dst_T>
 * @tparam Dst_T
 * @tparam Src_T
 * @param in
 * @return
 */
template<typename Dst_T, typename Src_T> static inline std::vector<coordinate<Dst_T>> coordinate_vector_cast(const std::vector<coordinate<Src_T>>& in) {
    std::vector<coordinate<Dst_T>> out(in.size());
    for (unsigned int i = 0; i < in.size(); ++i) { out[i] = coordinate<Dst_T>{static_cast<Dst_T>(in[i].x()), static_cast<Dst_T>(in[i].y()), static_cast<Dst_T>(in[i].z())}; }
    return out;
}

/**
 * Reads the particule file and returns a vector of coordinates (doubles)
 * @param filename
 * @return
 */
static inline std::vector<coordinate<double>> parse_particule_file(std::string&& filename) {
    auto fs = std::ifstream(filename);
    if (!fs.is_open()) throw std::runtime_error("File not found");
    auto comment = std::string{};
    std::getline(fs, comment);
    std::cout << "Comment is: " << comment << std::endl << std::endl;
    auto coordinates = std::vector<coordinate<double>>{};
    while (!fs.eof()) {
        auto tmp = 0;
        coordinate<double> c{};
        fs >> tmp >> c.x() >> c.y() >> c.z();
        if (tmp == 0) continue;
        coordinates.emplace_back(c);
    }
    return coordinates;
}

static inline std::vector<coordinate<double>> parse_vit_file(std::string&& filename) {
    auto fs = std::ifstream(filename);
    if (!fs.is_open()) throw std::runtime_error("File not found");
    auto coordinates = std::vector<coordinate<double>>{};
    while (!fs.eof()) {
        auto tmp = 0;
        coordinate<double> c{};
        fs >> tmp >> c.x() >> c.y() >> c.z();
        if (tmp == 0) continue;
        coordinates.emplace_back(c);
    }
    return coordinates;
}


}   // namespace sim
