#pragma once

#include <fstream>
#include <iostream>
#include <sim_common.hpp>
#include <string>
#include <vector>

template<typename T, typename Other> static inline std::vector<coordinate<T>> coordinate_vector_cast(const std::vector<coordinate<Other>>& in) {
    std::vector<coordinate<T>> out(in.size());
    for (unsigned int i = 0; i < in.size(); ++i) {
        coordinate<T> c;
        c.x() = in[i].x();
        c.y() = in[i].y();
        c.z() = in[i].z();
        out[i] = c;
    }
    return out;
}

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
        coordinates.emplace_back(c);
    }
    return coordinates;
}