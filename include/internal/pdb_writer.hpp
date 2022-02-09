#pragma once

#include <internal/cpp_utils.hpp>
#include <internal/sim_math.hpp>

#include <fstream>

namespace sim {
using namespace std::string_literals;

class pdb_writer {
public:
    explicit pdb_writer(const std::string& pdb_particles_out, bool store_opt_data)
        : out_file_(pdb_particles_out), gnuplot_ostream_(store_opt_data ? pdb_particles_out + ".gplt" : ""s), pdb_ostrem_(pdb_particles_out) {
        gnuplot_ostream_ << "# idx Temp Epot" << std::endl;
        //    if (file_name.empty()) { std::cout << "Empty file name given. " << std::endl; }
    }

    template<typename T> void store_new_iter(int i, const std::vector<coordinate<T>> particules, T temp, T epot) {
        if (gnuplot_ostream_.is_open()) { gnuplot_ostream_ << i << "   " << temp << "   " << epot << std::endl; }
        if (!pdb_ostrem_.is_open()) return;
        counter++;
        static const char pdb_atom_fmt[] = "ATOM  %5d %4s%1c%3s %1c%4d%1c   %8.3f%8.3f%8.3f%6.2f%6.2f      %4s%2s%2s";
        pdb_ostrem_ << "CRYST1 50 50 50 90.00 90.00 90.00 P 1\n";
        pdb_ostrem_ << "MODEL " << i << '\n';
        for (unsigned j = 0; j < particules.size(); ++j) {
            auto& particule = particules[j];
            char line[100];
            const char* empty = " ";
            std::sprintf(line, pdb_atom_fmt, j, "C", ' ', empty, ' ', 0, ' ', (double) particule.x(), (double) particule.y(), (double) particule.z(), 1., 1., empty, empty, "C");
            pdb_ostrem_ << line << '\n';
        }
        pdb_ostrem_ << "TER \n"
                       "ENDMDL\n";
        pdb_ostrem_.flush();
        std::cout << "[PDB_WRITER] Frame: " << i << ", sent to: " << out_file_ << ", frames written in total: " << counter << '\n';
    }


private:
    std::string out_file_;
    std::ofstream gnuplot_ostream_;
    std::ofstream pdb_ostrem_;
    size_t counter = 0;
};
}   // namespace sim
