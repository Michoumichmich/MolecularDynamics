#pragma once

#include <fstream>
#include <internal/sim_common.hpp>


class pdb_writer {


public:
    explicit pdb_writer(const std::string& file_name) : filename_(file_name), fs(file_name) {
        if (file_name.empty()) { std::cout << "Empty file name given. " << std::endl; }
        //   fs << "CRYST1 30 30 30 90.00 90.00 90.00 P 1\n";
        //   fs.flush();
    }

    template<typename T> void store_new_iter(const std::vector<coordinate<T>> particules, int i) {
        if (!fs.is_open()) return;

        static const char atom_line_oformat_[] = "ATOM  %5d %4s%1c%3s %1c%4d%1c   %8.3f%8.3f%8.3f%6.2f%6.2f      %4s%2s%2s";
        fs << "CRYST1 50 50 50 90.00 90.00 90.00 P 1\n";
        fs << "MODEL " << i << '\n';
        for (unsigned j = 0; j < particules.size(); ++j) {
            auto& particule = particules[j];
            char line[100];
            char alt = ' ';
            char chain = ' ';
            const char* empty = " ";
            std::sprintf(line, atom_line_oformat_, j, "C", alt, empty, chain, 0, ' ', particule.x(), particule.y(), particule.z(), 1., 1., empty, empty, "C");

            fs << line << '\n';
            //  fs << "ATOM " << j << " C 0 " << std::fixed << std::setprecision(3) << particule.x() << " " << particule.y() << ' ' << particule.z() << "     MRES\n";
        }
        fs << "TER \n"
              "ENDMDL\n";
        fs.flush();
        std::cout << "[PDB_WRITER] Frame: " << i << ", sent to: " << filename_ << '\n';
    }


private:
    std::string filename_;
    std::ofstream fs;
};