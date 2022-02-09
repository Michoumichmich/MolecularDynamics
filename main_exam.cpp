/**
 * @author Michel Migdal
 * Settings used to generate exam files.
 */

#include <sim>   // Header that takes care of including GPU support if needed.

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./" << argv[0] << "particles.xyz speeds.vit" << std::endl;
        return 1;
    }

    /* Loading the data */
    const auto coordinates = sim::parse_particule_file(argv[1]);
    const auto speeds = sim::parse_vit_file(argv[2]);
    std::cout << "Read " << coordinates.size() << " coordinates and " << speeds.size() << " speeds." << std::endl;

    /* Configuration of the simulation */
    size_t n = 10000;
    {
        sim::configuration<double> config{
                .m_i = 12.,                                // Weight
                .dt = 1,                                   // Time step in fs
                .T0 = 120.0,                               // Target temperature
                .use_berdensten_thermostate = false,       // Disabling the thermostate
                .r_star = 3.730,                           //
                .epsilon_star = 0.29304,                   //
                .r_cut = 10,                               //
                .n_symetries = 27,                         // 1, 27 or 125
                .L = 70,                                   // Symetries box width
                .iter_per_frame = 100,                     // Save a frame every 100 iters
                .out_file = "without_decompisition.pdb",   //
                .store_lennard_jones_metrics = true        //
        };
        auto md_simulation = sim::molecular_dynamics<double, sim::cpu_backend_regular>(coordinates, speeds, config);

        // Run the simulation
        for (size_t i = 0; i < n; ++i) {
            if (i % 100 == 0) std::cout << md_simulation << std::endl;
            md_simulation.run_iter();
        }
    }


    /* Alternative version that uses the domain decomposition, undomment me and the above one. */
    {
        sim::configuration<double> config{
                .m_i = 12.,                             // Weight
                .dt = 0.1,                              // Time step in fs
                .T0 = 120.0,                            // Target temperature
                .use_berdensten_thermostate = true,     // Disabling the thermostate
                .r_star = 3.730,                        //
                .epsilon_star = 0.29304,                //
                .r_cut = 5,                             //
                .n_symetries = 125,                     // 1, 27 or 125
                .domain_mins{-25, -25, -25},            // Domain decomposition settings
                .domain_maxs{+25, +25, +25},            // Domain decomposition settings
                .domain_widths{+8, +8, +8},             // Domain decomposition settings
                .iter_per_frame = 1000,                 // Save a frame every 100 iters
                .out_file = "with_decompisition.pdb",   //
                .store_lennard_jones_metrics = true     //
        };
        auto md_simulation = sim::molecular_dynamics<double, sim::cpu_backend_decompose>(coordinates, speeds, config);
        for (size_t i = 0; i < n; ++i) {
            if (i % 500 == 0) std::cout << md_simulation << std::endl;
            md_simulation.run_iter();
        }
    }


    return 0;
}
