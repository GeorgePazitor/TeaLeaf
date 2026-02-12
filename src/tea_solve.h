#ifndef TEA_DATA_H
#define TEA_DATA_H

#include <vector>
#include <string>

namespace TeaLeaf {

    // --- Structures de configuration ---
    struct ParallelConfig {
        bool boss;          // Vrai si rang 0
        int max_task;       // Nombre total de rangs MPI
    };

    struct GridConfig {
        int x_cells;
        int y_cells;
    };

    struct ProfilerData {
        double timestep = 0.0;
        double halo_exchange = 0.0;
        double halo_update = 0.0;
        double internal_halo_update = 0.0;
        double summary = 0.0;
        double visit = 0.0;
        double tea_init = 0.0;
        double tea_solve = 0.0;
        double tea_reset = 0.0;
        double dot_product = 0.0;
        double set_field = 0.0;
    };

    // --- Variables Globales (Externes) ---
    // Le mot-clé 'extern' indique que la mémoire est allouée dans un .cpp
    extern ParallelConfig parallel;
    extern GridConfig grid;
    extern ProfilerData profiler;

    extern int step;
    extern double timee;       // Votre variable de temps (timee)
    extern double dt;
    extern double g_small;
    extern double end_time;
    extern int end_step;

    extern int summary_frequency;
    extern int visit_frequency;
    extern bool complete;
    extern bool profiler_on;

    // --- Signatures des fonctions utilitaires ---
    double timer();
    void tea_allgather(double value, std::vector<double>& values);
    void tea_finalize();
    
    // Ajoutez ici les déclarations des drivers appelés dans diffuse
    void timestep();
    void tea_leaf();
    void field_summary();
    void visit();

} // namespace TeaLeaf

#endif