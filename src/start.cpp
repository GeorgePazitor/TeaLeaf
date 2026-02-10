#include "definitions.h"
#include "data.h"
#include "tea.h"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace TeaLeaf {

void start() {
    int fields[NUM_FIELDS] = {0};
    bool profiler_original;
    // Désactiver le profiler pendant l'initialisation
    profiler_original = profiler_on;
    profiler_on = false;
    bool is_boss = true; 
    // Si MPI est utilisé, is_boss = (rank == 0)
    if (is_boss) {
        std::cout << "Setting up initial geometry" << std::endl;
    }
    time = 0.0;
    step = 0;
    dt = dtinit;
    tea_barrier();
    // Décomposition du domaine (MPI)
    tea_decompose(grid.x_cells, grid.y_cells);
    // Allocation des tuiles
    // tiles_per_task doit être calculé par tea_decompose
    chunk.tiles.resize(tiles_per_task); 
    // Calcul de la taille du chunk (sous-grille locale au processus)
    chunk.x_cells = chunk.right - chunk.left + 1;
    chunk.y_cells = chunk.top - chunk.bottom + 1;
    chunk.chunk_x_min = 1;
    chunk.chunk_y_min = 1;
    chunk.chunk_x_max = chunk.x_cells;
    chunk.chunk_y_max = chunk.y_cells;
    // Décomposition en tuiles à l'intérieur du chunk
    tea_decompose_tiles(chunk.x_cells, chunk.y_cells);
    for (int t = 0; t < tiles_per_task; ++t) {
        chunk.tiles[t].x_cells = chunk.tiles[t].right - chunk.tiles[t].left + 1;
        chunk.tiles[t].y_cells = chunk.tiles[t].top - chunk.tiles[t].bottom + 1;
        chunk.tiles[t].field.x_min = 1;
        chunk.tiles[t].field.y_min = 1;
        chunk.tiles[t].field.x_max = chunk.tiles[t].x_cells;
        chunk.tiles[t].field.y_max = chunk.tiles[t].y_cells;
    }
    if (is_boss) {
        std::cout << "Tile size " << chunk.tiles[0].x_cells << " by " 
                  << chunk.tiles[0].y_cells << " cells" << std::endl;
        // Calcul des ranges de sub-tiles (exactitude mathématique du Fortran)
        double sub_x = (double)chunk.sub_tile_dims[0];
        double sub_y = (double)chunk.sub_tile_dims[1];
        int min_sub_x = std::floor(chunk.tiles[tiles_per_task - 1].x_cells / sub_x);
        int min_sub_y = std::floor(chunk.tiles[tiles_per_task - 1].y_cells / sub_y);
        int max_sub_x = std::ceil(chunk.tiles[0].x_cells / sub_x);
        int max_sub_y = std::ceil(chunk.tiles[0].y_cells / sub_y);
        std::cout << "Sub-tile size ranges from " << min_sub_x << " by " << min_sub_y
                  << " cells to " << max_sub_x << " by " << max_sub_y << " cells" << std::endl;
    }
    build_field();           // Alloue la mémoire des vecteurs dans field_type
    tea_allocate_buffers();   // Alloue les buffers MPI
    initialise_chunk();      // Prépare les données géométriques
    if (is_boss) {
        std::cout << "Generating chunk" << std::endl;
    }
    // Recalcul des dimensions globales de la grille
    grid.x_cells = mpi_dims[0] * chunk.tile_dims[0] * chunk.sub_tile_dims[0];
    grid.y_cells = mpi_dims[1] * chunk.tile_dims[1] * chunk.sub_tile_dims[1];
    generate_chunk();        // Remplit les champs (densité, énergie) selon les "states"
    // Initialisation des halos
    fields[FIELD_DENSITY] = 1;
    fields[FIELD_ENERGY0] = 1;
    fields[FIELD_ENERGY1] = 1;
    update_halo(fields, chunk.halo_exchange_depth);
    if (is_boss) {
        std::cout << "\nProblem initialised and generated" << std::endl;
    }
    set_field();             // Copie energy0 vers energy1
    field_summary();         // Calcule la masse/énergie totale initiale
    if (visit_frequency != 0) {
        visit();             // Sortie fichier pour visualisation
    }
    tea_barrier();
    profiler_on = profiler_original;
}
}