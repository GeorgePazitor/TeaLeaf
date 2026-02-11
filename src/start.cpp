#include "start.h"
#include "tea.h"
#include "data.h"
#include "definitions.h"
#include "global_mpi.h"
#include "build_field.h"
#include "initialise_chunk.h"
#include "generate_chunk.h"
#include "update_halo.h"
#include "set_field.h"
#include "field_summary.h"

#include <cmath>

void start() {
    using namespace TeaLeaf;

    int t;
    std::array<int, NUM_FIELDS> fields = {0};

    //bool profiler_original;

    //profiler_original = profiler_on;

    //profiler_on = false;

    if(parallel.boss){
        *g_out << "Setting up initial geometry" << "\n";
    }
    timee = 0.0;
    step = 0;
    dt = dtinit;

    tea_barrier(); 

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

    if(parallel.boss){
        *g_out << " Tile size "<<chunk.tiles[0].x_cells<<" by "<<chunk.tiles[0].y_cells<<" cells \n";

    if (chunk.sub_tile_dims[0] == 0 || chunk.sub_tile_dims[1] == 0) {
        *g_out << " Errorr : sub_tile_dims cannot be 0.\n";
    } else {
        int last_tile_index = tiles_per_task - 1;
        *g_out << " Sub-tile size ranges from "
       << std::floor((double)chunk.tiles[last_tile_index].x_cells / (double)chunk.sub_tile_dims[0]) << " by "
       << std::floor((double)chunk.tiles[last_tile_index].y_cells / (double)chunk.sub_tile_dims[1]) << " cells to "
       << std::ceil((double)chunk.tiles[0].x_cells / (double)chunk.sub_tile_dims[0]) << " by "
       << std::ceil((double)chunk.tiles[0].y_cells / (double)chunk.sub_tile_dims[1]) << " cells \n";

    }

                                               
    }

    build_field(); 
    tea_allocate_buffers();
    initialise_chunk(); 

    if(parallel.boss){
        *g_out << " Generating chunk \n";

    }
    // Recalcul des dimensions globales de la grille
    grid.x_cells = mpi_dims[0] * chunk.tile_dims[0] * chunk.sub_tile_dims[0];
    grid.y_cells = mpi_dims[1] * chunk.tile_dims[1] * chunk.sub_tile_dims[1];

    generate_chunk(); 

    fields[FIELD_DENSITY] = 1;
    fields[FIELD_ENERGY0] = 1;
    fields[FIELD_ENERGY1] = 1;

    update_halo(fields.data(), chunk.halo_exchange_depth); 

    if(parallel.boss){
        *g_out << " \n Problem initialized and generated \n";
    }
    set_field();             // Copie energy0 vers energy1
    field_summary();         // Calcule la masse/énergie totale initiale
    //if (visit_frequency != 0) {
    //    visit();             // Sortie fichier pour visualisation
    //}
    tea_barrier();
    //profiler_on = profiler_original;
}
