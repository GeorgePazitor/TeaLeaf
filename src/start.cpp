#include "include/start.h"
#include "include/tea.h"
#include "include/data.h"
#include "include/definitions.h"
#include "include/global_mpi.h"
#include "include/build_field.h"
#include "include/initialise_chunk.h"
#include "include/generate_chunk.h"
#include "include/update_halo.h"
#include "include/set_field.h"
#include "include/field_summary.h"
#include <cmath>

/**
 * Coordinates the setup of the simulation geometry and initial state.
 * This is where memory is mapped and the initial physical fields are populated.
 */
void start() {
    using namespace TeaLeaf;

    std::array<int, NUM_FIELDS> fields = {0};

    if(parallel.boss){
        *g_out << "Setting up initial geometry" << "\n";
    }

    // Initialize simulation time and step counter
    timee = 0.0;
    step = 0;
    dt = dtinit;

    // Synchronize all ranks before starting allocation
    tea_barrier(); 

    // MPI Decomposition: Calculate which part of the global grid this rank owns.
    tea_decompose(grid.x_cells, grid.y_cells);

    // Resize the tile vector based on decomposition results
    chunk.tiles.resize(tiles_per_task); 

    // Calculate local chunk dimensions
    chunk.x_cells = chunk.right - chunk.left + 1;
    chunk.y_cells = chunk.top - chunk.bottom + 1;
    chunk.chunk_x_min = 1;
    chunk.chunk_y_min = 1;
    chunk.chunk_x_max = chunk.x_cells;
    chunk.chunk_y_max = chunk.y_cells;

    // Tile Decomposition: Divide the local chunk into smaller tiles for OpenMP threads.
    tea_decompose_tiles(chunk.x_cells, chunk.y_cells);

    for (int t = 0; t < tiles_per_task; ++t) {
        chunk.tiles[t].x_cells = chunk.tiles[t].right - chunk.tiles[t].left + 1;
        chunk.tiles[t].y_cells = chunk.tiles[t].top - chunk.tiles[t].bottom + 1;
        chunk.tiles[t].field.x_min = 1;
        chunk.tiles[t].field.y_min = 1;
        chunk.tiles[t].field.x_max = chunk.tiles[t].x_cells;
        chunk.tiles[t].field.y_max = chunk.tiles[t].y_cells;
    }

    // Log tile information for debugging/tuning
    if(parallel.boss){
        *g_out << " Tile size " << chunk.tiles[0].x_cells << " by " << chunk.tiles[0].y_cells << " cells \n";

        if (chunk.sub_tile_dims[0] == 0 || chunk.sub_tile_dims[1] == 0) {
            *g_out << " Error: sub_tile_dims cannot be 0.\n";
        } else {
            int last_tile_index = tiles_per_task - 1;
            *g_out << " Sub-tile size ranges from "
                   << std::floor((double)chunk.tiles[last_tile_index].x_cells / (double)chunk.sub_tile_dims[0]) << " by "
                   << std::floor((double)chunk.tiles[last_tile_index].y_cells / (double)chunk.sub_tile_dims[1]) << " cells to "
                   << std::ceil((double)chunk.tiles[0].x_cells / (double)chunk.sub_tile_dims[0]) << " by "
                   << std::ceil((double)chunk.tiles[0].y_cells / (double)chunk.sub_tile_dims[1]) << " cells \n";
        }
    }

    // Memory Allocation
    build_field();           // Allocates memory for the physical variables (density, energy, etc.)
    tea_allocate_buffers();  // Allocates MPI communication buffers
    initialise_chunk();      // Sets up initial spatial coordinates (x_area, y_area, etc.)

    if(parallel.boss){
        *g_out << " Generating chunk \n";
    }

    // Refresh global grid dimensions to ensure consistency
    grid.x_cells = mpi_dims[0] * chunk.tile_dims[0] * chunk.sub_tile_dims[0];
    grid.y_cells = mpi_dims[1] * chunk.tile_dims[1] * chunk.sub_tile_dims[1];

    // State Generation
    generate_chunk();        // Populates the fields with the defined material states/shapes

    // Initial Halo Exchange
    fields[FIELD_DENSITY] = 1;
    fields[FIELD_ENERGY0] = 1;
    fields[FIELD_ENERGY1] = 1;
    update_halo(fields.data(), chunk.halo_exchange_depth); 

    if(parallel.boss){
        *g_out << " \n Problem initialized and generated \n";
    }

    // Final setup before main loop
    set_field();             // Clones initial energy state: energy0 -> energy1
    field_summary();         // Performs global reduction to calculate total mass and energy

    tea_barrier();
}