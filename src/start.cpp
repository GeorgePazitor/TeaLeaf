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

    timee = 0;
    step = 0;
    dt = dtinit;

    tea_barrier(); 

    tea_decompose(grid.x_cells, grid.y_cells);

    chunk.tiles.resize(tiles_per_task);

    chunk.x_cells = chunk.right - chunk.left;
    chunk.y_cells = chunk.top - chunk.bottom;

    chunk.chunk_x_min = 1;
    chunk.chunk_y_min = 1;

    chunk.chunk_x_max = chunk.x_cells;
    chunk.chunk_y_max = chunk.y_cells;

    tea_decompose_tiles(chunk.x_cells, chunk.y_cells);

    for (t = 0; t< tiles_per_task; t++){
        chunk.tiles[t].x_cells = chunk.tiles[t].right - chunk.tiles[t].left;
        chunk.tiles[t].y_cells = chunk.tiles[t].top - chunk.tiles[t].bottom;

        chunk.tiles[t].field.x_min = 1;
        chunk.tiles[t].field.y_min = 1;
        chunk.tiles[t].field.x_max = chunk.tiles[t].x_cells;
        chunk.tiles[t].field.y_max = chunk.tiles[t].y_cells;
    }

    if(parallel.boss){
        *g_out << "Tile size "<<chunk.tiles[0].x_cells<<" by "<<chunk.tiles[0].y_cells<<" cells \n";

        *g_out << "Sub-tile size ranges from "  << std::floor((double)chunk.tiles[tiles_per_task].x_cells / (double)chunk.sub_tile_dims[0]) << " by " 
                                                << std::floor((double)chunk.tiles[tiles_per_task].y_cells / (double)chunk.sub_tile_dims[1]) << " cells to" 
                                                << std::ceil((double)chunk.tiles[0].x_cells / (double)chunk.sub_tile_dims[0]) << " by "
                                                << std::ceil((double)chunk.tiles[0].y_cells / (double)chunk.sub_tile_dims[1]) << " cells \n";
                                               
    }

    build_field(); 
    tea_allocate_buffers();
    initialise_chunk(); 

    if(parallel.boss){
        *g_out << " Generating chunk \n";

    }

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

    set_field(); //TODO

    field_summary(); //TODO

    //if (visit_frequency != 0) visit();

    tea_barrier();

    //profiler_on=profiler_original

}   