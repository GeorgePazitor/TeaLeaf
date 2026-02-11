#include "update_halo.h"
#include "tea.h"
#include "data.h"
#include "definitions.h"
#include "pack.h" 
#include "kernels/update_halo_kernel.h"
#include "kernels/update_internal_halo_kernel.h" 
#include <mpi.h>
#include <omp.h>
#include <algorithm> // std::any_of 
using namespace TeaLeaf;


void update_halo(const int* fields, int depth) {

    double halo_time = 0.0;
    if (profiler_on) halo_time = MPI_Wtime();

    // MPI halo exchange, defined in pack_module 
    tea_exchange(fields, depth); 

    if (profiler_on) {
        profiler.halo_exchange += (MPI_Wtime() - halo_time);
    }

    // physical coundary conditions (reflective)
    update_boundary(fields, depth);

    // internal tile exchange 
    update_tile_boundary(fields, depth);
}

// considering reflective wall(if enabled)
void update_boundary(const int* fields, int depth) {

    double halo_time = 0.0;
    if (profiler_on) halo_time = MPI_Wtime();

    // check if we are on a boundary and if reflection is enabled
    bool is_external = false;
    for (int n : chunk.chunk_neighbours) {
        if (n == EXTERNAL_FACE) {
            is_external = true;
            break;
        }
    }

    if (reflective_boundary && is_external) {
        
        #pragma omp parallel for
        for (int t = 0; t < tiles_per_task; ++t) {
            auto& tile = chunk.tiles[t];

            update_halo_kernel(
                tile.field.x_min,
                tile.field.x_max,
                tile.field.y_min,
                tile.field.y_max,
                chunk.halo_exchange_depth,
                chunk.chunk_neighbours,
                tile.tile_neighbours, // Pass vector<int>
                
                // Fields
                tile.field.density.data(),
                tile.field.energy0.data(),
                tile.field.energy1.data(),
                tile.field.u.data(),
                tile.field.vector_p.data(),
                tile.field.vector_sd.data(),
                tile.field.vector_rtemp.data(),
                tile.field.vector_z.data(),
                tile.field.vector_Kx.data(),
                tile.field.vector_Ky.data(),
                tile.field.vector_Di.data(),
                fields,
                depth
            );
        }
    }

    if (profiler_on) {
        profiler.halo_update += (MPI_Wtime() - halo_time);
    }
}

// update Internal tile boundaries (shared memory copy)

void update_tile_boundary(const int* fields, int depth) {

    double halo_time = 0.0;
    if (profiler_on) halo_time = MPI_Wtime();

    // Only needed if we decomposed the rank into multiple tiles
    if (tiles_per_task > 1) {
        
        #pragma omp parallel
        {
            // --- Pass 1: Left / Right Exchange ---
            #pragma omp for nowait
            for (int t = 0; t < tiles_per_task; ++t) {
                // Get neighbor index (0-based adjustment if your neighbors are 1-based)
                // Assuming neighbor array stores 0-based indices for C++
                int right_idx = chunk.tiles[t].tile_neighbours[CHUNK_RIGHT];

                if (right_idx != EXTERNAL_FACE) {
                    // Call kernel to copy from Tile T to Tile RIGHT_IDX
                    // Note: You need to implement update_internal_halo_left_right_kernel
                    update_internal_halo_left_right_kernel(
                        chunk.tiles[t].field.x_min,
                        chunk.tiles[t].field.x_max,
                        chunk.tiles[t].field.y_min,
                        chunk.tiles[t].field.y_max,
                        chunk.tiles[t].field.density.data(),
                        chunk.tiles[t].field.energy0.data(),
                        chunk.tiles[t].field.energy1.data(),
                        chunk.tiles[t].field.u.data(),
                        chunk.tiles[t].field.vector_p.data(),
                        chunk.tiles[t].field.vector_sd.data(),
                        chunk.tiles[t].field.vector_rtemp.data(),
                        chunk.tiles[t].field.vector_z.data(),
                        chunk.tiles[t].field.vector_Kx.data(),
                        chunk.tiles[t].field.vector_Ky.data(),
                        chunk.tiles[t].field.vector_Di.data(),
                        
                        // Neighbor Data
                        chunk.tiles[right_idx].field.x_min,
                        chunk.tiles[right_idx].field.x_max,
                        chunk.tiles[right_idx].field.y_min,
                        chunk.tiles[right_idx].field.y_max,
                        chunk.tiles[right_idx].field.density.data(),
                        chunk.tiles[right_idx].field.energy0.data(),
                        chunk.tiles[right_idx].field.energy1.data(),
                        chunk.tiles[right_idx].field.u.data(),
                        chunk.tiles[right_idx].field.vector_p.data(),
                        chunk.tiles[right_idx].field.vector_sd.data(),
                        chunk.tiles[right_idx].field.vector_rtemp.data(),
                        chunk.tiles[right_idx].field.vector_z.data(),
                        chunk.tiles[right_idx].field.vector_Kx.data(),
                        chunk.tiles[right_idx].field.vector_Ky.data(),
                        chunk.tiles[right_idx].field.vector_Di.data(),
                        
                        chunk.halo_exchange_depth,
                        fields,
                        depth
                    );
                }
            }
            
            // Barrier needed between L/R and T/B updates? 
            // Fortran logic suggests barrier if depth > 1 inside kernels, 
            // but for safety in shared memory copies, a barrier here is wise.
            #pragma omp barrier

            // --- Pass 2: Top / Bottom Exchange ---
            #pragma omp for nowait
            for (int t = 0; t < tiles_per_task; ++t) {
                int up_idx = chunk.tiles[t].tile_neighbours[CHUNK_TOP];

                if (up_idx != EXTERNAL_FACE) {
                    update_internal_halo_bottom_top_kernel(
                         chunk.tiles[t].field.x_min,
                        chunk.tiles[t].field.x_max,
                        chunk.tiles[t].field.y_min,
                        chunk.tiles[t].field.y_max,
                        chunk.tiles[t].field.density.data(),
                        chunk.tiles[t].field.energy0.data(),
                        chunk.tiles[t].field.energy1.data(),
                        chunk.tiles[t].field.u.data(),
                        chunk.tiles[t].field.vector_p.data(),
                        chunk.tiles[t].field.vector_sd.data(),
                        chunk.tiles[t].field.vector_rtemp.data(),
                        chunk.tiles[t].field.vector_z.data(),
                        chunk.tiles[t].field.vector_Kx.data(),
                        chunk.tiles[t].field.vector_Ky.data(),
                        chunk.tiles[t].field.vector_Di.data(),
                        
                        // Neighbor Data
                        chunk.tiles[up_idx].field.x_min,
                        chunk.tiles[up_idx].field.x_max,
                        chunk.tiles[up_idx].field.y_min,
                        chunk.tiles[up_idx].field.y_max,
                        chunk.tiles[up_idx].field.density.data(),
                        chunk.tiles[up_idx].field.energy0.data(),
                        chunk.tiles[up_idx].field.energy1.data(),
                        chunk.tiles[up_idx].field.u.data(),
                        chunk.tiles[up_idx].field.vector_p.data(),
                        chunk.tiles[up_idx].field.vector_sd.data(),
                        chunk.tiles[up_idx].field.vector_rtemp.data(),
                        chunk.tiles[up_idx].field.vector_z.data(),
                        chunk.tiles[up_idx].field.vector_Kx.data(),
                        chunk.tiles[up_idx].field.vector_Ky.data(),
                        chunk.tiles[up_idx].field.vector_Di.data(),
                        
                        chunk.halo_exchange_depth,
                        fields,
                        depth
                    );
                }
            }
        } 
    }

    if (profiler_on) {
        profiler.internal_halo_update += (MPI_Wtime() - halo_time);
    }
}