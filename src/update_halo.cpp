#include "include/update_halo.h"
#include "include/tea.h"
#include "include/data.h"
#include "include/definitions.h"
#include "include/pack.h" 
#include "include/kernels/update_halo_kernel.h"
#include "include/kernels/update_internal_halo_kernel.h" 
#include <mpi.h>
#include <omp.h>
using namespace TeaLeaf;

/**
 * Manages the full halo exchange process across MPI ranks, 
 * physical boundaries, and internal tiles.
 */
void update_halo(const int* fields, int depth) {

    double halo_time = 0.0;
    if (profiler_on) halo_time = MPI_Wtime();

    tea_exchange(fields, depth); 

    if (profiler_on) {
        profiler.halo_exchange += (MPI_Wtime() - halo_time);
    }

    update_boundary(fields, depth);

    update_tile_boundary(fields, depth);
}

/**
 * Handles physical boundary conditions (e.g., reflective walls).
 * Checks if the current chunk sits on the global domain edge.
 */
void update_boundary(const int* fields, int depth) {

    double halo_time = 0.0;
    if (profiler_on) halo_time = MPI_Wtime();

    bool is_external = false;
    for (int n : chunk.chunk_neighbours) {
        if (n == EXTERNAL_FACE) {
            is_external = true;
            break;
        }
    }

    //apply kernels only if reflective boundaries are enabled and we are at the edge
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
                tile.tile_neighbours, 
                
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

/**
 * Performs shared-memory data copies between tiles inside a single MPI rank.
 * Required when the domain is decomposed into multiple tiles per task.
 */
void update_tile_boundary(const int* fields, int depth) {

    double halo_time = 0.0;
    if (profiler_on) halo_time = MPI_Wtime();

    if (tiles_per_task > 1) {
        
        #pragma omp parallel
        {
            #pragma omp for nowait
            for (int t = 0; t < tiles_per_task; ++t) {
                int right_idx = chunk.tiles[t].tile_neighbours[CHUNK_RIGHT];

                if (right_idx != EXTERNAL_FACE) {
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
            
            #pragma omp barrier

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