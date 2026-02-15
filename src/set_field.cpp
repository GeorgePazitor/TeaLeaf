#include "include/set_field.h"
#include "include/tea.h"
#include "include/data.h"
#include "include/definitions.h"
#include "include/kernels/set_field_kernel.h"
#include <mpi.h>
#include <omp.h>

/**
 * High-level routine to initialize or reset simulation fields.
 * Dispatches the workload across multiple tiles using OpenMP threads.
 */
void set_field() {
    using namespace TeaLeaf;

    double start_time = 0.0;
    if (profiler_on) start_time = MPI_Wtime();

    // Parallelize field initialization across available tiles
    #pragma omp parallel
    {
        // nowait allows threads to proceed to the next task without a barrier 
        // if they finish their allocated tiles early.
        #pragma omp for nowait
        for (int t = 0; t < tiles_per_task; ++t) {
            
            auto& tile = chunk.tiles[t];

            // Kernel call to perform the actual memory operations on energy fields.
            // Pass interior bounds and the halo depth to ensure correct grid alignment.
            set_field_kernel(
                tile.field.x_min,
                tile.field.x_max,
                tile.field.y_min,
                tile.field.y_max,
                chunk.halo_exchange_depth,
                tile.field.energy0.data(),
                tile.field.energy1.data()
            );
        }
    }

    if (profiler_on) {
        profiler.set_field += (MPI_Wtime() - start_time);
    }
}