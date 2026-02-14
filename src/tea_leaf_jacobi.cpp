#include "include/tea.h"
#include "include/data.h"
#include "include/definitions.h"
#include "include/kernels/tea_leaf_jacobi_kernel.h"

namespace TeaLeaf {

/**
 * Orchestrates the Jacobi solver across all tiles in the current MPI rank.
 * Uses OpenMP to process tiles in parallel.
 */
void tea_leaf_jacobi_solve(double& error) {
    double total_tile_error = 0.0;

    // Distribute tiles among OpenMP threads. 
    // We use a reduction on total_tile_error to aggregate results from all threads.
    #ifdef OMP
    #pragma omp parallel reduction(+:total_tile_error)
    #endif
    {
        #ifdef OMP
        #pragma omp for
        #endif
        for (int t = 0; t < tiles_per_task; ++t) {
            double tile_error = 0.0;
            auto& f = chunk.tiles[t].field;

            /**
             * The actual numerical work happens in the kernel.
             * * @param u0        : The initial state (source term)
             * @param u         : The current solution (updated in-place)
             * @param vector_r  : Used here as 'un' (values from previous iteration)
             * @param rx, ry    : Pre-calculated coefficients (dt / dx^2)
             * @param vector_Kx : Conductivity coefficients in X
             * @param vector_Ky : Conductivity coefficients in Y
             */
            tea_leaf_jacobi_solve_kernel(
                f.x_min, f.x_max, f.y_min, f.y_max,
                chunk.halo_exchange_depth,
                f.rx, f.ry,
                f.vector_Kx, f.vector_Ky,
                tile_error,
                f.u0, f.u, f.vector_r 
            );

            total_tile_error += tile_error;
        }
    }
    
    // Final error for this MPI task (will be summed globally in tea_solve.cpp)
    error = total_tile_error;
}

} // namespace TeaLeaf