#include "tea.h"
#include "data.h"
#include "definitions.h"
#include "kernels/tea_leaf_jacobi_kernel.h"

namespace TeaLeaf {

void tea_leaf_jacobi_solve(double& error) {
    double total_tile_error = 0.0;

    // OpenMP sur les tiles (private tile_error pour la réduction)
    #pragma omp parallel reduction(+:total_tile_error)
    {
        #pragma omp for
        for (int t = 0; t < tiles_per_task; ++t) {
            double tile_error = 0.0;
            auto& f = chunk.tiles[t].field;

            // Appel du kernel (vector_r est utilisé pour stocker 'un')
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
    error = total_tile_error;
}
}