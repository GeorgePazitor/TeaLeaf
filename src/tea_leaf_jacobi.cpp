#include "include/tea_leaf_jacobi.h"
#include "include/data.h"
#include "include/definitions.h"
#include "include/kernels/tea_leaf_jacobi_kernel.h"

namespace TeaLeaf {

void tea_leaf_jacobi_solve(double& error) {
    double total_error = 0.0;
    int tiles_per_task = chunk.tiles.size();

    for (int t = 0; t < tiles_per_task; ++t) {
        double tile_error = 0.0;
        auto& f = chunk.tiles[t].field; 

        tea_leaf_jacobi_solve_kernel(
            f.x_min, f.x_max, f.y_min, f.y_max,
            chunk.halo_exchange_depth,
            f.rx, f.ry,
            f.vector_Kx, f.vector_Ky,
            tile_error,
            f.u0, f.u, f.vector_r
        );
        total_error += tile_error;
    }
    error = total_error;
}

} // namespace TeaLeaf
