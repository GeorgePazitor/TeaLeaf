#include <vector>
#include <cmath>
#include <omp.h>
#include <array>
#include <iostream>

#include "include/data.h"
#include "include/tea.h"
#include "include/definitions.h"
#include "include/kernels/tea_leaf_common_kernel.h" 

namespace TeaLeaf {

/**
 * Common initialization for all solvers.
 * Sets up coefficients (rx, ry), conductivity fields (Kx, Ky), and initial vectors.
 */
void tea_leaf_init_common() {
    #ifdef OMP
    #pragma omp parallel
    #endif
    {
        std::array<bool, 4> zero_boundary;

        #ifdef OMP
        #pragma omp for
        #endif
        for (int t = 0; t < tiles_per_task; ++t) {
            auto& tile = chunk.tiles[t];
            auto& f = tile.field;

            //scale diffusion in x
            f.rx = dt / (f.celldx[0] * f.celldx[0]);
            f.ry = dt / (f.celldy[0] * f.celldy[0]);

            for (int i = 0; i < 4; ++i) {
                zero_boundary[i] = (tile.tile_neighbours[i] == EXTERNAL_FACE && 
                                    chunk.chunk_neighbours[i] == EXTERNAL_FACE);
            }

            tea_leaf_common_init_kernel(
                f.x_min, f.x_max, f.y_min, f.y_max,
                chunk.halo_exchange_depth,
                zero_boundary,
                reflective_boundary,
                f.density, f.energy1, f.u, f.u0,
                f.vector_r, f.vector_w, f.vector_Kx, f.vector_Ky,
                f.vector_Di, f.tri_cp, f.tri_bfp, f.vector_Mi,
                f.rx, f.ry,
                tl_preconditioner_type, coefficient
            );
        }
    }
}

/**
 * Calculates the residual r = b - Ax.
 * This tells us how far the current solution is from the exact solution.
 */
void tea_leaf_calc_residual() {
    #ifdef OMP
    #pragma omp parallel for
    #endif
    for (int t = 0; t < tiles_per_task; ++t) {
        auto& f = chunk.tiles[t].field;
        tea_leaf_calc_residual_kernel(
            f.x_min, f.x_max, f.y_min, f.y_max,
            chunk.halo_exchange_depth,
            f.u, f.u0, f.vector_r, f.vector_Kx, f.vector_Ky,
            f.vector_Di, f.rx, f.ry
        );
    }
}

/**
 * Calculates the L2 norm of a vector (u0 or residual r).
 * Used to monitor convergence during iterations.
 */
void tea_leaf_calc_2norm(int norm_array, double& norm) {
    double total_norm = 0.0;

    #ifdef OMP
    #pragma omp parallel reduction(+:total_norm)
    #endif
    {
        #ifdef OMP
        #pragma omp for
        #endif
        for (int t = 0; t < tiles_per_task; ++t) {
            double tile_norm = 0.0;
            auto& f = chunk.tiles[t].field;

            // Select array based on norm_array index (0: u0, 1: r)
            const std::vector<double>& arr = (norm_array == 0) ? f.u0 : f.vector_r;

            tea_leaf_calc_2norm_kernel(
                f.x_min, f.x_max, f.y_min, f.y_max,
                chunk.halo_exchange_depth, arr, tile_norm
            );

            total_norm += tile_norm;
        }
    }
    norm = total_norm;
}

/**
 * Copies the final solution back into the energy field.
 */
void tea_leaf_finalise() {
    #ifdef OMP
    #pragma omp parallel for
    #endif
    for (int t = 0; t < tiles_per_task; ++t) {
        auto& f = chunk.tiles[t].field;
        tea_leaf_kernel_finalise(
            f.x_min, f.x_max, f.y_min, f.y_max,
            chunk.halo_exchange_depth,
            f.energy1, f.density, f.u
        );
    }
}

} // namespace TeaLeaf