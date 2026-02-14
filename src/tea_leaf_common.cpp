#include <vector>
#include <cmath>
#include <omp.h>
#include <array>
#include <iostream>

#include "data.h"
#include "tea.h"
#include "definitions.h"
#include "kernels/tea_leaf_common_kernel.h" 

namespace TeaLeaf {

// --- INITIALISATION COMMUNE ---
void tea_leaf_init_common() {
    #pragma omp parallel
    {
        std::array<bool, 4> zero_boundary;

        #pragma omp for
        for (int t = 0; t < tiles_per_task; ++t) {
            auto& tile = chunk.tiles[t];
            auto& f = tile.field;

            // Calcul de rx et ry : conforme au Fortran utilisant celldx(x_min)
            // En C++, l'index 0 correspond au premier élément de la grille utile
            f.rx = dt / (f.celldx[0] * f.celldx[0]);
            f.ry = dt / (f.celldy[0] * f.celldy[0]);

            // Détection des faces externes du domaine global
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

// --- CALCUL DU RÉSIDU ---
void tea_leaf_calc_residual() {
    #pragma omp parallel for
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

// --- CALCUL DE LA NORME L2 ---
void tea_leaf_calc_2norm(int norm_array, double& norm) {
    double total_norm = 0.0;

    #pragma omp parallel reduction(+:total_norm)
    {
        #pragma omp for
        for (int t = 0; t < tiles_per_task; ++t) {
            double tile_norm = 0.0;
            auto& f = chunk.tiles[t].field;

            // Sélection du tableau selon norm_array (0: u0, 1: r)
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

// --- FINALISATION ---
void tea_leaf_finalise() {
    #pragma omp parallel for
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