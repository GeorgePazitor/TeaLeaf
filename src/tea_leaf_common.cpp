#include <vector>
#include <cmath>
#include <omp.h>
#include <array>

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

            // Calcul de rx et ry
            // En C++, l'indexation commence à 0, donc x_min est déjà l'index correct
            f.rx = dt / std::pow(f.celldx[f.x_min], 2);
            f.ry = dt / std::pow(f.celldy[f.y_min], 2);

            // Détection des conditions aux limites (Faces externes du domaine global)
            for (int i = 0; i < 4; ++i) {
                // EXTERNAL_FACE est une constante (souvent -1) définie dans definitions.h
                if (tile.tile_neighbours[i] == EXTERNAL_FACE && 
                    chunk.chunk_neighbours[i] == EXTERNAL_FACE) {
                    zero_boundary[i] = true;
                } else {
                    zero_boundary[i] = false;
                }
            }

            // Appel du kernel d'initialisation
            tea_leaf_common_init_kernel(
                f.x_min, f.x_max, f.y_min, f.y_max,
                chunk.halo_exchange_depth,
                zero_boundary,
                reflective_boundary, // Variable globale
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
        double tile_norm = 0.0;

        #pragma omp for
        for (int t = 0; t < tiles_per_task; ++t) {
            auto& f = chunk.tiles[t].field;
            tile_norm = 0.0;

            if (norm_array == 0) { // u0.u0
                tea_leaf_calc_2norm_kernel(
                    f.x_min, f.x_max, f.y_min, f.y_max,
                    chunk.halo_exchange_depth, f.u0, tile_norm
                );
            } 
            else if (norm_array == 1) { // r.r
                tea_leaf_calc_2norm_kernel(
                    f.x_min, f.x_max, f.y_min, f.y_max,
                    chunk.halo_exchange_depth, f.vector_r, tile_norm
                );
            }
            // else: Gérer l'erreur via un logger ou std::cerr

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