#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

#include "data.h"
#include "tea.h"
#include "global_mpi.h"
#include "update_halo.h"
#include "definitions.h"
#include "tea_leaf_common.h"
#include "kernels/tea_leaf_jacobi_kernel.h"

namespace TeaLeaf {

void tea_leaf() {
    int n;
    double error = 0.0;
    double initial_residual = 0.0;
    
    // Identification des champs pour les mises à jour de halos
    std::vector<int> fields(NUM_FIELDS, 0);

    // --- PHASE D'INITIALISATION ---
    
    // Mise à jour initiale pour l'énergie et la densité (nécessaire pour Kx, Ky)
    fields[FIELD_ENERGY1] = 1;
    fields[FIELD_DENSITY] = 1;
    update_halo(fields.data(), chunk.halo_exchange_depth);

    // Initialisation des coefficients (rx, ry, Kx, Ky, Di, etc.)
    tea_leaf_init_common();

    // Calcul du résidu initial pour le critère de convergence
    tea_leaf_calc_residual();
    tea_leaf_calc_2norm(1, initial_residual); // norm_array=1 correspond à r.r
    tea_allsum(initial_residual);             // Réduction MPI
    initial_residual = std::sqrt(std::abs(initial_residual));

    if (parallel.boss && verbose_on) {
        *g_out << "Initial residual: " << std::scientific << std::setprecision(6) 
               << initial_residual << std::endl;
    }

    // Préparation pour la boucle Jacobi (mise à jour de u)
    std::fill(fields.begin(), fields.end(), 0);
    fields[FIELD_U] = 1;

    // --- BOUCLE PRINCIPALE JACOBI ---

    for (n = 1; n <= max_iters; ++n) {
        double step_error = 0.0;

        // Appel du kernel sur chaque tuile (OpenMP interne au wrapper/kernel)
        // Note : En Jacobi, on calcule u1 à partir de u0 (fixe) et un (ancienne itération)
        for (int t = 0; t < tiles_per_task; ++t) {
            auto& f = chunk.tiles[t].field;
            double tile_error = 0.0;

            tea_leaf_jacobi_solve_kernel(
                f.x_min, f.x_max, f.y_min, f.y_max,
                chunk.halo_exchange_depth, f.rx, f.ry,
                f.vector_Kx, f.vector_Ky,
                tile_error,
                f.u0, f.u, f.vector_w // vector_w est souvent utilisé comme 'un'
            );
            step_error += tile_error;
        }

        // Réduction globale de l'erreur (Somme sur tous les processus MPI)
        tea_allsum(step_error);
        error = step_error; 

        // Mise à jour des halos pour la prochaine itération
        update_halo(fields.data(), 1);

        // Vérification de la convergence
        if (error < eps * initial_residual) {
            if (parallel.boss && verbose_on) {
                *g_out << "Jacobi converged at iteration " << n 
                       << ". Error: " << error << std::endl;
            }
            break;
        }
    }

    // --- FINALISATION ---
    // Conversion de u (densité * énergie) en énergie finale
    tea_leaf_finalise();
}

} // namespace TeaLeaf