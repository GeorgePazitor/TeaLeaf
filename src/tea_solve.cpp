#include "tea.h"
#include "data.h"
#include "global_mpi.h"
#include "update_halo.h"
#include "definitions.h"
#include "tea_leaf_common.h"
#include "tea_leaf_jacobi.h"
#include <cmath>
#include <iomanip>

namespace TeaLeaf {

void tea_leaf() {
    int n = 0;
    double error = 0.0, initial_residual = 0.0;
    std::vector<int> fields(NUM_FIELDS, 0);

    // Initialisation commune
    fields[FIELD_ENERGY1] = 1;
    fields[FIELD_DENSITY] = 1;
    update_halo(fields.data(), chunk.halo_exchange_depth);
    tea_leaf_init_common();

    // Calcul du résidu initial (Lignes 102-114 Fortran)
    fields.assign(NUM_FIELDS, 0);
    fields[FIELD_U] = 1;
    update_halo(fields.data(), 1);

    tea_leaf_calc_residual();
    tea_leaf_calc_2norm(1, initial_residual);
    tea_allsum(initial_residual);
    initial_residual = std::sqrt(std::abs(initial_residual));

    if (parallel.boss && verbose_on) {
        *g_out << "Initial residual " << std::scientific << std::setprecision(6) << initial_residual << std::endl;
    }

    // Boucle de résolution Jacobi (Ligne 140 Fortran)
    for (n = 1; n <= max_iters; ++n) {
        error = 0.0;

        // Appel du wrapper de gestion des tiles
        tea_leaf_jacobi_solve(error);

        // Somme globale (MPI) de l'erreur
        tea_allsum(error);

        // Update Halos APRES le calcul (Ligne 331 Fortran)
        fields.assign(NUM_FIELDS, 0);
        fields[FIELD_U] = 1;
        update_halo(fields.data(), 1);

        // Racine carrée de l'erreur pour la convergence (Ligne 339 Fortran)
        error = std::sqrt(std::abs(error));

        if (parallel.boss && verbose_on) {
            *g_out << "Residual " << std::scientific << std::setprecision(6) << error << std::endl;
        }

        // Test de convergence
        if (error < eps * initial_residual) break;
    }

    tea_leaf_finalise();

    if (parallel.boss) {
        *g_out << "Conduction error " << std::scientific << std::setprecision(7) << (error / initial_residual) << std::endl;
        *g_out << "Iteration count " << std::setw(8) << (n > max_iters ? max_iters : n) << std::endl;
    }
}
}