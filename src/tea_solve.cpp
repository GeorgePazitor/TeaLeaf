#include "include/tea.h"
#include "include/data.h"
#include "include/global_mpi.h"
#include "include/update_halo.h"
#include "include/definitions.h"
#include "include/tea_leaf_common.h"
#include "include/tea_leaf_jacobi.h"
#include "include/tea_leaf_cg.h"
#include <cmath>
#include <iomanip>

namespace TeaLeaf {

/**
 * Main solver routine for the heat diffusion equation.
 * Currently implements the Jacobi iterative method.
 */
void tea_leaf() {
    int n = 0;
    double error = 0.0,  initial_residual = 0.0;
    std::vector<int> fields(NUM_FIELDS, 0);

    std::vector<double> cg_alphas(max_iters, 0);
    std::vector<double> cg_betas(max_iters, 0);

    // initialisation commune
    fields[FIELD_ENERGY1] = 1;
    fields[FIELD_DENSITY] = 1;
    update_halo(fields.data(), chunk.halo_exchange_depth);
    
    // Prepare coefficients for the linear system Ax = b
    tea_leaf_init_common();

    // pour le cg
    double rrn, rro, pw, alpha, beta;

    if(tl_use_cg){
        tea_leaf_cg_init(rro);

        tea_allsum(rro);

        fields.assign(NUM_FIELDS, 0);

        fields[FIELD_U] = 1;
        fields[FIELD_P] = 1;

        update_halo(fields.data(), 1);

        fields.assign(NUM_FIELDS, 0);

        fields[FIELD_P] = 1;
    }
    else{// jacobi by default
        // calcul du résidu initial 
        fields.assign(NUM_FIELDS, 0);
        fields[FIELD_U] = 1;
        update_halo(fields.data(), 1);
    }

    tea_leaf_calc_residual();
    tea_leaf_calc_2norm(1, initial_residual);
    
    // Global MPI reduction to get the sum of residuals across all ranks
    tea_allsum(initial_residual);
    initial_residual = std::sqrt(std::abs(initial_residual));

    if (parallel.boss && verbose_on) {
        *g_out << "Initial residual " << std::scientific << std::setprecision(6) << initial_residual << std::endl;
    }

    // boucle de résolution (ligne 140 fortran)
    for (n = 1; n <= max_iters; ++n) {
        error = 0.0;

        if (tl_use_cg){


            tea_leaf_cg_calc_w(pw);

            tea_allsum(pw);
            alpha = rro / pw;

            cg_alphas[n] = alpha;

            tea_leaf_cg_calc_ur(alpha, rrn);

            tea_allsum(rrn);

            beta = rrn / rro;
            cg_betas[n] = beta;

            tea_leaf_cg_calc_p(beta);

            rro = rrn;
            error = rrn;
        }
        else {// jacobi by default
            // appel du wrapper de gestion des tiles
            tea_leaf_jacobi_solve(error);

            // somme globale (MPI) de l'erreur
            tea_allsum(error);
        }
        
        

        // update halos APRES le calcul (ligne 331 fortran)
        fields.assign(NUM_FIELDS, 0);
        fields[FIELD_U] = 1;

        if (tl_use_cg) {
            fields[FIELD_P] = 1;      // FIX: CG strictly requires Search Direction (P) updated
        }
        
        update_halo(fields.data(), 1);

        // Racine carrée de l'erreur pour la convergence (ligne 339 fortran)
        error = std::sqrt(std::abs(error));

        if (parallel.boss && verbose_on) {
            *g_out << "Iteration " << n << " - Residual: " << std::scientific << std::setprecision(6) << error << std::endl;
        }

        //  Convergence Test
        // Stop if the relative error is smaller than the requested epsilon
        if (error < eps * initial_residual) break;
    }

    // Wrap up: Copy results back to main fields
    tea_leaf_finalise();

    // Final Report
    if (parallel.boss) {
        *g_out << "Conduction error " << std::scientific << std::setprecision(7) << (error / initial_residual) << std::endl;
        *g_out << "Iteration count " << std::setw(8) << (n > max_iters ? max_iters : n) << std::endl;
    }
}

} // namespace TeaLeaf