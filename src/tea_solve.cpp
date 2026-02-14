#include "include/tea.h"
#include "include/data.h"
#include "include/global_mpi.h"
#include "include/update_halo.h"
#include "include/definitions.h"
#include "include/tea_leaf_common.h"
#include "include/tea_leaf_jacobi.h"
#include <cmath>
#include <iomanip>

namespace TeaLeaf {

/**
 * Main solver routine for the heat diffusion equation.
 * Currently implements the Jacobi iterative method.
 */
void tea_leaf() {
    int n = 0;
    double error = 0.0, initial_residual = 0.0;
    std::vector<int> fields(NUM_FIELDS, 0);

    // Initial Synchronization
    // Ensure all ranks have up-to-date density and energy1 fields
    fields[FIELD_ENERGY1] = 1;
    fields[FIELD_DENSITY] = 1;
    update_halo(fields.data(), chunk.halo_exchange_depth);
    
    // Prepare coefficients for the linear system Ax = b
    tea_leaf_init_common();

    // Calculate Initial Residual (L2 Norm)
    fields.assign(NUM_FIELDS, 0);
    fields[FIELD_U] = 1;
    update_halo(fields.data(), 1);

    tea_leaf_calc_residual();
    tea_leaf_calc_2norm(1, initial_residual);
    
    // Global MPI reduction to get the sum of residuals across all ranks
    tea_allsum(initial_residual);
    initial_residual = std::sqrt(std::abs(initial_residual));

    if (parallel.boss && verbose_on) {
        *g_out << "Initial residual " << std::scientific << std::setprecision(6) << initial_residual << std::endl;
    }

    // Jacobi Iterative Loop
    for (n = 1; n <= max_iters; ++n) {
        error = 0.0;

        // Perform one Jacobi iteration step on all local tiles (OpenMP)
        tea_leaf_jacobi_solve(error);

        // Global sum of the local error contributions
        tea_allsum(error);

        // Critical: Update halos so neighbors see the changes from this iteration
        fields.assign(NUM_FIELDS, 0);
        fields[FIELD_U] = 1;
        update_halo(fields.data(), 1);

        // L2 Norm of the current error
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