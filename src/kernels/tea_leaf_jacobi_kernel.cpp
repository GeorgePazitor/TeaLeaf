#include <cmath>
#include <vector>
#include "include/data.h"

namespace TeaLeaf {

/**
 * Performs a single Jacobi iteration for the heat equation solver.
 * This kernel updates the temperature field based on a 5-point stencil
 * and computes the local L1 error for convergence checking.
 */
void tea_leaf_jacobi_solve_kernel(
    int x_min, int x_max, int y_min, int y_max,
    int halo_depth, double rx, double ry,
    const std::vector<double>& Kx,
    const std::vector<double>& Ky,
    double& error,
    const std::vector<double>& u0,
    std::vector<double>& u1,
    std::vector<double>& un) 
{
    // Calculate stride for 2D to 1D mapping including ghost cells
    const int x_width = (x_max - x_min + 1) + 2 * halo_depth;
    
    // Indexing macro to handle local tile coordinates with halo offsets
    #define IDX(j, k) (((k) - y_min + halo_depth) * x_width + ((j) - x_min + halo_depth))

    double local_error = 0.0;

    // State Backup
    // Store the current solution (u1) into 'un' to use as the source 
    // for the next iteration. This ensures the Jacobi method stays stationary.
    for (int k = y_min; k <= y_max; ++k) {
        for (int j = x_min; j <= x_max; ++j) {
            un[IDX(j, k)] = u1[IDX(j, k)];
        }
    }

    // Jacobi Iteration Calculation
    // Update the solution using the stencil: 
    // u1 = [u0 + Σ(Conductivity * Neighbor_Temperature)] / [1 + Σ(Conductivity)]
    for (int k = y_min; k <= y_max; ++k) {
        for (int j = x_min; j <= x_max; ++j) {
            
            // Calculate numerator: Initial state plus the flux from 4 neighbors (West, East, South, North)
            double numerator = u0[IDX(j, k)] 
                + rx * (Kx[IDX(j + 1, k)] * un[IDX(j + 1, k)] + Kx[IDX(j, k)] * un[IDX(j - 1, k)])
                + ry * (Ky[IDX(j, k + 1)] * un[IDX(j, k + 1)] + Ky[IDX(j, k)] * un[IDX(j, k - 1)]);

            // Calculate denominator: The central coefficient (diagonal of the matrix A)
            double denominator = 1.0 
                + rx * (Kx[IDX(j, k)] + Kx[IDX(j + 1, k)])
                + ry * (Ky[IDX(j, k)] + Ky[IDX(j, k + 1)]);

            // Compute new temperature value
            u1[IDX(j, k)] = numerator / denominator;

            // Compute L1 norm error: The absolute difference between this and the previous iteration
            local_error += std::abs(u1[IDX(j, k)] - un[IDX(j, k)]);
        }
    }

    // Return the accumulated tile error to the calling driver for global MPI reduction
    error = local_error;
    
    #undef IDX
}
} // namespace TeaLeaf