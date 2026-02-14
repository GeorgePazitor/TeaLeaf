#include <cmath>
#include <vector>
#include <omp.h>
#include "include/kernels/tea_leaf_jacobi_kernel.h"

namespace TeaLeaf {

/**
 * Orchestrates the Jacobi solver across all tiles in the current MPI rank.
 * Uses OpenMP to process tiles in parallel.
 */
void tea_leaf_jacobi_solve_kernel(
    int x_min, int x_max, int y_min, int y_max,
    int halo, double rx, double ry,
    const std::vector<double>& Kx_v,
    const std::vector<double>& Ky_v,
    double& error,
    const std::vector<double>& u0_v,
    std::vector<double>& u1_v,
    std::vector<double>& un_v) 
{
    const int x_inc = (x_max - x_min + 1) + 2 * halo;
    double local_error = 0.0;

    // Obtain raw pointers for better performance
    const double* __restrict u0 = u0_v.data();
    double* __restrict u1 = u1_v.data();
    double* __restrict un = un_v.data();
    const double* __restrict Kx = Kx_v.data();
    const double* __restrict Ky = Ky_v.data();

    #define IDX(j, k) (((k) - y_min + halo) * x_inc + ((j) - x_min + halo))

    // Single parallel region to reduce fork/join overhead
    #pragma omp parallel
    {
        // Backup current state into un array
        #pragma omp for
        for (int k = y_min; k <= y_max; ++k) {
            #pragma omp simd
            for (int j = x_min; j <= x_max; ++j) {
                un[IDX(j, k)] = u1[IDX(j, k)];
            }
        }

        // Perform Jacobi update and accumulate local error
        #pragma omp for reduction(+:local_error)
        for (int k = y_min; k <= y_max; ++k) {
            #pragma omp simd
            for (int j = x_min; j <= x_max; ++j) {
                int idx = IDX(j, k);
                
                // Compute numerator: weighted sum of neighboring fluxes
                double num = u0[idx] 
                    + rx * (Kx[IDX(j+1, k)] * un[IDX(j+1, k)] + Kx[idx] * un[IDX(j-1, k)])
                    + ry * (Ky[IDX(j, k+1)] * un[IDX(j, k+1)] + Ky[idx] * un[IDX(j, k-1)]);

                // Compute denominator: sum of diagonal coefficients
                double den = 1.0 
                    + rx * (Kx[idx] + Kx[IDX(j+1, k)])
                    + ry * (Ky[idx] + Ky[IDX(j, k+1)]);

                // Update solution and accumulate error
                u1[idx] = num / den;
                local_error += std::abs(u1[idx] - un[idx]);
            }
        }
    }

    error = local_error;

    #undef IDX
}

} // namespace TeaLeaf
