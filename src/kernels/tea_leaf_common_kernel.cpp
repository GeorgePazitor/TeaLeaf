#include "include/kernels/tea_leaf_common_kernel.h"
#include "include/definitions.h"
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>

namespace TeaLeaf {

#define IDX(j, k) (((k) - y_min + halo) * x_inc + ((j) - x_min + halo))

// --- DIAG PRECONDITIONER (JACOBI) ---
/**
 * Initializes the diagonal preconditioner (Mi), 
 * this is effectively a point-Jacobi step where Mi = 1/Diag(A).
 */
void tea_diag_init(int x_min, int x_max, int y_min, int y_max, int halo,
                   std::vector<double>& Mi, const std::vector<double>& Di) {
    const int x_inc = (x_max - x_min + 1) + 2 * halo;
    const double omega = 1.0; // Relaxation factor
    for (int k = y_min; k <= y_max; ++k) {
        for (int j = x_min; j <= x_max; ++j) {
            int idx = IDX(j, k);
            Mi[idx] = (Di[idx] != 0.0) ? (omega / Di[idx]) : 0.0;
        }
    }
}

// --- BLOCK PRECONDITIONER (THOMAS ALGORITHM) ---
/**
 * Sets up the tridiagonal matrix decomposition for block preconditioning,
 * uses a simplified Thomas Algorithm (forward elimination) to store
 * coefficients for solving a vertical tridiagonal strip.
 */
void tea_block_init(int x_min, int x_max, int y_min, int y_max, int halo,
                    std::vector<double>& cp, std::vector<double>& bfp,
                    const std::vector<double>& Ky, const std::vector<double>& Di,
                    double ry) {
    const int x_inc = (x_max - x_min + 1) + 2 * halo;
    const int jac_block_size = 4; // Sub-division of the block size

    for (int ko = y_min; ko <= y_max; ko += jac_block_size) {
        int bottom = ko;
        int top = std::min(ko + jac_block_size - 1, y_max);

        for (int j = x_min; j <= x_max; ++j) {
            int idx_b = IDX(j, bottom);
            cp[idx_b] = (-Ky[IDX(j, bottom + 1)] * ry) / Di[idx_b];

            for (int k = bottom + 1; k <= top; ++k) {
                int idx = IDX(j, k);
                // bfp stores the reciprocal of the modified diagonal
                bfp[idx] = 1.0 / (Di[idx] - (-Ky[idx] * ry) * cp[IDX(j, k - 1)]);
                // cp stores the modified upper off-diagonal
                cp[idx] = (-Ky[IDX(j, k + 1)] * ry) * bfp[idx];
            }
        }
    }
}

// --- MAIN INITIALIZATION KERNEL ---
/**
 * The core setup kernel. It:
 * 1 computes initial temperature (u) and conductivity (w).
 * 2 calculates face-centered conductivities (Kx, Ky) using harmonic means.
 * 3 sets boundary conditions (dirichlet vs reflective).
 * 4 forms the matrix diagonal (Di) and the initial residual (r).
 */
void tea_leaf_common_init_kernel(
    int x_min, int x_max, int y_min, int y_max, int halo,
    const std::array<bool, 4>& zero_boundary, bool reflective_boundary,
    const std::vector<double>& density, const std::vector<double>& energy,
    std::vector<double>& u, std::vector<double>& u0,
    std::vector<double>& r, std::vector<double>& w,
    std::vector<double>& Kx, std::vector<double>& Ky,
    std::vector<double>& Di, std::vector<double>& cp,
    std::vector<double>& bfp, std::vector<double>& Mi,
    double rx, double ry, int preconditioner_type, int coef) 
{
    const int x_inc = (x_max - x_min + 1) + 2 * halo;

    //1  Physical Field Initialization
    
    for (int k = y_min - halo; k <= y_max + halo; ++k) {
        for (int j = x_min - halo; j <= x_max + halo; ++j) {
            int idx = IDX(j, k);
            u[idx] = energy[idx] * density[idx]; //temperature/energy relation
            u0[idx] = u[idx];
            w[idx] = (coef == RECIP_CONDUCTIVITY) ? (1.0 / density[idx]) : density[idx];
        }
    }

    //2 compute face conductivities (harmonic mean)
    // K values are defined at cell faces between neighbors.
    for (int k = y_min - halo + 1; k <= y_max + halo; ++k) {
        for (int j = x_min - halo + 1; j <= x_max + halo; ++j) {
            double w1_x = w[IDX(j-1, k)];
            double w2_x = w[IDX(j, k)];
            Kx[IDX(j, k)] = (w1_x + w2_x) / (2.0 * w1_x * w2_x);

            double w1_y = w[IDX(j, k-1)];
            double w2_y = w[IDX(j, k)];
            Ky[IDX(j, k)] = (w1_y + w2_y) / (2.0 * w1_y * w2_y);
        }
    }

    //3 applies external boundary conditions
    // if not reflective (dirichlet) zero out the transmission coefficients at the boundary.
    if (!reflective_boundary) {
        if (zero_boundary[0]) for(int k=y_min-halo; k<=y_max+halo; ++k) Kx[IDX(x_min, k)] = 0.0;
        if (zero_boundary[1]) for(int k=y_min-halo; k<=y_max+halo; ++k) Kx[IDX(x_max+1, k)] = 0.0;
        if (zero_boundary[2]) for(int j=x_min-halo; j<=x_max+halo; ++j) Ky[IDX(j, y_min)] = 0.0;
        if (zero_boundary[3]) for(int j=x_min-halo; j<=x_max+halo; ++j) Ky[IDX(j, y_max+1)] = 0.0;
    }

    //4 matrix main diagonal construction (Di)
    for (int k = y_min; k <= y_max; ++k) {
        for (int j = x_min; j <= x_max; ++j) {
            int idx = IDX(j, k);
            Di[idx] = 1.0 + rx * (Kx[IDX(j+1, k)] + Kx[IDX(j, k)])
                          + ry * (Ky[IDX(j, k+1)] + Ky[IDX(j, k)]);
        }
    }

    //5 setup preconditioners (Optional depending on solver choice)
    if (preconditioner_type == TL_PREC_JAC_DIAG) {
        tea_diag_init(x_min, x_max, y_min, y_max, halo, Mi, Di);
    } else if (preconditioner_type == TL_PREC_JAC_BLOCK) {
        tea_block_init(x_min, x_max, y_min, y_max, halo, cp, bfp, Ky, Di, ry);
    }

    // 6. initial residual calculation: r = b - Au
    for (int k = y_min; k <= y_max; ++k) {
        for (int j = x_min; j <= x_max; ++j) {
            int idx = IDX(j, k);
            double smvp = Di[idx] * u[idx]
                - ry * (Ky[IDX(j, k+1)] * u[IDX(j, k+1)] + Ky[IDX(j, k)] * u[IDX(j, k-1)])
                - rx * (Kx[IDX(j+1, k)] * u[IDX(j+1, k)] + Kx[IDX(j, k)] * u[IDX(j-1, k)]);
            r[idx] = u0[idx] - smvp;
        }
    }
}

/**
 * calculates the current residual (r = b - Au).
 * used at the end of solver iterations to determine convergence.
 */
void tea_leaf_calc_residual_kernel(
    int x_min, int x_max, int y_min, int y_max, int halo,
    const std::vector<double>& u, const std::vector<double>& u0,
    std::vector<double>& r, const std::vector<double>& Kx,
    const std::vector<double>& Ky, const std::vector<double>& Di,
    double rx, double ry) 
{
    const int x_inc = (x_max - x_min + 1) + 2 * halo;
    for (int k = y_min; k <= y_max; ++k) {
        for (int j = x_min; j <= x_max; ++j) {
            int idx = IDX(j, k);
            double smvp = Di[idx] * u[idx]
                        - ry * (Ky[IDX(j, k + 1)] * u[IDX(j, k + 1)] + Ky[IDX(j, k)] * u[IDX(j, k - 1)])
                        - rx * (Kx[IDX(j + 1, k)] * u[IDX(j + 1, k)] + Kx[IDX(j, k)] * u[IDX(j - 1, k)]);
            r[idx] = u0[idx] - smvp;
        }
    }
}

/**
 * Computes the L2 norm (sum of squares) of a vector,
 * the result is reduced locally before being summed globally via MPI elsewhere.
 */
void tea_leaf_calc_2norm_kernel(
    int x_min, int x_max, int y_min, int y_max, int halo,
    const std::vector<double>& arr, double& norm) 
{
    const int x_inc = (x_max - x_min + 1) + 2 * halo;
    double local_norm = 0.0;
    for (int k = y_min; k <= y_max; ++k) {
        for (int j = x_min; j <= x_max; ++j) {
            double val = arr[IDX(j, k)];
            local_norm += val * val;
        }
    }
    norm = local_norm;
}

/**
 * Converts the solved state (u) back into physical energy values 
 * by dividing by density.
 */
void tea_leaf_kernel_finalise(
    int x_min, int x_max, int y_min, int y_max, int halo,
    std::vector<double>& energy, const std::vector<double>& density,
    const std::vector<double>& u) 
{
    const int x_inc = (x_max - x_min + 1) + 2 * halo;
    for (int k = y_min; k <= y_max; ++k) {
        for (int j = x_min; j <= x_max; ++j) {
            int idx = IDX(j, k);
            energy[idx] = u[idx] / density[idx];
        }
    }
}

#undef IDX

} // namespace TeaLeaf