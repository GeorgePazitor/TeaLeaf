#include "tea_leaf_common_kernel.h"
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace TeaLeaf {

// Macro pour l'indexation 2D (j=x, k=y)
#define IDX(j, k) ((k - y_min + halo) * x_inc + (j - x_min + halo))

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

    #pragma omp parallel
    {
        // 1. Initialisation de u et u0
        #pragma omp for
        for (int k = y_min; k <= y_max; ++k) {
            for (int j = x_min; j <= x_max; ++j) {
                u[IDX(j, k)] = energy[IDX(j, k)] * density[IDX(j, k)];
                u0[IDX(j, k)] = u[IDX(j, k)];
            }
        }

        // 2. REUSE de w pour stocker la conductivité (w = 1/rho ou rho)
        #pragma omp for
        for (int k = y_min - halo; k <= y_max + halo; ++k) {
            for (int j = x_min - halo; j <= x_max + halo; ++j) {
                if (coef == RECIP_CONDUCTIVITY)
                    w[IDX(j, k)] = 1.0 / density[IDX(j, k)];
                else
                    w[IDX(j, k)] = density[IDX(j, k)];
            }
        }

        // 3. Calcul de Kx et Ky (Moyenne harmonique)
        #pragma omp for
        for (int k = y_min - halo + 1; k <= y_max + halo; ++k) {
            for (int j = x_min - halo + 1; j <= x_max + halo; ++j) {
                Kx[IDX(j, k)] = (w[IDX(j - 1, k)] + w[IDX(j, k)]) / (2.0 * w[IDX(j - 1, k)] * w[IDX(j, k)]);
                Ky[IDX(j, k)] = (w[IDX(j, k - 1)] + w[IDX(j, k)]) / (2.0 * w[IDX(j, k - 1)] * w[IDX(j, k)]);
            }
        }

        // 4. Conditions aux limites Zero Boundary (Utilise les constantes de data.h)
        if (!reflective_boundary) {
            if (zero_boundary[CHUNK_LEFT]) {
                #pragma omp for
                for (int k = y_min - halo; k <= y_max + halo; ++k)
                    for (int j = x_min - halo; j <= x_min; ++j) Kx[IDX(j, k)] = 0.0;
            }
            if (zero_boundary[CHUNK_RIGHT]) {
                #pragma omp for
                for (int k = y_min - halo; k <= y_max + halo; ++k)
                    for (int j = x_max + 1; j <= x_max + halo; ++j) Kx[IDX(j, k)] = 0.0;
            }
            if (zero_boundary[CHUNK_BOTTOM]) {
                #pragma omp for
                for (int k = y_min - halo; k <= y_min; ++k)
                    for (int j = x_min - halo; j <= x_max + halo; ++j) Ky[IDX(j, k)] = 0.0;
            }
            if (zero_boundary[CHUNK_TOP]) {
                #pragma omp for
                for (int k = y_max + 1; k <= y_max + halo; ++k)
                    for (int j = x_min - halo; j <= x_max + halo; ++j) Ky[IDX(j, k)] = 0.0;
            }
        }

        // 5. Calcul de la Diagonale (Di)
        #pragma omp for
        for (int k = y_min - halo + 1; k <= y_max + halo - 1; ++k) {
            for (int j = x_min - halo + 1; j <= x_max + halo - 1; ++j) {
                Di[IDX(j, k)] = 1.0 + ry * (Ky[IDX(j, k + 1)] + Ky[IDX(j, k)])
                                    + rx * (Kx[IDX(j + 1, k)] + Kx[IDX(j, k)]);
            }
        }
    } // Fin bloc parallèle

    // Initialisation du préconditionneur
    if (preconditioner_type == TL_PREC_JAC_DIAG) {
        tea_diag_init(x_min, x_max, y_min, y_max, halo, Mi, Kx, Ky, Di, rx, ry);
    }

    // 6. REUSE FINAL de w : Calcul du résidu initial (w = A * u)
    #pragma omp parallel for
    for (int k = y_min; k <= y_max; ++k) {
        for (int j = x_min; j <= x_max; ++j) {
            w[IDX(j, k)] = Di[IDX(j, k)] * u[IDX(j, k)]
                         - ry * (Ky[IDX(j, k + 1)] * u[IDX(j, k + 1)] + Ky[IDX(j, k)] * u[IDX(j, k - 1)])
                         - rx * (Kx[IDX(j + 1, k)] * u[IDX(j + 1, k)] + Kx[IDX(j, k)] * u[IDX(j - 1, k)]);
            
            r[IDX(j, k)] = u[IDX(j, k)] - w[IDX(j, k)];
        }
    }
}

void tea_leaf_calc_residual_kernel(
    int x_min, int x_max, int y_min, int y_max, int halo,
    const std::vector<double>& u, const std::vector<double>& u0,
    std::vector<double>& r, const std::vector<double>& Kx,
    const std::vector<double>& Ky, const std::vector<double>& Di,
    double rx, double ry) 
{
    const int x_inc = (x_max - x_min + 1) + 2 * halo;
    #pragma omp parallel for
    for (int k = y_min; k <= y_max; ++k) {
        for (int j = x_min; j <= x_max; ++j) {
            double smvp = Di[IDX(j, k)] * u[IDX(j, k)]
                        - ry * (Ky[IDX(j, k + 1)] * u[IDX(j, k + 1)] + Ky[IDX(j, k)] * u[IDX(j, k - 1)])
                        - rx * (Kx[IDX(j + 1, k)] * u[IDX(j + 1, k)] + Kx[IDX(j, k)] * u[IDX(j - 1, k)]);
            r[IDX(j, k)] = u0[IDX(j, k)] - smvp;
        }
    }
}

void tea_leaf_calc_2norm_kernel(
    int x_min, int x_max, int y_min, int y_max, int halo,
    const std::vector<double>& arr, double& norm) 
{
    const int x_inc = (x_max - x_min + 1) + 2 * halo;
    double local_norm = 0.0;
    #pragma omp parallel for reduction(+:local_norm)
    for (int k = y_min; k <= y_max; ++k) {
        for (int j = x_min; j <= x_max; ++j) {
            local_norm += arr[IDX(j, k)] * arr[IDX(j, k)];
        }
    }
    norm = local_norm;
}

void tea_leaf_kernel_finalise(
    int x_min, int x_max, int y_min, int y_max, int halo,
    std::vector<double>& energy, const std::vector<double>& density,
    const std::vector<double>& u) 
{
    const int x_inc = (x_max - x_min + 1) + 2 * halo;
    #pragma omp parallel for
    for (int k = y_min; k <= y_max; ++k) {
        for (int j = x_min; j <= x_max; ++j) {
            energy[IDX(j, k)] = u[IDX(j, k)] / density[IDX(j, k)];
        }
    }
}

void tea_diag_init(
    int x_min, int x_max, int y_min, int y_max, int halo,
    std::vector<double>& Mi, const std::vector<double>& Kx, 
    const std::vector<double>& Ky, const std::vector<double>& Di, 
    double rx, double ry) 
{
    const int x_inc = (x_max - x_min + 1) + 2 * halo;
    const double omega = 1.0;
    #pragma omp parallel for
    for (int k = y_min - halo + 1; k <= y_max + halo - 1; ++k) {
        for (int j = x_min - halo + 1; j <= x_max + halo - 1; ++j) {
            int idx = IDX(j, k);
            Mi[idx] = (Di[idx] != 0.0) ? (omega / Di[idx]) : 0.0;
        }
    }
}

} // namespace TeaLeaf