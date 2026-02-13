#include "kernels/tea_leaf_common_kernel.h"
#include "definitions.h"
#include <cmath>
#include <vector>
#include <array>

namespace TeaLeaf {

#define IDX(j, k) (((k) - y_min + halo) * x_inc + ((j) - x_min + halo))

// --- KERNEL D'INITIALISATION (Calcul de Kx, Ky, u, u0 et r) ---
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

    // 1. Initialisation de u, u0 et w
    for (int k = y_min - halo; k <= y_max + halo; ++k) {
        for (int j = x_min - halo; j <= x_max + halo; ++j) {
            int idx = IDX(j, k);
            u[idx] = energy[idx] * density[idx];
            u0[idx] = u[idx];
            if (coef == RECIP_CONDUCTIVITY) {
                w[idx] = (density[idx] > 1e-12) ? (1.0 / density[idx]) : 1e12;
            } else {
                w[idx] = density[idx];
            }
        }
    }

    // 2. Kx (Moyenne harmonique pour Test 2)
    for (int k = y_min; k <= y_max; ++k) {
        for (int j = x_min; j <= x_max + 1; ++j) {
            double w1 = w[IDX(j-1, k)];
            double w2 = w[IDX(j, k)];
            if (coef == RECIP_CONDUCTIVITY) {
                Kx[IDX(j, k)] = (w1 + w2) / (2.0 * w1 * w2);
            } else {
                Kx[IDX(j, k)] = (w1 + w2) / 2.0;
            }
        }
    }

    // 3. Ky
    for (int k = y_min; k <= y_max + 1; ++k) {
        for (int j = x_min; j <= x_max; ++j) {
            double w1 = w[IDX(j, k-1)];
            double w2 = w[IDX(j, k)];
            if (coef == RECIP_CONDUCTIVITY) {
                Ky[IDX(j, k)] = (w1 + w2) / (2.0 * w1 * w2);
            } else {
                Ky[IDX(j, k)] = (w1 + w2) / 2.0;
            }
        }
    }

    // 4. CL
    if (zero_boundary[0]) for(int k=y_min; k<=y_max; ++k) Kx[IDX(x_min, k)] = 0.0;
    if (zero_boundary[1]) for(int k=y_min; k<=y_max; ++k) Kx[IDX(x_max+1, k)] = 0.0;
    if (zero_boundary[2]) for(int j=x_min; j<=x_max; ++j) Ky[IDX(j, y_min)] = 0.0;
    if (zero_boundary[3]) for(int j=x_min; j<=x_max; ++j) Ky[IDX(j, y_max+1)] = 0.0;

    // 5. Di et r
    for (int k = y_min; k <= y_max; ++k) {
        for (int j = x_min; j <= x_max; ++j) {
            int idx = IDX(j, k);
            Di[idx] = 1.0 + rx * (Kx[IDX(j+1, k)] + Kx[IDX(j, k)])
                          + ry * (Ky[IDX(j, k+1)] + Ky[IDX(j, k)]);
            
            double Au = Di[idx] * u[idx]
                - rx * (Kx[IDX(j+1, k)] * u[IDX(j+1, k)] + Kx[IDX(j, k)] * u[IDX(j-1, k)])
                - ry * (Ky[IDX(j, k+1)] * u[IDX(j, k+1)] + Ky[IDX(j, k)] * u[IDX(j, k-1)]);
            r[idx] = u0[idx] - Au;
        }
    }
}

// --- KERNEL DU CALCUL DU RÉSIDU (Manquant précédemment) ---
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
            double smvp = Di[IDX(j, k)] * u[IDX(j, k)]
                        - ry * (Ky[IDX(j, k + 1)] * u[IDX(j, k + 1)] + Ky[IDX(j, k)] * u[IDX(j, k - 1)])
                        - rx * (Kx[IDX(j + 1, k)] * u[IDX(j + 1, k)] + Kx[IDX(j, k)] * u[IDX(j - 1, k)]);
            r[IDX(j, k)] = u0[IDX(j, k)] - smvp;
        }
    }
}

// --- KERNEL DE LA NORME L2 (Manquant précédemment) ---
void tea_leaf_calc_2norm_kernel(
    int x_min, int x_max, int y_min, int y_max, int halo,
    const std::vector<double>& arr, double& norm) 
{
    const int x_inc = (x_max - x_min + 1) + 2 * halo;
    double local_norm = 0.0;
    for (int k = y_min; k <= y_max; ++k) {
        for (int j = x_min; j <= x_max; ++j) {
            local_norm += arr[IDX(j, k)] * arr[IDX(j, k)];
        }
    }
    norm = local_norm;
}

// --- KERNEL DE FINALISATION (Manquant précédemment) ---
void tea_leaf_kernel_finalise(
    int x_min, int x_max, int y_min, int y_max, int halo,
    std::vector<double>& energy, const std::vector<double>& density,
    const std::vector<double>& u) 
{
    const int x_inc = (x_max - x_min + 1) + 2 * halo;
    for (int k = y_min; k <= y_max; ++k) {
        for (int j = x_min; j <= x_max; ++j) {
            energy[IDX(j, k)] = u[IDX(j, k)] / density[IDX(j, k)];
        }
    }
}

#undef IDX

} // namespace TeaLeaf