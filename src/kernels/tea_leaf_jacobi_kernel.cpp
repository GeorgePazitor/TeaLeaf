#include <cmath>
#include <vector>
#include <omp.h>
#include "data.h"

namespace TeaLeaf {

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
    // Largeur totale du tableau incluant les halos pour l'indexation
    int x_width = (x_max - x_min + 1) + 2 * halo_depth;
    
    // Macro locale pour simuler l'accès 2D (j,k) en Row-Major
    // L'offset j-x_min+halo_depth permet de gérer les indices commençant à 0
    #define IDX(j, k) ((k - y_min + halo_depth) * x_width + (j - x_min + halo_depth))

    double local_error = 0.0;

    #pragma omp parallel
    {
        // 1. Copie de u1 dans un (ancienne itération)
        #pragma omp for
        for (int k = y_min; k <= y_max; ++k) {
            for (int j = x_min; j <= x_max; ++j) {
                un[IDX(j, k)] = u1[IDX(j, k)];
            }
        }

        // 2. Calcul du stencil Jacobi et de l'erreur
        #pragma omp for reduction(+:local_error)
        for (int k = y_min; k <= y_max; ++k) {
            for (int j = x_min; j <= x_max; ++j) {
                
                double numerator = u0[IDX(j, k)] 
                    + rx * (Kx[IDX(j + 1, k)] * un[IDX(j + 1, k)] + Kx[IDX(j, k)] * un[IDX(j - 1, k)])
                    + ry * (Ky[IDX(j, k + 1)] * un[IDX(j, k + 1)] + Ky[IDX(j, k)] * un[IDX(j, k - 1)]);

                double denominator = 1.0 
                    + rx * (Kx[IDX(j, k)] + Kx[IDX(j + 1, k)])
                    + ry * (Ky[IDX(j, k)] + Ky[IDX(j, k + 1)]);

                u1[IDX(j, k)] = numerator / denominator;

                // Somme des différences absolues pour la convergence
                local_error += std::abs(u1[IDX(j, k)] - un[IDX(j, k)]);
            }
        }
    }

    error = local_error;

    #undef IDX
}

} // namespace TeaLeaf