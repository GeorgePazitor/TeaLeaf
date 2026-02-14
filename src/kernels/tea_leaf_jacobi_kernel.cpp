#include <cmath>
#include <vector>
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
    const int x_width = (x_max - x_min + 1) + 2 * halo_depth;
    
    // Macro d'indexation
    #define IDX(j, k) (((k) - y_min + halo_depth) * x_width + ((j) - x_min + halo_depth))

    double local_error = 0.0;

    // Étape 1 : un = u (Sauvegarde de l'itération précédente)
    // Pas de pragma ici : le thread traite toute sa tuile
    for (int k = y_min; k <= y_max; ++k) {
        for (int j = x_min; j <= x_max; ++j) {
            un[IDX(j, k)] = u1[IDX(j, k)];
        }
    }

    // Étape 2 : Calcul Jacobi
    // On retire le pragma pour éviter l'erreur de réduction "private in outer context"
    for (int k = y_min; k <= y_max; ++k) {
        for (int j = x_min; j <= x_max; ++j) {
            
            // Calcul du numérateur (flux avec les voisins)
            double numerator = u0[IDX(j, k)] 
                + rx * (Kx[IDX(j + 1, k)] * un[IDX(j + 1, k)] + Kx[IDX(j, k)] * un[IDX(j - 1, k)])
                + ry * (Ky[IDX(j, k + 1)] * un[IDX(j, k + 1)] + Ky[IDX(j, k)] * un[IDX(j, k - 1)]);

            // Calcul du dénominateur (diagonale de la matrice)
            double denominator = 1.0 
                + rx * (Kx[IDX(j, k)] + Kx[IDX(j + 1, k)])
                + ry * (Ky[IDX(j, k)] + Ky[IDX(j, k + 1)]);

            u1[IDX(j, k)] = numerator / denominator;

            // Cumul de l'erreur L1 pour cette tuile
            local_error += std::abs(u1[IDX(j, k)] - un[IDX(j, k)]);
        }
    }

    // On stocke le résultat dans la référence passée par le driver
    error = local_error;
    
    #undef IDX
}
}