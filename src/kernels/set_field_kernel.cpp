#include "set_field_kernel.h"
#include <omp.h>

void set_field_kernel(
    int x_min, int x_max, 
    int y_min, int y_max, 
    int halo_exchange_depth,
    double* energy0, 
    double* energy1) 
{
    // La largeur totale du tableau inclut les cellules réelles + les halos des deux côtés
    int width = (x_max - x_min + 1) + 2 * halo_exchange_depth;

    // L'offset pour j=x_min est halo_exchange_depth. 
    // Donc l'index 0 du pointeur correspond à (x_min - halo_depth)
    #define IDX(j, k) ((k - (y_min - halo_exchange_depth)) * width + (j - (x_min - halo_exchange_depth)))

    #pragma omp parallel for collapse(2)
    for (int k = y_min; k <= y_max; ++k) {
        for (int j = x_min; j <= x_max; ++j) {
            int index = IDX(j, k);
            energy1[index] = energy0[index];
        }
    }

    #undef IDX
}