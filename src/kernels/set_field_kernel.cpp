#include "include/kernels/set_field_kernel.h"
#include <omp.h>

/**
 * synchronize the energy fields by copying the content of energy0 to energy1,
 * typically called at the beginning of a time step or during, 
 * ensure the 'current' and 'previous' states are consistent
 */
void set_field_kernel(
    int x_min, int x_max, 
    int y_min, int y_max, 
    int halo_exchange_depth,
    double* energy0, 
    double* energy1) 
{
    int width = (x_max - x_min + 1) + 2 * halo_exchange_depth;

    #define IDX(j, k) ((k - (y_min - halo_exchange_depth)) * width + (j - (x_min - halo_exchange_depth)))

    //iterate over the allocated region including halos to maintain data consistency across the full memory block.
    #pragma omp parallel for collapse(2)
    for (int k = y_min - halo_exchange_depth; k <= y_max + halo_exchange_depth; ++k) {
        for (int j = x_min - halo_exchange_depth; j <= x_max + halo_exchange_depth; ++j) {
            int index = IDX(j, k);
            energy1[index] = energy0[index];
        }
    }

    #undef IDX
}