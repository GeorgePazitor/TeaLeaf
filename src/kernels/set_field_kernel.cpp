#include "set_field_kernel.h"
#include <omp.h>

void set_field_kernel(
    int x_min, int x_max, 
    int y_min, int y_max, 
    int halo_exchange_depth,
    double* energy0, 
    double* energy1) 
{
    // 1. Calculate Stride
    // The width includes the internal cells plus the full halo depth on both sides.
    int field_width = (x_max - x_min + 1) + 2 * halo_exchange_depth;

    // 2. Index Macro
    // Memory starts at index corresponding to (x_min - halo_depth).
    // So, Offset = (y - y_start) * width + (x - x_start)
    // where x_start = x_min - halo_exchange_depth
    #define IDX(x, y) ((y - (y_min - halo_exchange_depth)) * field_width + (x - (x_min - halo_exchange_depth)))

    // 3. Loop over Internal Domain
    // Fortran: DO k=y_min,y_max; DO j=x_min,x_max
    // We maintain this range to copy only the "Real" cells, not the halo.
    
    #pragma omp for collapse(2)
    for (int k = y_min; k <= y_max; ++k) {
        for (int j = x_min; j <= x_max; ++j) {
            int index = IDX(j, k);
            energy1[index] = energy0[index];
        }
    }

    #undef IDX
}