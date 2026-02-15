#include "include/kernels/field_summary_kernel.h"
#include <omp.h>

/**
 * computes a summary of the physical state of the chunk.
 * Calculates total volume, mass, internal energy, and a mass-weighted temperature equivalent.
 * Necessary to verifying conservation laws during the simulation.
 */
void field_summary_kernel(
    int x_min, int x_max, 
    int y_min, int y_max, 
    int halo_exchange_depth,
    double* volume, 
    double* density, 
    double* energy1, 
    double* u, 
    double& vol, 
    double& mass, 
    double& ie, 
    double& temp) 
{
    int x_off_geo = x_min - 2;
    int y_off_geo = y_min - 2;
    int width_geo = (x_max + 2) - (x_min - 2) + 1;

    int x_off_field = x_min - halo_exchange_depth;
    int y_off_field = y_min - halo_exchange_depth;
    int width_field = (x_max + halo_exchange_depth) - (x_min - halo_exchange_depth) + 1;

    #define IDX_GEO(x, y)   ((y - y_off_geo)   * width_geo   + (x - x_off_geo))
    #define IDX_FIELD(x, y) ((y - y_off_field) * width_field + (x - x_off_field))

    //for thread-safe reduction.
    double local_vol  = 0.0;
    double local_mass = 0.0;
    double local_ie   = 0.0;
    double local_temp = 0.0;

    //parallel reduction with collapse for better thread balancing.
    #pragma omp parallel for collapse(2) reduction(+:local_vol, local_mass, local_ie, local_temp)
    for (int k = y_min; k <= y_max; ++k) {
        for (int j = x_min; j <= x_max; ++j) {
            
            double cell_vol = volume[IDX_GEO(j, k)];
            double dens     = density[IDX_FIELD(j, k)];
            double ener     = energy1[IDX_FIELD(j, k)];
            double u_val    = u[IDX_FIELD(j, k)];

            double cell_mass = cell_vol * dens;

            local_vol  += cell_vol;
            local_mass += cell_mass;
            local_ie   += cell_mass * ener;  //total internal energy
            local_temp += cell_mass * u_val; //mass-weighted temperature proxy
        }
    }

    //outputs the local chunk totals to the reference variables.
    vol  = local_vol;
    mass = local_mass;
    ie   = local_ie;
    temp = local_temp;

    #undef IDX_GEO
    #undef IDX_FIELD
}