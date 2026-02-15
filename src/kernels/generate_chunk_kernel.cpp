#include "include/kernels/generate_chunk_kernel.h"
#include <omp.h>
#include <cmath>

/**
 * Initializes the physical fields (density, energy) based on predefined states.
 * States are applied with overlapping logic:
 *  state 1 is the first applied in the whole domain (it is the background) 
 *  states 2 to N overwrite the background if the cell falls within their geometry
 */
void generate_chunk_kernel(
    int x_min, int x_max, int y_min, int y_max, int halo_depth,
    double* vertexx, double* vertexy, double* cellx, double* celly,
    double* density, double* energy0, double* u0,
    int number_of_states,
    double* state_density, double* state_energy, 
    double* state_xmin, double* state_xmax, 
    double* state_ymin, double* state_ymax, 
    double* state_radius, int* state_geometry, 
    int g_rect, int g_circ, int g_point) 
{
    int width = (x_max + halo_depth) - (x_min - halo_depth) + 1;

    //row-major mapping of field arrays: density, energy, ...
    #define FIELD_IDX(j, k) ((k - (y_min - halo_depth)) * width + (j - (x_min - halo_depth)))
    
    //mapping for coordinate arrays, vertex,cell, accounting for the 2 padding offset.
    #define V_IDX(p, p_min) (p - (p_min - 2)) 

    //set the background everywhere 
    #pragma omp parallel for collapse(2)
    for (int k = y_min - halo_depth; k <= y_max + halo_depth; ++k) {
        for (int j = x_min - halo_depth; j <= x_max + halo_depth; ++j) {
            int idx = FIELD_IDX(j, k);
            density[idx] = state_density[1];
            energy0[idx] = state_energy[1];
        }
    }
    //set the 2 to N subsequent states one on top of the other (the last has higher priority if overlapping)
    for (int s = 2; s <= number_of_states; ++s) {
        #pragma omp parallel for collapse(2)
        for (int k = y_min - halo_depth; k <= y_max + halo_depth; ++k) {
            for (int j = x_min - halo_depth; j <= x_max + halo_depth; ++j) {
                
                bool apply = false;
                //rectangulare geom
                if (state_geometry[s] == g_rect) {
                    if (j >= x_min && j <= x_max && k >= y_min && k <= y_max) {
                        int j_idx = V_IDX(j, x_min);
                        int k_idx = V_IDX(k, y_min);
                        
                        if (vertexx[j_idx + 1] >= state_xmin[s] && vertexx[j_idx] < state_xmax[s] &&
                            vertexy[k_idx + 1] >= state_ymin[s] && vertexy[k_idx] < state_ymax[s]) {
                            apply = true;
                        }
                    }
                } 
                //circular geom
                else if (state_geometry[s] == g_circ) {
                    int j_idx = V_IDX(j, x_min);
                    int k_idx = V_IDX(k, y_min);
                    double dx = cellx[j_idx] - state_xmin[s]; // xmin used as center_x
                    double dy = celly[k_idx] - state_ymin[s]; // ymin used as center_y
                    if (std::sqrt(dx*dx + dy*dy) <= state_radius[s]) {
                        apply = true;
                    }
                }
                //point geom
                else if (state_geometry[s] == g_point) {
                    int j_idx = V_IDX(j, x_min);
                    int k_idx = V_IDX(k, y_min);
                    if (vertexx[j_idx] == state_xmin[s] && vertexy[k_idx] == state_ymin[s]) {
                        apply = true;
                    }
                }

                if (apply) {
                    int idx = FIELD_IDX(j, k);
                    density[idx] = state_density[s];
                    energy0[idx] = state_energy[s];
                }
            }
        }
    }

    // u0 = density * energy0 ( physical quantity actually used by the solverm ).
    #pragma omp parallel for collapse(2)
    for (int k = y_min - halo_depth; k <= y_max + halo_depth; ++k) {
        for (int j = x_min - halo_depth; j <= x_max + halo_depth; ++j) {
            int idx = FIELD_IDX(j, k);
            u0[idx] = density[idx] * energy0[idx];
        }
    }

    #undef FIELD_IDX
    #undef V_IDX
}