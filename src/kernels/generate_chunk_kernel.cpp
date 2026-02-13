#include "generate_chunk_kernel.h"
#include <omp.h>
#include <cmath>

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

    #define FIELD_IDX(j, k) ((k - (y_min - halo_depth)) * width + (j - (x_min - halo_depth)))
    #define V_IDX(p, p_min) (p - (p_min - halo_depth))

    // 1. Remplissage du Background (State 1) PARTOUT
    #pragma omp parallel for collapse(2)
    for (int k = y_min - halo_depth; k <= y_max + halo_depth; ++k) {
        for (int j = x_min - halo_depth; j <= x_max + halo_depth; ++j) {
            int idx = FIELD_IDX(j, k);
            density[idx] = state_density[1];
            energy0[idx] = state_energy[1];
        }
    }

    // 2. Application des Overlays (States 2 à N)
    for (int s = 2; s <= number_of_states; ++s) {
        #pragma omp parallel for collapse(2)
        for (int k = y_min; k <= y_max; ++k) {
            for (int j = x_min; j <= x_max; ++j) {
                int j_idx = V_IDX(j, x_min);
                int k_idx = V_IDX(k, y_min);
                bool apply = false;

                if (state_geometry[s] == g_rect) {
                    if (cellx[j_idx] >= state_xmin[s] && cellx[j_idx] <= state_xmax[s] &&
                        celly[k_idx] >= state_ymin[s] && celly[k_idx] <= state_ymax[s]) {
                        apply = true;
                    }
                } 
                else if (state_geometry[s] == g_circ) {
                    double dx = cellx[j_idx] - state_xmin[s];
                    double dy = celly[k_idx] - state_ymin[s];
                    if (std::sqrt(dx*dx + dy*dy) <= state_radius[s]) apply = true;
                }

                if (apply) {
                    int idx = FIELD_IDX(j, k);
                    density[idx] = state_density[s];
                    energy0[idx] = state_energy[s];
                }
            }
        }
    }

    // 3. Calcul de u0
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