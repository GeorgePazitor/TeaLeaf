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
    // Calcul de la largeur avec halo pour l'indexation du champ 2D à plat
    int width = (x_max + halo_depth) - (x_min - halo_depth) + 1;

    // Macro pour l'indexation des champs (density, energy, u0)
    #define FIELD_IDX(j, k) ((k - (y_min - halo_depth)) * width + (j - (x_min - halo_depth)))
    
    // Macro pour l'indexation des coordonnées (vertex/cell)
    // Ces tableaux commencent généralement à 0 dans le chunk/tile
    #define V_IDX(p, p_min) (p - (p_min - 2)) 

    // 1. Initialisation avec le State 1 (Background)
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
        for (int k = y_min - halo_depth; k <= y_max + halo_depth; ++k) {
            for (int j = x_min - halo_depth; j <= x_max + halo_depth; ++j) {
                
                bool apply = false;

                if (state_geometry[s] == g_rect) {
                    // Les conditions strictes de TeaLeaf :
                    // On vérifie si la maille est dans les limites globales ET si ses sommets intersectent le rectangle
                    if (j >= x_min && j <= x_max && k >= y_min && k <= y_max) {
                        int j_idx = V_IDX(j, x_min);
                        int k_idx = V_IDX(k, y_min);
                        
                        // Utilisation des sommets (vertices) comme en Fortran
                        if (vertexx[j_idx + 1] >= state_xmin[s] && vertexx[j_idx] < state_xmax[s] &&
                            vertexy[k_idx + 1] >= state_ymin[s] && vertexy[k_idx] < state_ymax[s]) {
                            apply = true;
                        }
                    }
                } 
                else if (state_geometry[s] == g_circ) {
                    int j_idx = V_IDX(j, x_min);
                    int k_idx = V_IDX(k, y_min);
                    double dx = cellx[j_idx] - state_xmin[s];
                    double dy = celly[k_idx] - state_ymin[s];
                    if (std::sqrt(dx*dx + dy*dy) <= state_radius[s]) {
                        apply = true;
                    }
                }
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

    // 3. Calcul final de u0 (Énergie interne volumique)
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