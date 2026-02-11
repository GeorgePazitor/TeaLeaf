#include "generate_chunk_kernel.h"
#include <omp.h>
#include <cmath>
#include <algorithm>

void generate_chunk_kernel(
    int x_min, int x_max, 
    int y_min, int y_max, 
    int halo_exchange_depth,
    double* vertexx, 
    double* vertexy, 
    double* cellx, 
    double* celly,
    double* density, 
    double* energy0, 
    double* u0,
    int number_of_states,
    double* state_density, 
    double* state_energy, 
    double* state_xmin, 
    double* state_xmax, 
    double* state_ymin, 
    double* state_ymax, 
    double* state_radius, 
    int* state_geometry, 
    int g_rect, 
    int g_circ, 
    int g_point) 
{
    
    //Field arrays (Density/Energy) start at [x_min - halo]
    int x_off_field = x_min - halo_exchange_depth;
    int y_off_field = y_min - halo_exchange_depth;
    
    // Width of the field arrays (including full halos)
    int field_width = (x_max + halo_exchange_depth) - (x_min - halo_exchange_depth) + 1;

    // Geometry Arrays (Vertex/Cell) start at [x_min - 2]
    int x_off_geo = x_min - 2;
    int y_off_geo = y_min - 2;

    #pragma omp parallel
    {
        // state 1: background 
        double bg_dens = state_density[0];
        double bg_ener = state_energy[0];

        #pragma omp for collapse(2)
        for (int k = y_min - halo_exchange_depth; k <= y_max + halo_exchange_depth; ++k) {
            for (int j = x_min - halo_exchange_depth; j <= x_max + halo_exchange_depth; ++j) {
                int idx = (k - y_off_field) * field_width + (j - x_off_field);
                density[idx] = bg_dens;
                energy0[idx] = bg_ener;
            }
        }

        // sequential Loop over states
        // goes from 1 since the first is the background state
        for (int s = 1; s < number_of_states; ++s) {
            
            double s_xmin = state_xmin[s];
            double s_xmax = state_xmax[s];
            double s_ymin = state_ymin[s];
            double s_ymax = state_ymax[s];
            double s_rad  = state_radius[s];
            int    s_geo  = state_geometry[s];
            double s_den  = state_density[s];
            double s_ene  = state_energy[s];

            #pragma omp for collapse(2)
            for (int k = y_min - halo_exchange_depth; k <= y_max + halo_exchange_depth; ++k) {
                for (int j = x_min - halo_exchange_depth; j <= x_max + halo_exchange_depth; ++j) {
                    
                    int idx = (k - y_off_field) * field_width + (j - x_off_field);
                    
                    // Indices for geometry arrays
                    int j_geo = j - x_off_geo;
                    int k_geo = k - y_off_geo;

                    bool apply_state = false;

                    if (s_geo == g_rect) {
                        //rectangles only apply to the internal domain 
                        if (j >= x_min && j <= x_max && k >= y_min && k <= y_max) {
                            // Check X overlap
                            // vertexx[j_geo] is left node, vertexx[j_geo+1] is right node
                            if (vertexx[j_geo+1] >= s_xmin && vertexx[j_geo] < s_xmax) {
                                // Check overlap y
                                if (vertexy[k_geo+1] >= s_ymin && vertexy[k_geo] < s_ymax) {
                                    apply_state = true;
                                }
                            }
                        }
                    } 
                    else if (s_geo == g_circ) {
                        double x_cent = s_xmin; 
                        double y_cent = s_ymin; 
                        
                        double dx = cellx[j_geo] - x_cent;
                        double dy = celly[k_geo] - y_cent;
                        double radius = std::sqrt(dx*dx + dy*dy);

                        if (radius <= s_rad) {
                            apply_state = true;
                        }
                    } 
                    else if (s_geo == g_point) {
                        double x_cent = s_xmin;
                        double y_cent = s_ymin;
                        
                        //exact match on vertex coordinates
                        if (vertexx[j_geo] == x_cent && vertexy[k_geo] == y_cent) {
                            apply_state = true;
                        }
                    }

                    if (apply_state) {
                        density[idx] = s_den;
                        energy0[idx] = s_ene;
                    }
                }
            }
        } 

        // calculate internal energy (u0) 
        #pragma omp for collapse(2)
        for (int k = y_min - halo_exchange_depth; k <= y_max + halo_exchange_depth; ++k) {
            for (int j = x_min - halo_exchange_depth; j <= x_max + halo_exchange_depth; ++j) {
                int idx = (k - y_off_field) * field_width + (j - x_off_field);
                u0[idx] = energy0[idx] * density[idx];
            }
        }

    } 
}