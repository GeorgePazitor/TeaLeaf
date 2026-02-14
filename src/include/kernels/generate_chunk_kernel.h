#pragma once

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
    int g_point
);

