#pragma once
#include <vector>

void update_internal_halo_left_right_kernel(
    int x_min_left, int x_max_left, int y_min_left, int y_max_left,
    double* density_l, double* energy0_l, double* energy1_l, double* u_l, double* p_l, 
    double* sd_l, double* r_l, double* z_l, double* kx_l, double* ky_l, double* di_l,
    
    int x_min_right, int x_max_right, int y_min_right, int y_max_right,
    double* density_r, double* energy0_r, double* energy1_r, double* u_r, double* p_r, 
    double* sd_r, double* r_r, double* z_r, double* kx_r, double* ky_r, double* di_r,
    
    int halo_exchange_depth,
    const int* fields,
    int depth
);

void update_internal_halo_bottom_top_kernel(
    int x_min_bot, int x_max_bot, int y_min_bot, int y_max_bot,
    double* density_b, double* energy0_b, double* energy1_b, double* u_b, double* p_b, 
    double* sd_b, double* r_b, double* z_b, double* kx_b, double* ky_b, double* di_b,
    
    int x_min_top, int x_max_top, int y_min_top, int y_max_top,
    double* density_t, double* energy0_t, double* energy1_t, double* u_t, double* p_t, 
    double* sd_t, double* r_t, double* z_t, double* kx_t, double* ky_t, double* di_t,
    
    int halo_exchange_depth,
    const int* fields,
    int depth
);
