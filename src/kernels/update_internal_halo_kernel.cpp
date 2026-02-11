#include "update_internal_halo_kernel.h"
#include "data.h" 
#include "definitions.h"
#include <algorithm>
using namespace TeaLeaf;

static void update_internal_halo_cell_left_right(
    int x_min_left, int x_max_left, int y_min_left, int y_max_left,
    double* mesh_left,
    int x_min_right, int x_max_right, int y_min_right, int y_max_right,
    double* mesh_right,
    int halo_depth,
    int depth)
{
    // Stride calculations
    int width_left  = (x_max_left - x_min_left + 1) + 2 * halo_depth;
    int width_right = (x_max_right - x_min_right + 1) + 2 * halo_depth;

    // Index Macros: (y - y_start) * width + (x - x_start)
    // Note: y_start = y_min - halo
    #define IDX_L(x, y) ((y - (y_min_left - halo_depth)) * width_left + (x - (x_min_left - halo_depth)))
    #define IDX_R(x, y) ((y - (y_min_right - halo_depth)) * width_right + (x - (x_min_right - halo_depth)))

    // 1. Copy FROM Left Tile (Right Boundary) TO Right Tile (Left Halo)
    for (int k = y_min_left - depth; k <= y_max_left + depth; ++k) {
        for (int j = 1; j <= depth; ++j) {
            // Fortran: mesh_right(1-j, k) = mesh_left(x_max_left+1-j, k)
            // C++:
            // Dest: Right Tile, x = x_min_right - j
            // Src:  Left Tile,  x = x_max_left - j + 1
            mesh_right[IDX_R(x_min_right - j, k)] = mesh_left[IDX_L(x_max_left - j + 1, k)];
        }
    }

    // 2. Copy FROM Right Tile (Left Boundary) TO Left Tile (Right Halo)
    for (int k = y_min_left - depth; k <= y_max_left + depth; ++k) {
        for (int j = 1; j <= depth; ++j) {
            // Fortran: mesh_left(x_max_left+j, k) = mesh_right(0+j, k)
            // C++:
            // Dest: Left Tile,  x = x_max_left + j
            // Src:  Right Tile, x = x_min_right + j - 1
            mesh_left[IDX_L(x_max_left + j, k)] = mesh_right[IDX_R(x_min_right + j - 1, k)];
        }
    }

    #undef IDX_L
    #undef IDX_R
}


static void update_internal_halo_cell_bottom_top(
    int x_min_bot, int x_max_bot, int y_min_bot, int y_max_bot,
    double* mesh_bot,
    int x_min_top, int x_max_top, int y_min_top, int y_max_top,
    double* mesh_top,
    int halo_depth,
    int depth)
{
    int width_bot = (x_max_bot - x_min_bot + 1) + 2 * halo_depth;
    int width_top = (x_max_top - x_min_top + 1) + 2 * halo_depth;

    #define IDX_B(x, y) ((y - (y_min_bot - halo_depth)) * width_bot + (x - (x_min_bot - halo_depth)))
    #define IDX_T(x, y) ((y - (y_min_top - halo_depth)) * width_top + (x - (x_min_top - halo_depth)))

    // 1. Copy FROM Bottom Tile (Top Boundary) TO Top Tile (Bottom Halo)
    for (int k = 1; k <= depth; ++k) {
        for (int j = x_min_bot - depth; j <= x_max_bot + depth; ++j) {
            // Fortran: mesh_top(j, 1-k) = mesh_bot(j, y_max_bot+1-k)
            // C++:
            // Dest: Top Tile,    y = y_min_top - k
            // Src:  Bottom Tile, y = y_max_bot - k + 1
            mesh_top[IDX_T(j, y_min_top - k)] = mesh_bot[IDX_B(j, y_max_bot - k + 1)];
        }
    }

    // 2. Copy FROM Top Tile (Bottom Boundary) TO Bottom Tile (Top Halo)
    for (int k = 1; k <= depth; ++k) {
        for (int j = x_min_bot - depth; j <= x_max_bot + depth; ++j) {
            // Fortran: mesh_bot(j, y_max_bot+k) = mesh_top(j, 0+k)
            // C++:
            // Dest: Bottom Tile, y = y_max_bot + k
            // Src:  Top Tile,    y = y_min_top + k - 1
            mesh_bot[IDX_B(j, y_max_bot + k)] = mesh_top[IDX_T(j, y_min_top + k - 1)];
        }
    }

    #undef IDX_B
    #undef IDX_T
}

void update_internal_halo_left_right_kernel(
    int x_min_left, int x_max_left, int y_min_left, int y_max_left,
    double* density_l, double* energy0_l, double* energy1_l, double* u_l, double* p_l, 
    double* sd_l, double* r_l, double* z_l, double* kx_l, double* ky_l, double* di_l,
    
    int x_min_right, int x_max_right, int y_min_right, int y_max_right,
    double* density_r, double* energy0_r, double* energy1_r, double* u_r, double* p_r, 
    double* sd_r, double* r_r, double* z_r, double* kx_r, double* ky_r, double* di_r,
    
    int halo_exchange_depth,
    const int* fields,
    int depth)
{

    if (fields[FIELD_DENSITY]) 
        update_internal_halo_cell_left_right(x_min_left, x_max_left, y_min_left, y_max_left, density_l,
                                             x_min_right, x_max_right, y_min_right, y_max_right, density_r,
                                             halo_exchange_depth, depth);

    if (fields[FIELD_ENERGY0])
        update_internal_halo_cell_left_right(x_min_left, x_max_left, y_min_left, y_max_left, energy0_l,
                                             x_min_right, x_max_right, y_min_right, y_max_right, energy0_r,
                                             halo_exchange_depth, depth);

    if (fields[FIELD_ENERGY1])
        update_internal_halo_cell_left_right(x_min_left, x_max_left, y_min_left, y_max_left, energy1_l,
                                             x_min_right, x_max_right, y_min_right, y_max_right, energy1_r,
                                             halo_exchange_depth, depth);
    
    if (fields[FIELD_U])
        update_internal_halo_cell_left_right(x_min_left, x_max_left, y_min_left, y_max_left, u_l,
                                             x_min_right, x_max_right, y_min_right, y_max_right, u_r,
                                             halo_exchange_depth, depth);

    if (fields[FIELD_P])
        update_internal_halo_cell_left_right(x_min_left, x_max_left, y_min_left, y_max_left, p_l,
                                             x_min_right, x_max_right, y_min_right, y_max_right, p_r,
                                             halo_exchange_depth, depth);

    if (fields[FIELD_SD])
        update_internal_halo_cell_left_right(x_min_left, x_max_left, y_min_left, y_max_left, sd_l,
                                             x_min_right, x_max_right, y_min_right, y_max_right, sd_r,
                                             halo_exchange_depth, depth);

    if (fields[FIELD_R])
        update_internal_halo_cell_left_right(x_min_left, x_max_left, y_min_left, y_max_left, r_l,
                                             x_min_right, x_max_right, y_min_right, y_max_right, r_r,
                                             halo_exchange_depth, depth);

    if (fields[FIELD_Z])
        update_internal_halo_cell_left_right(x_min_left, x_max_left, y_min_left, y_max_left, z_l,
                                             x_min_right, x_max_right, y_min_right, y_max_right, z_r,
                                             halo_exchange_depth, depth);

    if (fields[FIELD_KX])
        update_internal_halo_cell_left_right(x_min_left, x_max_left, y_min_left, y_max_left, kx_l,
                                             x_min_right, x_max_right, y_min_right, y_max_right, kx_r,
                                             halo_exchange_depth, depth);

    if (fields[FIELD_KY])
        update_internal_halo_cell_left_right(x_min_left, x_max_left, y_min_left, y_max_left, ky_l,
                                             x_min_right, x_max_right, y_min_right, y_max_right, ky_r,
                                             halo_exchange_depth, depth);

    if (fields[FIELD_DI])
        update_internal_halo_cell_left_right(x_min_left, x_max_left, y_min_left, y_max_left, di_l,
                                             x_min_right, x_max_right, y_min_right, y_max_right, di_r,
                                             halo_exchange_depth, depth);
}


void update_internal_halo_bottom_top_kernel(
    int x_min_bot, int x_max_bot, int y_min_bot, int y_max_bot,
    double* density_b, double* energy0_b, double* energy1_b, double* u_b, double* p_b, 
    double* sd_b, double* r_b, double* z_b, double* kx_b, double* ky_b, double* di_b,
    
    int x_min_top, int x_max_top, int y_min_top, int y_max_top,
    double* density_t, double* energy0_t, double* energy1_t, double* u_t, double* p_t, 
    double* sd_t, double* r_t, double* z_t, double* kx_t, double* ky_t, double* di_t,
    
    int halo_exchange_depth,
    const int* fields,
    int depth)
{
    

    if (fields[FIELD_DENSITY]) 
        update_internal_halo_cell_bottom_top(x_min_bot, x_max_bot, y_min_bot, y_max_bot, density_b,
                                             x_min_top, x_max_top, y_min_top, y_max_top, density_t,
                                             halo_exchange_depth, depth);

    if (fields[FIELD_ENERGY0])
        update_internal_halo_cell_bottom_top(x_min_bot, x_max_bot, y_min_bot, y_max_bot, energy0_b,
                                             x_min_top, x_max_top, y_min_top, y_max_top, energy0_t,
                                             halo_exchange_depth, depth);

    if (fields[FIELD_ENERGY1])
        update_internal_halo_cell_bottom_top(x_min_bot, x_max_bot, y_min_bot, y_max_bot, energy1_b,
                                             x_min_top, x_max_top, y_min_top, y_max_top, energy1_t,
                                             halo_exchange_depth, depth);
    
    if (fields[FIELD_U])
        update_internal_halo_cell_bottom_top(x_min_bot, x_max_bot, y_min_bot, y_max_bot, u_b,
                                             x_min_top, x_max_top, y_min_top, y_max_top, u_t,
                                             halo_exchange_depth, depth);

    if (fields[FIELD_P])
        update_internal_halo_cell_bottom_top(x_min_bot, x_max_bot, y_min_bot, y_max_bot, p_b,
                                             x_min_top, x_max_top, y_min_top, y_max_top, p_t,
                                             halo_exchange_depth, depth);

    if (fields[FIELD_SD])
        update_internal_halo_cell_bottom_top(x_min_bot, x_max_bot, y_min_bot, y_max_bot, sd_b,
                                             x_min_top, x_max_top, y_min_top, y_max_top, sd_t,
                                             halo_exchange_depth, depth);

    if (fields[FIELD_R])
        update_internal_halo_cell_bottom_top(x_min_bot, x_max_bot, y_min_bot, y_max_bot, r_b,
                                             x_min_top, x_max_top, y_min_top, y_max_top, r_t,
                                             halo_exchange_depth, depth);

    if (fields[FIELD_Z])
        update_internal_halo_cell_bottom_top(x_min_bot, x_max_bot, y_min_bot, y_max_bot, z_b,
                                             x_min_top, x_max_top, y_min_top, y_max_top, z_t,
                                             halo_exchange_depth, depth);

    if (fields[FIELD_KX])
        update_internal_halo_cell_bottom_top(x_min_bot, x_max_bot, y_min_bot, y_max_bot, kx_b,
                                             x_min_top, x_max_top, y_min_top, y_max_top, kx_t,
                                             halo_exchange_depth, depth);

    if (fields[FIELD_KY])
        update_internal_halo_cell_bottom_top(x_min_bot, x_max_bot, y_min_bot, y_max_bot, ky_b,
                                             x_min_top, x_max_top, y_min_top, y_max_top, ky_t,
                                             halo_exchange_depth, depth);

    if (fields[FIELD_DI])
        update_internal_halo_cell_bottom_top(x_min_bot, x_max_bot, y_min_bot, y_max_bot, di_b,
                                             x_min_top, x_max_top, y_min_top, y_max_top, di_t,
                                             halo_exchange_depth, depth);
}