#pragma once
#include <vector>
#include <array>
//using namespace TeaLeaf;

void update_halo_kernel(
    int x_min, int x_max, 
    int y_min, int y_max, 
    int halo_exchange_depth,
    const std::array<int, 4>& chunk_neighbours,
    const std::array<int, 4>& tile_neighbours, 
    double* density, 
    double* energy0, 
    double* energy1, 
    double* u, 
    double* p, 
    double* sd, 
    double* r, 
    double* z, 
    double* kx, 
    double* ky, 
    double* di, 
    const int* fields, 
    int depth
);

