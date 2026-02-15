#pragma once

#include <vector>
#include <array>
#include "include/data.h"
#define GET_IDX(x, y, stride) ((y) * (stride) + (x))


int yincs(int field_type);

int xincs(int field_type);

typedef void (*pack_func_t)(
    int x_min, int x_max, int y_min, int y_max, int halo_exchange_depth,
    double* field, double* buffer,
    int depth, int x_inc, int y_inc,
    int buffer_offset, int edge_minus, int edge_plus
);

void tea_pack_message_left(int, int, int, int, int, double*, double*, int, int, int, int, int, int);
void tea_unpack_message_left(int, int, int, int, int, double*, double*, int, int, int, int, int, int);
void tea_pack_message_right(int, int, int, int, int, double*, double*, int, int, int, int, int, int);
void tea_unpack_message_right(int, int, int, int, int, double*, double*, int, int, int, int, int, int);
void tea_pack_message_top(int, int, int, int, int, double*, double*, int, int, int, int, int, int);
void tea_unpack_message_top(int, int, int, int, int, double*, double*, int, int, int, int, int, int);
void tea_pack_message_bottom(int, int, int, int, int, double*, double*, int, int, int, int, int, int);
void tea_unpack_message_bottom(int, int, int, int, int, double*, double*, int, int, int, int, int, int);


void pack_all(
    int x_min, int x_max, int y_min, int y_max, int halo_exchange_depth,
    const std::array<int, 4> tile_neighbours,
    double* density, double* energy0, double* energy1,
    double* u, double* p, double* sd,
    double* r, double* z, double* kx, double* ky, double* di,
    const int* fields, int depth, int face, bool packing, 
    double* mpi_buffer, int* offsets, int tile_offset
);

