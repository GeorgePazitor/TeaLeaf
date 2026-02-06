#ifndef PACK_KERNEL_H
#define PACK_KERNEL_H
// ----------------------------------------------------------------------------
// Helper Macros for Indexing
// ----------------------------------------------------------------------------
// ASSUMPTION: The grid is stored in a 1D array. 
// You must adjust 'row_stride' to match your actual allocated width 
// (e.g., x_max - x_min + 2*halo + 1).
// Below assumes Row-Major (y * stride + x) common in C++. 
// If you are keeping Fortran Column-Major in C++, swap x and y.

// Note: 'stride' here assumes the allocated width of the field.
// We approximate stride based on x_min/x_max + halo. 
// A safer way is to pass the 'stride' explicitly from the calling code.
// For this translation, I calculate it assuming standard TeaLeaf layout.
#include <vector>
#include <array>
#include "data.h"
#define GET_IDX(x, y, stride) ((y) * (stride) + (x))


int yincs(int field_type);

int xincs(int field_type);

// Function Pointer Type Definition
typedef void (*pack_func_t)(
    int x_min, int x_max, int y_min, int y_max, int halo_exchange_depth,
    std::vector<double>& field, std::vector<double>& buffer,
    int depth, int x_inc, int y_inc,
    int buffer_offset, int edge_minus, int edge_plus
);

// Forward declarations of specific kernels
void tea_pack_message_left(int, int, int, int, int, std::vector<double>&, std::vector<double>&, int, int, int, int, int, int);
void tea_unpack_message_left(int, int, int, int, int, std::vector<double>&, std::vector<double>&, int, int, int, int, int, int);
void tea_pack_message_right(int, int, int, int, int, std::vector<double>&, std::vector<double>&, int, int, int, int, int, int);
void tea_unpack_message_right(int, int, int, int, int, std::vector<double>&, std::vector<double>&, int, int, int, int, int, int);
void tea_pack_message_top(int, int, int, int, int, std::vector<double>&, std::vector<double>&, int, int, int, int, int, int);
void tea_unpack_message_top(int, int, int, int, int, std::vector<double>&, std::vector<double>&, int, int, int, int, int, int);
void tea_pack_message_bottom(int, int, int, int, int, std::vector<double>&, std::vector<double>&, int, int, int, int, int, int);
void tea_unpack_message_bottom(int, int, int, int, int, std::vector<double>&, std::vector<double>&, int, int, int, int, int, int);

// Main Dispatcher

void pack_all(
    int x_min, int x_max, int y_min, int y_max, int halo_exchange_depth,
    const std::array<int, 4> tile_neighbours,
    std::vector<double>& density, std::vector<double>& energy0, std::vector<double>& energy1,
    std::vector<double>& u, std::vector<double>& p, std::vector<double>& sd,
    std::vector<double>& r, std::vector<double>& z, std::vector<double>& kx, std::vector<double>& ky, std::vector<double>& di,
    int* fields, int depth, int face, bool packing, 
    std::vector<double>& mpi_buffer, int* offsets, int tile_offset
);

#endif