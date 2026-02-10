#include "pack_kernel.h"

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


using namespace TeaLeaf;

#define GET_IDX(x, y, stride) ((y) * (stride) + (x))


int yincs(int field_type) {
    if (field_type == VERTEX_DATA || field_type == Y_FACE_DATA) return 1;
    return 0;
}

int xincs(int field_type) {
    if (field_type == VERTEX_DATA || field_type == X_FACE_DATA) return 1;
    return 0;
}

// Function Pointer Type Definition
typedef void (*pack_func_t)(
    int x_min, int x_max, int y_min, int y_max, int halo_exchange_depth,
    double* field, double* buffer,
    int depth, int x_inc, int y_inc,
    int buffer_offset, int edge_minus, int edge_plus
);

// Forward declarations of specific kernels
void tea_pack_message_left(int, int, int, int, int, double*, double*, int, int, int, int, int, int);
void tea_unpack_message_left(int, int, int, int, int, double*, double*, int, int, int, int, int, int);
void tea_pack_message_right(int, int, int, int, int, double*, double*, int, int, int, int, int, int);
void tea_unpack_message_right(int, int, int, int, int, double*, double*, int, int, int, int, int, int);
void tea_pack_message_top(int, int, int, int, int, double*, double*, int, int, int, int, int, int);
void tea_unpack_message_top(int, int, int, int, int, double*, double*, int, int, int, int, int, int);
void tea_pack_message_bottom(int, int, int, int, int, double*, double*, int, int, int, int, int, int);
void tea_unpack_message_bottom(int, int, int, int, int, double*, double*, int, int, int, int, int, int);

// ----------------------------------------------------------------------------
// Main Dispatcher
// ----------------------------------------------------------------------------

void pack_all(
    int x_min, int x_max, int y_min, int y_max, int halo_exchange_depth,
    const std::array<int, 4> tile_neighbours,
    double* density, double* energy0, double* energy1,
    double* u, double* p, double* sd,
    double* r, double* z, double* kx, double* ky, double* di,
    int* fields, int depth, int face, bool packing, 
    double* mpi_buffer, int* offsets, int tile_offset
) {
    int edge_minus = 0;
    int edge_plus = 0;
    pack_func_t pack_func = nullptr;

    // Logic to determine edge offsets
    switch (face) {
        case CHUNK_LEFT:
        case CHUNK_RIGHT:
            if (tile_neighbours[CHUNK_BOTTOM] == EXTERNAL_FACE) edge_minus = depth;
            if (tile_neighbours[CHUNK_TOP] == EXTERNAL_FACE)    edge_plus = depth;
            break;
        case CHUNK_BOTTOM:
        case CHUNK_TOP:
            if (tile_neighbours[CHUNK_LEFT] == EXTERNAL_FACE)   edge_minus = depth;
            if (tile_neighbours[CHUNK_RIGHT] == EXTERNAL_FACE)  edge_plus = depth;
            break;
    }

    // Select the correct kernel function
    if (packing) {
        switch (face) {
            case CHUNK_LEFT:   pack_func = &tea_pack_message_left; break;
            case CHUNK_RIGHT:  pack_func = &tea_pack_message_right; break;
            case CHUNK_BOTTOM: pack_func = &tea_pack_message_bottom; break;
            case CHUNK_TOP:    pack_func = &tea_pack_message_top; break;
        }
    } else {
        switch (face) {
            case CHUNK_LEFT:   pack_func = &tea_unpack_message_left; break;
            case CHUNK_RIGHT:  pack_func = &tea_unpack_message_right; break;
            case CHUNK_BOTTOM: pack_func = &tea_unpack_message_bottom; break;
            case CHUNK_TOP:    pack_func = &tea_unpack_message_top; break;
        }
    }

    #ifdef OMP
    #pragma omp parallel 
    {
    #endif
        if (fields[FIELD_DENSITY]) {
            pack_func(x_min, x_max, y_min, y_max, halo_exchange_depth, density, mpi_buffer,
                      depth, xincs(CELL_DATA), yincs(CELL_DATA),
                      tile_offset + offsets[FIELD_DENSITY], edge_minus, edge_plus);
        }
        if (fields[FIELD_ENERGY0]) {
            pack_func(x_min, x_max, y_min, y_max, halo_exchange_depth, energy0, mpi_buffer,
                      depth, xincs(CELL_DATA), yincs(CELL_DATA),
                      tile_offset + offsets[FIELD_ENERGY0], edge_minus, edge_plus);
        }
        if (fields[FIELD_ENERGY1]) {
            pack_func(x_min, x_max, y_min, y_max, halo_exchange_depth, energy1, mpi_buffer,
                      depth, xincs(CELL_DATA), yincs(CELL_DATA),
                      tile_offset + offsets[FIELD_ENERGY1], edge_minus, edge_plus);
        }
        if (fields[FIELD_U]) {
            pack_func(x_min, x_max, y_min, y_max, halo_exchange_depth, u, mpi_buffer,
                      depth, xincs(CELL_DATA), yincs(CELL_DATA),
                      tile_offset + offsets[FIELD_U], edge_minus, edge_plus);
        }
        if (fields[FIELD_P]) {
            pack_func(x_min, x_max, y_min, y_max, halo_exchange_depth, p, mpi_buffer,
                      depth, xincs(CELL_DATA), yincs(CELL_DATA),
                      tile_offset + offsets[FIELD_P], edge_minus, edge_plus);
        }
        if (fields[FIELD_SD]) {
            pack_func(x_min, x_max, y_min, y_max, halo_exchange_depth, sd, mpi_buffer,
                      depth, xincs(CELL_DATA), yincs(CELL_DATA),
                      tile_offset + offsets[FIELD_SD], edge_minus, edge_plus);
        }
        if (fields[FIELD_R]) {
            pack_func(x_min, x_max, y_min, y_max, halo_exchange_depth, r, mpi_buffer,
                      depth, xincs(CELL_DATA), yincs(CELL_DATA),
                      tile_offset + offsets[FIELD_R], edge_minus, edge_plus);
        }
        if (fields[FIELD_Z]) {
            pack_func(x_min, x_max, y_min, y_max, halo_exchange_depth, z, mpi_buffer,
                      depth, xincs(CELL_DATA), yincs(CELL_DATA),
                      tile_offset + offsets[FIELD_Z], edge_minus, edge_plus);
        }
        if (fields[FIELD_KX]) {
            pack_func(x_min, x_max, y_min, y_max, halo_exchange_depth, kx, mpi_buffer,
                      depth, xincs(CELL_DATA), yincs(CELL_DATA),
                      tile_offset + offsets[FIELD_KX], edge_minus, edge_plus);
        }
        if (fields[FIELD_KY]) {
            pack_func(x_min, x_max, y_min, y_max, halo_exchange_depth, ky, mpi_buffer,
                      depth, xincs(CELL_DATA), yincs(CELL_DATA),
                      tile_offset + offsets[FIELD_KY], edge_minus, edge_plus);
        }
        if (fields[FIELD_DI]) {
            pack_func(x_min, x_max, y_min, y_max, halo_exchange_depth, di, mpi_buffer,
                      depth, xincs(CELL_DATA), yincs(CELL_DATA),
                      tile_offset + offsets[FIELD_DI], edge_minus, edge_plus);
        }
    #ifdef OMP
    }
    #endif
}

// ----------------------------------------------------------------------------
// Implementations of Specific Kernels
// ----------------------------------------------------------------------------

void tea_pack_message_left(int x_min, int x_max, int y_min, int y_max, int halo, double* field, double* buf, int depth, int x_inc, int y_inc, int buf_off, int e_minus, int e_plus) {
    int stride = (x_max + halo) - (x_min - halo) + 1; 

    #pragma omp for nowait
    for (int k = y_min - e_minus; k <= y_max + y_inc + e_plus; ++k) {
        for (int j = 1; j <= depth; ++j) {
            int index = buf_off + j + (k + depth - 1) * depth; 
            // Fortran: field(x_min+x_inc-1+j, k)
            // C++ Adjustment required for 0-based indexing if k/j are 1-based logic
            buf[index - 1] = field[GET_IDX(x_min + x_inc - 1 + j, k, stride)]; 
        }
    }
}

void tea_unpack_message_left(int x_min, int x_max, int y_min, int y_max, int halo, double* field, double* buf, int depth, int x_inc, int y_inc, int buf_off, int e_minus, int e_plus) {
    int stride = (x_max + halo) - (x_min - halo) + 1;

    #pragma omp for nowait
    for (int k = y_min - e_minus; k <= y_max + y_inc + e_plus; ++k) {
        for (int j = 1; j <= depth; ++j) {
            int index = buf_off + j + (k + depth - 1) * depth;
            field[GET_IDX(x_min - j, k, stride)] = buf[index - 1];
        }
    }
}

void tea_pack_message_right(int x_min, int x_max, int y_min, int y_max, int halo, double* field, double* buf, int depth, int x_inc, int y_inc, int buf_off, int e_minus, int e_plus) {
    int stride = (x_max + halo) - (x_min - halo) + 1;

    #pragma omp for nowait
    for (int k = y_min - e_minus; k <= y_max + y_inc + e_plus; ++k) {
        for (int j = 1; j <= depth; ++j) {
            int index = buf_off + j + (k + depth - 1) * depth;
            field[GET_IDX(x_max + 1 - j, k, stride)] = buf[index - 1];
        }
    }
}

void tea_unpack_message_right(int x_min, int x_max, int y_min, int y_max, int halo, double* field, double* buf, int depth, int x_inc, int y_inc, int buf_off, int e_minus, int e_plus) {
    int stride = (x_max + halo) - (x_min - halo) + 1;

    #pragma omp for nowait
    for (int k = y_min - e_minus; k <= y_max + y_inc + e_plus; ++k) {
        for (int j = 1; j <= depth; ++j) {
            int index = buf_off + j + (k + depth - 1) * depth;
            field[GET_IDX(x_max + x_inc + j, k, stride)] = buf[index - 1];
        }
    }
}

void tea_pack_message_top(int x_min, int x_max, int y_min, int y_max, int halo, double* field, double* buf, int depth, int x_inc, int y_inc, int buf_off, int e_minus, int e_plus) {
    int stride = (x_max + halo) - (x_min - halo) + 1;

    #pragma omp for nowait
    for (int k = 1; k <= depth; ++k) {
        for (int j = x_min - e_minus; j <= x_max + x_inc + e_plus; ++j) {
            int index = buf_off + j + e_minus + (k - 1) * (x_max + x_inc + (e_plus + e_minus));
            buf[index - 1] = field[GET_IDX(j, y_max + 1 - k, stride)];
        }
    }
}

void tea_unpack_message_top(int x_min, int x_max, int y_min, int y_max, int halo, double* field, double* buf, int depth, int x_inc, int y_inc, int buf_off, int e_minus, int e_plus) {
    int stride = (x_max + halo) - (x_min - halo) + 1;

    #pragma omp for nowait
    for (int k = 1; k <= depth; ++k) {
        for (int j = x_min - e_minus; j <= x_max + x_inc + e_plus; ++j) {
            int index = buf_off + j + e_minus + (k - 1) * (x_max + x_inc + (e_plus + e_minus));
            field[GET_IDX(j, y_max + y_inc + k, stride)] = buf[index - 1];
        }
    }
}

void tea_pack_message_bottom(int x_min, int x_max, int y_min, int y_max, int halo, double* field, double* buf, int depth, int x_inc, int y_inc, int buf_off, int e_minus, int e_plus) {
    int stride = (x_max + halo) - (x_min - halo) + 1;

    #pragma omp for nowait
    for (int k = 1; k <= depth; ++k) {
        for (int j = x_min - e_minus; j <= x_max + x_inc + e_plus; ++j) {
            int index = buf_off + j + e_minus + (k - 1) * (x_max + x_inc + (e_plus + e_minus));
            buf[index - 1] = field[GET_IDX(j, y_min + y_inc - 1 + k, stride)];
        }
    }
}

void tea_unpack_message_bottom(int x_min, int x_max, int y_min, int y_max, int halo, double* field, double* buf, int depth, int x_inc, int y_inc, int buf_off, int e_minus, int e_plus) {
    int stride = (x_max + halo) - (x_min - halo) + 1;

    #pragma omp for nowait
    for (int k = 1; k <= depth; ++k) {
        for (int j = x_min - e_minus; j <= x_max + x_inc + e_plus; ++j) {
            int index = buf_off + j + e_minus + (k - 1) * (x_max + x_inc + (e_plus + e_minus));
            field[GET_IDX(j, y_min - k, stride)] = buf[index - 1];
        }
    }
}