#pragma once

#include <mpi.h>
#include <iostream>
#include <vector>

namespace TeaLeaf {

    // Version, constants
    constexpr double g_version = 0.1;
    constexpr int    g_ibig    = 640000;

    constexpr double g_small   = 1.0e-16;
    constexpr double g_big     = 1.0e+21;

    constexpr int g_name_len_max = 255;
    constexpr int g_xdir         = 1;
    constexpr int g_ydir         = 2;

    // Halo Faces 
    constexpr int CHUNK_LEFT    = 0;
    constexpr int CHUNK_RIGHT   = 1;
    constexpr int CHUNK_BOTTOM  = 2;
    constexpr int CHUNK_TOP     = 3;
    constexpr int EXTERNAL_FACE = -1;

    // Fields
    constexpr int FIELD_DENSITY = 0;
    constexpr int FIELD_ENERGY0 = 1;
    constexpr int FIELD_ENERGY1 = 2;
    constexpr int FIELD_U       = 3;
    constexpr int FIELD_P       = 4;
    constexpr int FIELD_SD      = 5;
    constexpr int FIELD_R       = 6;
    constexpr int FIELD_Z       = 7;
    constexpr int FIELD_KX      = 8;
    constexpr int FIELD_KY      = 9;
    constexpr int FIELD_DI      = 10;
    constexpr int NUM_FIELDS    = 11;

    constexpr int CELL_DATA   = 0;
    constexpr int VERTEX_DATA = 1;
    constexpr int X_FACE_DATA = 2;
    constexpr int Y_FACE_DATA = 3;

    constexpr int FIXED = 1;

    constexpr int g_rect  = 1;
    constexpr int g_circ  = 2;
    constexpr int g_point = 3;

    constexpr int CONDUCTIVITY       = 1;
    constexpr int RECIP_CONDUCTIVITY = 2;

    constexpr int TL_PREC_NONE      = 1;
    constexpr int TL_PREC_JAC_DIAG  = 2;
    constexpr int TL_PREC_JAC_BLOCK = 3;

    constexpr int g_len_max = 500;

    // Parallel structure
    struct Parallel_type {
        bool boss;
        int max_task;
        int boss_task;
        int task;
    };

    // Globals
    extern std::ostream* g_out;

    extern int tiles_per_task;
    extern int sub_tiles_per_tile;
    extern MPI_Comm mpi_cart_comm;
    extern int mpi_dims[2];
    extern int mpi_coords[2];

    extern Parallel_type parallel;

}
