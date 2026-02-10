#ifndef DATA_H
#define DATA_H

#include <mpi.h>
#include <iostream>
#include <vector>

namespace TeaLeaf {
    constexpr double g_version = 0.1;
    constexpr int    g_ibig    = 640000;

    constexpr double g_small   = 1.0e-16;
    constexpr double g_big     = 1.0e+21;

    constexpr int g_name_len_max = 255;
    constexpr int g_xdir         = 1;
    constexpr int g_ydir         = 2;

    // Neighbors / Halo Faces (Converted to 0-based)
    constexpr int CHUNK_LEFT    = 0;
    constexpr int CHUNK_RIGHT   = 1;
    constexpr int CHUNK_BOTTOM  = 2;
    constexpr int CHUNK_TOP     = 3;
    constexpr int EXTERNAL_FACE = -1;

    // Field Indices (Converted to 0-based)
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
    constexpr int NUM_FIELDS    = 11; // pr 10 idk, in the source code it's the same as FIELD_DI

    // Data Locations (Converted to 0-based for consistency, though these act as Enums)
    constexpr int CELL_DATA   = 0;
    constexpr int VERTEX_DATA = 1;
    constexpr int X_FACE_DATA = 2;
    constexpr int Y_FACE_DATA = 3;

    // Time step control constants
    constexpr int FIXED = 1;

    // Geometry types
    constexpr int g_rect  = 1;
    constexpr int g_circ  = 2;
    constexpr int g_point = 3;

    // Solver Options
    constexpr int CONDUCTIVITY       = 1;
    constexpr int RECIP_CONDUCTIVITY = 2;

    // Preconditioners
    constexpr int TL_PREC_NONE      = 1;
    constexpr int TL_PREC_JAC_DIAG  = 2;
    constexpr int TL_PREC_JAC_BLOCK = 3;

    constexpr int g_len_max = 500;


    // Structures
    struct parallel_type {
        bool boss        = false;
        int max_task     = 0;
        int boss_task    = 0;
        int task         = 0;
    };

    // Global Variable Declarations (Extern)

    //extern int g_in; file 
    extern std::ostream* g_out;
    extern parallel_type parallel;
    extern int tiles_per_task;
    extern int sub_tiles_per_tile;
    extern MPI_Comm mpi_cart_comm;
    extern int mpi_dims[2];
    extern int mpi_coords[2];

    struct tile_type {
        // Coordonnées et voisins (ce que tu avais déjà)
        int left, right, bottom, top;
        int tile_coords[2];
        int tile_neighbours[4];

        // --- AJOUT : Limites locales de la tile pour les noyaux ---
        int x_min, x_max;
        int y_min, y_max;

        // --- AJOUT : Les données physiques (le "field") ---
        // On les met directement ici pour simplifier l'accès
        std::vector<double> density;
        std::vector<double> energy0;
        std::vector<double> energy1;
        std::vector<double> u;
        std::vector<double> vector_p;
        std::vector<double> vector_sd;
        std::vector<double> vector_rtemp;
        std::vector<double> vector_z;
        std::vector<double> vector_Kx;
        std::vector<double> vector_Ky;
        std::vector<double> vector_Di;
    };

    struct chunk_type {
        int left, right, bottom, top;
        int x_cells, y_cells;
        int chunk_neighbours[4]; 
        
        int tile_dims[2];
        int sub_tile_dims[2];
        int halo_exchange_depth;
        
        std::vector<tile_type> tiles; 

        // Buffers de communication MPI
        std::vector<double> left_snd_buffer,  left_rcv_buffer;
        std::vector<double> right_snd_buffer, right_rcv_buffer;
        std::vector<double> bottom_snd_buffer, bottom_rcv_buffer;
        std::vector<double> top_snd_buffer,    top_rcv_buffer;
    };

    extern chunk_type chunk;

};
#endif