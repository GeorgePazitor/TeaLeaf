#include "include/data.h"
namespace TeaLeaf {
    
    // Globals definitions
    std::ostream* g_out = &std::cout;

    // Structure holding rank info, "boss" status, and decomposition state
    Parallel_type parallel = {};

    // Tiling and MPI Topology variables
    int tiles_per_task = 0;
    int sub_tiles_per_tile = 0;
    MPI_Comm mpi_cart_comm = MPI_COMM_NULL; // Cartesian communicator for 2D decomposition

    int mpi_dims[2] = {0, 0};   // Number of MPI processes in X and Y
    int mpi_coords[2] = {0, 0}; // Rank coordinates in the Cartesian grid
};
