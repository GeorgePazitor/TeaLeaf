#include "data.h"
//#include "definitions.h"
namespace TeaLeaf {
    // Actually create the variables here (Initialize them!)

    // System & MPI
    
    std::ostream* g_out = &std::cout; 

    Parallel_type parallel;

    int tiles_per_task;
    int sub_tiles_per_tile;
    MPI_Comm mpi_cart_comm;

    int mpi_dims[2];
    int mpi_coords[2];

}
