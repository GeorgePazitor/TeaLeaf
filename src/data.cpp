#include "data.h"

namespace TeaLeaf {

    std::ostream* g_out = &std::cout;
    parallel_type parallel;
    int tiles_per_task = 0;
    int sub_tiles_per_tile = 0;
    MPI_Comm mpi_cart_comm = MPI_COMM_NULL;
    int mpi_dims[2] = {0, 0};
    int mpi_coords[2] = {0, 0};
};
