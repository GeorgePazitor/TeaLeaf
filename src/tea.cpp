#include "tealeaf.hpp"
namespace Tea{

  using namespace Definitions;
  // using namespace  pack_module
  // using namespace  global_mpi_module
  // using namespace  report_module

  void tea_init_comms(){
    int err, rank, size;
    int periodic[2] = {0,0};
     
    mpi_dims[0]=0;
    mpi_dims[1]=0;

    rank=0;
    size=1;

    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Dims_create(size, 2, mpi_dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, mpi_dims, periodic, 1, &mpi_cart_comm);

    MPI_Comm_rank(mpi_cart_comm, &rank);
    MPI_Comm_size(mpi_cart_comm, &size);
    MPI_Cart_coords(mpi_cart_comm, rank, 2, mpi_coords);

    if (rank == 0) {
        parallel.boss = true;
    }

    parallel.task = rank;
    parallel.boss_task = 0;
    parallel.max_task = size;
  }
}