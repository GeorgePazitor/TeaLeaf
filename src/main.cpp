#include "include/data.h"
#include "include/tea.h"
#include "include/initialise.h"
#include "include/diffuse.h"

#include <mpi.h>
#include <omp.h>

using namespace TeaLeaf;

/**
 * Entry point for TeaLeaf C++ port. 
 * Orchestrates the heat conduction simulation using MPI and OpenMP.
 */
int main(int argc, char** argv){

    // Setup MPI environment and global communicators
    tea_init_comms();

    // The 'boss' (rank 0) handles administrative logging and metadata output
    if (parallel.boss) {
        std::cout << "\n\nTea Version: " << g_version;
        std::cout << "\nMPI Version:\n";
        std::cout << "OpenMP Version\n";
        std::cout << "\nTask Count: " << parallel.max_task;
        std::cout << "Thread Count: " << omp_get_max_threads();
    }

    // Parses input files, allocates grids, and sets up the initial state
    initialise();

    // Core computational loop for the diffusion solvers (CG, PPCG, or Cheby)
    diffuse();

    // Final cleanup: MPI termination is typically handled inside a destructor 
    // or a finalization call within the tea_init_comms lifecycle.
}