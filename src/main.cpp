#include "data.h"
#include "tea.hpp"
#include "initialise.hpp"

#include <mpi.h>

int main(int argc, char** argv){

    tea_init_comms();

    #ifdef OMP
    #pragma omp parallel
    {
    #endif
        if(parallel.boss){
            #ifdef OMP
            if(OMP_GET_THREAD_NUM() == 0){
            #endif
                std::cout << "\n";
                std::cout << "\nTea Version: " << g_version;
                std::cout << "\nMPI Version: ";
                #ifdef OMP
                std::cout << "OpenMP Version";
                std::cout << "\nTask Count: " << parallel.max_task ;
                std::cout << "Thread Count: " << OMP_GET_NUM_THREADS() ;
            }
            #endif
        }
    #ifdef OMP 
    }
    #endif

  initialise();

  diffuse();

//Deallocate everything

}