#include "data.h"
#include "tea.h"
#include "initialise.h"
#include "diffuse.h"

#include <mpi.h>
#include <omp.h>

using namespace TeaLeaf;

int main(int argc, char** argv){

    tea_init_comms();

    #pragma omp parallel
    {
    
        if(parallel.boss){
            if(omp_get_thread_num() == 0){
                std::cout << "\n";
                std::cout << "\nTea Version: " << g_version;
                std::cout << "\nMPI Version: \n";
                std::cout << "OpenMP Version. \n";
                std::cout << "\nTask Count: " << parallel.max_task ;
                std::cout << "Thread Count: " << omp_get_num_threads() ;
            }
        }
     
    }
    
  initialise();

  diffuse();

//Deallocate everything

}
