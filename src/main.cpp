# include "tealeaf.hpp"

int main(int argc, char** argv){


    //!$ INTEGER :: OMP_GET_NUM_THREADS,OMP_GET_THREAD_NUM

    tea_init_comms();

   //#pragma omp parallel
   //{
   if(parallel.boss){
   //if(OMP_GET_THREAD_NUM() == 0){
        fprintf(stdout, "\n");
        fprintf(stdout, "\nTea Version: ",g_version);
        fprintf(stdout, "\nMPI Version: ");
    //fprintf(stdout, "OpenMP Version");
    //fprintf(stdout, "\nTask Count: ",parallel.max_task );
    //fprintf(stdout, "Thread Count: "',OMP_GET_NUM_THREADS()
        fprintf(stdout, "\n");
    //}
   }
   //}

  initialise();

  diffuse();

//Deallocate everything

}