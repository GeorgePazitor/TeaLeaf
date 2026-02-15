#include <iostream>
#include <iomanip>
#include <algorithm>
#include <omp.h>

#include "include/timestep.h"
#include "include/data.h"
#include "include/calc_dt.h"
#include "include/tea.h"
#include "include/global_mpi.h"
#include "include/definitions.h"

namespace TeaLeaf {

/**
*Computes the next time step (dt) for the simulation, ensures the time step is small enough to capture the physics accurately
*synchronisation among all MPI ranks 
*/
void timestep() {
    double kernel_time = 0.0;
    double dtlp; //local time step for a single tile

    if (profiler_on) {
        kernel_time = timer();
    }

    #ifdef OMP
    #pragma omp parallel private(dtlp)
    #endif
    {
        #ifdef OMP
        #pragma omp for nowait
        #endif
        for (int t = 0; t < tiles_per_task; ++t) {

            //computes dt for this specific tile
            calc_dt(dtlp);

            #ifdef OMP
            #pragma omp critical
            #endif
            {
                if (dtlp <= dt) {
                    dt = dtlp;
                }
            }
        }
    }

    //global MPI reduction minimum dt across all processes
    tea_min(dt);

    if (profiler_on) {
        profiler.timestep += (timer() - kernel_time);
    }

    if (parallel.boss) {
        auto print_msg = [&](std::ostream& os) {
            os << " Step " << std::setw(7) << step
               << " time " << std::fixed << std::setprecision(7) << std::setw(11) << timee
               << " timestep  " << std::scientific << std::setprecision(2) << std::setw(9) << dt 
               << std::endl;
        };

        if (g_out) print_msg(*g_out); 
        print_msg(std::cerr);        
    }
}

} // namespace TeaLeaf