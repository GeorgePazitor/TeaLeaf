#include <iostream>
#include <iomanip>
#include <algorithm>
#include <omp.h>

#include "timestep.h"
#include "data.h"
#include "calc_dt.h"
#include "tea.h"
#include "global_mpi.h"
#include "definitions.h"

namespace TeaLeaf {

void timestep() {
    double kernel_time = 0.0;
    double dtlp; // Pas de temps local par tuile

    if (profiler_on) {
        kernel_time = timer();
    }

    // Parallélisation OpenMP sur les tuiles
    #pragma omp parallel private(dtlp)
    {
        #pragma omp for nowait
        for (int t = 0; t < tiles_per_task; ++t) {

            calc_dt(dtlp);

            // Section critique pour mettre à jour le dt global du chunk
            #pragma omp critical
            {
                if (dtlp <= dt) {
                    dt = dtlp;
                }
            }
        }
    }

    // Réduction MPI pour obtenir le minimum sur tous les rangs
    // Équivalent de CALL tea_min(dt)
    tea_min(dt);

    if (profiler_on) {
        profiler.timestep += (timer() - kernel_time);
    }

    // Sorties console et fichier pour le processus maître
    if (parallel.boss) {
        // Formatage Fortran : (' Step ', i7,' time ', f11.7,' timestep  ',1pe9.2,i8)
        auto print_msg = [&](std::ostream& os) {
            os << " Step " << std::setw(7) << step
               << " time " << std::fixed << std::setprecision(7) << std::setw(11) << timee
               << " timestep  " << std::scientific << std::setprecision(2) << std::setw(9) << dt 
               << std::endl;
        };

        if (g_out) print_msg(*g_out); // Vers le fichier log
        print_msg(std::cerr);        // Vers l'unité 0 (stderr en C++)
    }
}

} // namespace TeaLeaf