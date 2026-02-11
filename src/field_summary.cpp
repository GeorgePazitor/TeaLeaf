#include "field_summary.h"
#include "tea.h"
#include "data.h"
#include "definitions.h"
#include "kernels/field_summary_kernel.h"
#include "global_mpi.h" // For tea_sum
#include <iostream>
#include <iomanip>
#include <cmath>
#include <mpi.h>
#include <omp.h>

void field_summary() {
    using namespace TeaLeaf;

    // Local accumulators
    double vol = 0.0;
    double mass = 0.0;
    double ie = 0.0;
    double temp = 0.0;

    double start_time = 0.0;
    if (profiler_on) start_time = MPI_Wtime();

    if (parallel.boss) {
        *g_out << "\n Time " << timee << "\n";
        *g_out << "             " 
               << std::setw(26) << "Volume"
               << std::setw(26) << "Mass"
               << std::setw(26) << "Density"
               << std::setw(26) << "Energy"
               << std::setw(26) << "U" << "\n";
    }

    #pragma omp parallel for reduction(+:vol, mass, ie, temp)
    for (int t = 0; t < tiles_per_task; ++t) {
        
        double tile_vol = 0.0;
        double tile_mass = 0.0;
        double tile_ie = 0.0;
        double tile_temp = 0.0;

        auto& tile = chunk.tiles[t];

        field_summary_kernel(
            tile.field.x_min,
            tile.field.x_max,
            tile.field.y_min,
            tile.field.y_max,
            chunk.halo_exchange_depth,
            tile.field.volume.data(),
            tile.field.density.data(),
            tile.field.energy1.data(),
            tile.field.u.data(),
            tile_vol,
            tile_mass,
            tile_ie,
            tile_temp
        );

        // Accumulate into reduction variables
        vol  += tile_vol;
        mass += tile_mass;
        ie   += tile_ie;
        temp += tile_temp;
    }

    
    // global mPI Reduction
    tea_sum(vol);
    tea_sum(mass);
    tea_sum(ie);
    tea_sum(temp);

    if (profiler_on) {
        profiler.summary += (MPI_Wtime() - start_time);
    }

    if (parallel.boss) {
        double density_avg = (vol > 0.0) ? (mass / vol) : 0.0;
        
        // fortran: (a6,i7,5e26.17)
        *g_out << " step: " << std::setw(7) << step
               << std::scientific << std::setprecision(17)
               << std::setw(26) << vol
               << std::setw(26) << mass
               << std::setw(26) << density_avg
               << std::setw(26) << ie
               << std::setw(26) << temp 
               << "\n\n";
        
        g_out->flush();
    }

    if (complete) {
        if (parallel.boss && test_problem >= 1) {
            double qa_diff = 0.0;

            // Note: Constants taken directly from Fortran reference
            if (test_problem == 1) qa_diff = std::abs((100.0 * (temp / 157.55084183279294)) - 100.0);
            if (test_problem == 2) qa_diff = std::abs((100.0 * (temp / 106.27221178646569)) - 100.0);
            if (test_problem == 3) qa_diff = std::abs((100.0 * (temp / 99.955877498324000)) - 100.0);
            if (test_problem == 4) qa_diff = std::abs((100.0 * (temp / 97.277332050749976)) - 100.0);
            if (test_problem == 5) qa_diff = std::abs((100.0 * (temp / 95.462351583362249)) - 100.0);
            if (test_problem == 6) qa_diff = std::abs((100.0 * (temp / 95.174738768320850)) - 100.0);

            std::cout << "Test problem " << test_problem 
                      << " is within " << std::scientific << std::setprecision(7) << qa_diff 
                      << "% of the expected solution\n";
            
            *g_out << "Test problem " << test_problem 
                   << " is within " << std::scientific << std::setprecision(7) << qa_diff 
                   << "% of the expected solution\n";

            if (qa_diff < 0.001) {
                std::cout << "This test is considered PASSED" << std::endl;
                *g_out    << "This test is considered PASSED" << std::endl;
            } else {
                std::cout << "This test is considered NOT PASSED" << std::endl;
                *g_out    << "This test is considered NOT PASSED" << std::endl;
            }
        }
    }
}