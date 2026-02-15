#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <omp.h>
#include <chrono>

#include "include/diffuse.h"
#include "include/timestep.h"
#include "include/tea_solve.h"
#include "include/data.h"
#include "include/definitions.h"
#include "include/field_summary.h"
#include "include/visit.h"
#include "include/global_mpi.h"
#include "include/tea.h"

namespace TeaLeaf {

// The main time-stepping loop. Controls the simulation flow, output frequency, and performance profiling.
void diffuse() {
    int loc_idx = 0;
    double timer_start, wall_clock, step_clock;
    double grind_time, cells, rstep;
    double step_time, step_grind;
    double first_step = 0.0, second_step = 0.0;
    double kernel_total;
    
    ///collects profiling data from all MPI tasks
    std::vector<double> totals(parallel.max_task);

    timer_start = timer(); 
    second_step = 0.0;

    while (true) {
        step_time = timer();
        step++;

        timestep();    
        
        // rxecute the diffusion solver 
        tea_leaf();    
        
        timee += dt;

        //reporting and visualization
        if (summary_frequency != 0 && (step % summary_frequency == 0)) {
            field_summary();
        }
        if (visit_frequency != 0 && (step % visit_frequency == 0)) {
            visit();
        }

        // measure overhead 
        if (step == 1) first_step = timer() - step_time;
        if (step == 2) second_step = timer() - step_time;

        if (parallel.boss) {
            wall_clock = timer() - timer_start;
            step_clock = timer() - step_time;
            
            cells = static_cast<double>(grid.x_cells) * grid.y_cells;
            rstep = static_cast<double>(step);
            grind_time = wall_clock / (rstep * cells);
            step_grind = step_clock / cells;

            *g_out << "Step " << step << " time " << step_clock  << std::endl;
        }

        if (timee + g_small > end_time || step >= end_step) {
            complete = true;
            field_summary();
            if (visit_frequency != 0) visit();

            wall_clock = timer() - timer_start;
            if (parallel.boss) {
                *g_out << "\nCalculation complete\nTea is finishing" << std::endl;
                *g_out << "First step overhead " << (first_step - second_step) << std::endl;
                *g_out << "Wall clock " << wall_clock << std::endl;
            }
            break; 
        }
    }

    if (profiler_on) {
        kernel_total = profiler.timestep + profiler.halo_exchange + profiler.summary + 
                       profiler.visit + profiler.tea_init + profiler.set_field + 
                       profiler.tea_solve + profiler.tea_reset + profiler.dot_product + 
                       profiler.halo_update + profiler.internal_halo_update;

        tea_allgather(kernel_total, totals);

        //find the index of the rank that spent the most time (max load)
        auto it = std::max_element(totals.begin(), totals.begin() + parallel.max_task);
        loc_idx = std::distance(totals.begin(), it);
        kernel_total = totals[loc_idx];

        //sync all metrics based on the slowest rank's data
        auto sync_metric = [&](double &metric) {
            tea_allgather(metric, totals);
            metric = totals[loc_idx];
        };

        sync_metric(profiler.timestep);
        sync_metric(profiler.halo_exchange);
        sync_metric(profiler.internal_halo_update);
        sync_metric(profiler.halo_update);
        sync_metric(profiler.summary);
        sync_metric(profiler.dot_product);
        sync_metric(profiler.visit);
        sync_metric(profiler.tea_init);
        sync_metric(profiler.tea_solve);
        sync_metric(profiler.tea_reset);
        sync_metric(profiler.set_field);

        if (parallel.boss) {
            auto print_line = [&](std::string label, double val) {
                std::cout << std::left << std::setw(23) << label << ":" 
                          << std::fixed << std::setprecision(4) << std::right 
                          << std::setw(16) << val 
                          << std::setw(16) << (100.0 * val / wall_clock) << "%" << std::endl;
            };

            std::cout << "\nProfiler Output (Slowest Rank " << loc_idx << ")" << std::endl;
            std::cout << "------------------------------------------------------------" << std::endl;
            std::cout << "Kernel                  |      Time (s)    |   Percentage" << std::endl;
            print_line("Timestep", profiler.timestep);
            print_line("MPI Halo Exchange", profiler.halo_exchange);
            print_line("Self Halo Update", profiler.halo_update);
            print_line("Internal Halo Update", profiler.internal_halo_update);
            print_line("Summary", profiler.summary);
            print_line("Visit", profiler.visit);
            print_line("Tea Init", profiler.tea_init);
            print_line("Dot Product", profiler.dot_product);
            print_line("Tea Solve", profiler.tea_solve);
            print_line("Tea Reset", profiler.tea_reset);
            print_line("Set Field", profiler.set_field);
            print_line("Total", kernel_total);
            print_line("System/Misc", wall_clock - kernel_total);
        }
    }

    tea_finalize();
}

} // namespace TeaLeaf