#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>

#include "include/data.h"
#include "include/visit.h"
#include "include/global_mpi.h"
#include "include/definitions.h"
#include "include/tea.h"

namespace TeaLeaf {

/**
 * Generates visualization files compatible with VisIt.
 * It creates a master .visit file and individual .vtk files for each tile.
 */
void visit() {
    static bool first_call = true;
    std::string name = "tea";
    double kernel_time = 0.0;

    // Master .visit File Management
    // The boss (master) process creates/initializes the header for VisIt
    if (first_call) {
        if (parallel.boss) {
            int nblocks = parallel.max_task * tiles_per_task;
            std::ofstream visit_file("tea.visit", std::ios::out);
            if (visit_file.is_open()) {
                // Write the total number of blocks across all MPI ranks
                visit_file << "!NBLOCKS " << nblocks << std::endl;
                visit_file.close();
            }
        }
        first_call = false;
    }

    // Append the current step's VTK file list to the master .visit file
    if (parallel.boss) {
        std::ofstream visit_file("tea.visit", std::ios::app);
        if (visit_file.is_open()) {
            for (int c = 1; c <= parallel.max_task * tiles_per_task; ++c) {
                std::stringstream ss_chunk, ss_step;
                // Reproduce the Fortran index offset trick for filename formatting
                ss_chunk << std::setfill('0') << std::setw(6) << (c + 100000);
                ss_step << std::setfill('0') << std::setw(6) << (step + 100000);
                
                std::string chunk_n = ss_chunk.str();
                std::string step_n = ss_step.str();
                chunk_n[0] = '.'; // Replace leading '1' with '.'
                step_n[0] = '.';

                // Record the filename for this specific block and timestep
                visit_file << name << chunk_n << step_n << ".vtk" << std::endl;
            }
            visit_file.close();
        }
    }

    // Individual VTK File Generation
    if (profiler_on) kernel_time = timer();

    // Loop through each tile assigned to this MPI task
    for (int c = 0; c < tiles_per_task; ++c) {
        auto& tile = chunk.tiles[c];
        auto& f = tile.field;

        // Calculate dimensions: nxc/nyc for cells, nxv/nyv for vertices
        int nxc = f.x_max - f.x_min + 1;
        int nyc = f.y_max - f.y_min + 1;
        int nxv = nxc + 1;
        int nyv = nyc + 1;

        // Construct the unique .vtk filename for this tile and step
        std::stringstream ss_chunk, ss_step;
        ss_chunk << std::setfill('0') << std::setw(6) << (parallel.task * tiles_per_task + c + 100001);
        ss_step << std::setfill('0') << std::setw(6) << (step + 100000);
        
        std::string chunk_n = ss_chunk.str();
        std::string step_n = ss_step.str();
        chunk_n[0] = '.';
        step_n[0] = '.';
        std::string filename = name + chunk_n + step_n + ".vtk";

        std::ofstream vtk(filename);
        if (!vtk.is_open()) continue;

        // VTK Header Information
        vtk << "# vtk DataFile Version 3.0" << "\n";
        vtk << "vtk output" << "\n";
        vtk << "ASCII" << "\n";
        vtk << "DATASET RECTILINEAR_GRID" << "\n";
        vtk << "DIMENSIONS " << nxv << " " << nyv << " 1" << "\n";

        // X Coordinates (Vertex based)
        vtk << "X_COORDINATES " << nxv << " double" << "\n";
        vtk << std::scientific << std::setprecision(4);
        for (int j = f.x_min; j <= f.x_max + 1; ++j) {
            vtk << f.vertexx[j] << "\n";
        }

        // Y Coordinates (Vertex based)
        vtk << "Y_COORDINATES " << nyv << " double" << "\n";
        for (int k = f.y_min; k <= f.y_max + 1; ++k) {
            vtk << f.vertexy[k] << "\n";
        }

        // Z Coordinates (Constant for 2D)
        vtk << "Z_COORDINATES 1 double" << "\n";
        vtk << "0" << "\n";

        // Cell Data Section: Map grid fields to the VTK file
        int ncells = nxc * nyc;
        vtk << "CELL_DATA " << ncells << "\n";
        vtk << "FIELD FieldData 3" << "\n";

        // DENSITY FIELD
        vtk << "density 1 " << ncells << " double" << "\n";
        for (int k = f.y_min; k <= f.y_max; ++k) {
            for (int j = f.x_min; j <= f.x_max; ++j) {
                int idx = (k - f.y_min) * nxc + (j - f.x_min);
                vtk << f.density[idx] << (j == f.x_max ? "" : " ");
            }
            vtk << "\n";
        }

        // ENERGY FIELD (energy0)
        vtk << "energy 1 " << ncells << " double" << "\n";
        for (int k = f.y_min; k <= f.y_max; ++k) {
            for (int j = f.x_min; j <= f.x_max; ++j) {
                int idx = (k - f.y_min) * nxc + (j - f.x_min);
                vtk << f.energy0[idx] << (j == f.x_max ? "" : " ");
            }
            vtk << "\n";
        }

        // TEMPERATURE FIELD (u)
        vtk << "temperature 1 " << ncells << " double" << "\n";
        for (int k = f.y_min; k <= f.y_max; ++k) {
            for (int j = f.x_min; j <= f.x_max; ++j) {
                int idx = (k - f.y_min) * nxc + (j - f.x_min);
                vtk << f.u[idx] << (j == f.x_max ? "" : " ");
            }
            vtk << "\n";
        }

        vtk.close();
    }

    if (profiler_on) {
        profiler.visit += (timer() - kernel_time);
    }
}

} // namespace TeaLeaf