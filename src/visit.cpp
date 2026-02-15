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
 * Generates visualization files compatible with VisIt and ParaView.
 * It creates a master .visit file and individual .vtk files for each tile.
 */
void visit() {
    static bool first_call = true;
    std::string name = "tea";
    double kernel_time = 0.0;

    if (first_call) {
        if (parallel.boss) {
            int nblocks = parallel.max_task * tiles_per_task;
            std::ofstream visit_file("tea.visit", std::ios::out);
            if (visit_file.is_open()) {
                visit_file << "!NBLOCKS " << nblocks << std::endl;
                visit_file.close();
            }
        }
        first_call = false;
    }

    if (parallel.boss) {
        std::ofstream visit_file("tea.visit", std::ios::app);
        if (visit_file.is_open()) {
            for (int c = 1; c <= parallel.max_task * tiles_per_task; ++c) {
                std::stringstream ss_chunk, ss_step;

                ss_chunk << std::setfill('0') << std::setw(6) << (c + 100000);
                ss_step << std::setfill('0') << std::setw(6) << (step + 100000);
                
                std::string chunk_n = ss_chunk.str();
                std::string step_n = ss_step.str();
                chunk_n[0] = '.'; 
                step_n[0] = '.';

                visit_file << name << chunk_n << step_n << ".vtk" << std::endl;
            }
            visit_file.close();
        }
    }

    if (profiler_on) kernel_time = timer();

    for (int c = 0; c < tiles_per_task; ++c) {
        auto& tile = chunk.tiles[c];
        auto& f = tile.field;
        
        int hd = chunk.halo_exchange_depth;

        int nxc = f.x_max - f.x_min + 1;
        int nyc = f.y_max - f.y_min + 1;
        int nxv = nxc + 1;
        int nyv = nyc + 1;

        //construct the unique .vtk filename for this tile and step
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

        vtk << "# vtk DataFile Version 3.0" << "\n";
        vtk << "vtk output" << "\n";
        vtk << "ASCII" << "\n";
        vtk << "DATASET RECTILINEAR_GRID" << "\n";
        vtk << "DIMENSIONS " << nxv << " " << nyv << " 1" << "\n";

        
        int field_width = (f.x_max + hd) - (f.x_min - hd) + 1;
        
        #define FIELD_IDX(j, k) ((k - (f.y_min - hd)) * field_width + (j - (f.x_min - hd)))
        #define V_IDX(p, p_min) (p - (p_min - 2))

        vtk << "X_COORDINATES " << nxv << " double" << "\n";
        vtk << std::scientific << std::setprecision(4);
        for (int j = f.x_min; j <= f.x_max + 1; ++j) {
            vtk << f.vertexx[V_IDX(j, f.x_min)] << "\n";
        }

        vtk << "Y_COORDINATES " << nyv << " double" << "\n";
        for (int k = f.y_min; k <= f.y_max + 1; ++k) {
            vtk << f.vertexy[V_IDX(k, f.y_min)] << "\n";
        }

        vtk << "Z_COORDINATES 1 double" << "\n";
        vtk << "0" << "\n";

        int ncells = nxc * nyc;
        vtk << "CELL_DATA " << ncells << "\n";
        vtk << "FIELD FieldData 3" << "\n";

        vtk << "density 1 " << ncells << " double" << "\n";
        for (int k = f.y_min; k <= f.y_max; ++k) {
            for (int j = f.x_min; j <= f.x_max; ++j) {
                vtk << f.density[FIELD_IDX(j, k)] << (j == f.x_max ? "" : " ");
            }
            vtk << "\n";
        }

        vtk << "energy 1 " << ncells << " double" << "\n";
        for (int k = f.y_min; k <= f.y_max; ++k) {
            for (int j = f.x_min; j <= f.x_max; ++j) {
                vtk << f.energy0[FIELD_IDX(j, k)] << (j == f.x_max ? "" : " ");
            }
            vtk << "\n";
        }

        vtk << "temperature 1 " << ncells << " double" << "\n";
        for (int k = f.y_min; k <= f.y_max; ++k) {
            for (int j = f.x_min; j <= f.x_max; ++j) {
                vtk << f.u[FIELD_IDX(j, k)] << (j == f.x_max ? "" : " ");
            }
            vtk << "\n";
        }

        #undef FIELD_IDX
        #undef V_IDX

        vtk.close();
    }

    if (profiler_on) {
        profiler.visit += (timer() - kernel_time);
    }
}

} // namespace TeaLeaf