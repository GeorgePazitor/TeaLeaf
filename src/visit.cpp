#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>

#include "data.h"
#include "visit.h"
#include "global_mpi.h"
#include "definitions.h"
#include "tea.h"

namespace TeaLeaf {

void visit() {
    static bool first_call = true;
    std::string name = "tea";
    double kernel_time;

    // --- 1. Gestion du fichier Maître .visit ---
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
                // Reproduit l'astuce Fortran du décalage d'index pour le nommage
                ss_chunk << std::setfill('0') << std::setw(6) << (c + 100000);
                ss_step << std::setfill('0') << std::setw(6) << (step + 100000);
                
                std::string chunk_n = ss_chunk.str();
                std::string step_n = ss_step.str();
                chunk_n[0] = '.'; // Remplace le '1' par '.'
                step_n[0] = '.';

                visit_file << name << chunk_n << step_n << ".vtk" << std::endl;
            }
            visit_file.close();
        }
    }

    // --- 2. Génération des fichiers VTK individuels ---
    if (profiler_on) kernel_time = timer();

    for (int c = 0; c < tiles_per_task; ++c) {
        auto& tile = chunk.tiles[c];
        auto& f = tile.field;

        int nxc = f.x_max - f.x_min + 1;
        int nyc = f.y_max - f.y_min + 1;
        int nxv = nxc + 1;
        int nyv = nyc + 1;

        // Construction du nom de fichier .vtk
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

        // Entête VTK
        vtk << "# vtk DataFile Version 3.0" << "\n";
        vtk << "vtk output" << "\n";
        vtk << "ASCII" << "\n";
        vtk << "DATASET RECTILINEAR_GRID" << "\n";
        vtk << "DIMENSIONS " << nxv << " " << nyv << " 1" << "\n";

        // Coordonnées X (Vertex)
        vtk << "X_COORDINATES " << nxv << " double" << "\n";
        vtk << std::scientific << std::setprecision(4);
        for (int j = f.x_min; j <= f.x_max + 1; ++j) {
            vtk << f.vertexx[j] << "\n";
        }

        // Coordonnées Y (Vertex)
        vtk << "Y_COORDINATES " << nyv << " double" << "\n";
        for (int k = f.y_min; k <= f.y_max + 1; ++k) {
            vtk << f.vertexy[k] << "\n";
        }

        // Coordonnées Z
        vtk << "Z_COORDINATES 1 double" << "\n";
        vtk << "0" << "\n";

        // Données des mailles (Cell Data)
        int ncells = nxc * nyc;
        vtk << "CELL_DATA " << ncells << "\n";
        vtk << "FIELD FieldData 3" << "\n";

        // 1. DENSITÉ
        vtk << "density 1 " << ncells << " double" << "\n";
        for (int k = f.y_min; k <= f.y_max; ++k) {
            for (int j = f.x_min; j <= f.x_max; ++j) {
                int idx = (k - f.y_min) * nxc + (j - f.x_min);
                vtk << f.density[idx] << (j == f.x_max ? "" : " ");
            }
            vtk << "\n";
        }

        // 2. ÉNERGIE (energy0)
        vtk << "energy 1 " << ncells << " double" << "\n";
        for (int k = f.y_min; k <= f.y_max; ++k) {
            for (int j = f.x_min; j <= f.x_max; ++j) {
                int idx = (k - f.y_min) * nxc + (j - f.x_min);
                vtk << f.energy0[idx] << (j == f.x_max ? "" : " ");
            }
            vtk << "\n";
        }

        // 3. TEMPÉRATURE (u)
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