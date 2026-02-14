#include "include/kernels/initialise_chunk_kernel.h"
#include <omp.h>
#include <algorithm>

/**
 * Initializes the geometric properties of a grid chunk.
 * This includes defining vertex positions, cell centers, and computing
 * the volumes and face areas used in the finite volume discretization.
 */
void initialise_chunk_kernel(
    int x_min, int x_max, 
    int y_min, int y_max,
    double xmin, double ymin, 
    double dx, double dy,
    double* vertexx, double* vertexdx,
    double* vertexy, double* vertexdy,
    double* cellx, double* celldx,
    double* celly, double* celldy,
    double* volume, double* xarea, double* yarea) 
{
    // Exact width calculations to mirror the Fortran-style padded indexing.
    // 'vol_width' and 'yarea_width' cover -2 to +2 (size + 4)
    // 'xarea_width' covers -2 to +3 (size + 5) to account for the right-hand face.
    int vol_width   = (x_max + 2) - (x_min - 2) + 1; 
    int xarea_width = (x_max + 3) - (x_min - 2) + 1; 
    int yarea_width = (x_max + 2) - (x_min - 2) + 1; 

    #pragma omp parallel
    {
        // Vertex Coordinates
        // Vertices define the corners of the cells.
        #pragma omp for nowait
        for (int j = x_min - 2; j <= x_max + 3; ++j) {
            int idx = j - (x_min - 2);
            vertexx[idx]  = xmin + dx * (double)(j - x_min);
            vertexdx[idx] = dx;
        }

        #pragma omp for
        for (int k = y_min - 2; k <= y_max + 3; ++k) {
            int idx = k - (y_min - 2);
            vertexy[idx]  = ymin + dy * (double)(k - y_min);
            vertexdy[idx] = dy;
        }

        // Cell Center Coordinates (1D)
        // Cells are located halfway between vertices.
        #pragma omp for nowait
        for (int j = x_min - 2; j <= x_max + 2; ++j) {
            int idx = j - (x_min - 2);
            cellx[idx]  = 0.5 * (vertexx[idx] + vertexx[idx + 1]);
            celldx[idx] = dx;
        }

        #pragma omp for
        for (int k = y_min - 2; k <= y_max + 2; ++k) {
            int idx = k - (y_min - 2);
            celly[idx]  = 0.5 * (vertexy[idx] + vertexy[idx + 1]);
            celldy[idx] = dy;
        }

        // 2D Metric Fields (Volumes and Y-Areas)
        // 'volume' is the 2D area (dx*dy) of a cell. 
        // 'yarea' represents the length of the horizontal faces (dx).
        
        #pragma omp for
        for (int k = y_min - 2; k <= y_max + 2; ++k) {
            int k_idx = k - (y_min - 2);
            for (int j = x_min - 2; j <= x_max + 2; ++j) {
                int j_idx = j - (x_min - 2);
                
                volume[k_idx * vol_width + j_idx] = dx * dy;
                yarea[k_idx * yarea_width + j_idx] = dx;
            }
        }

        // 2D Metric Fields (X-Areas)
        // 'xarea' represents the length of the vertical faces (dy).
        // It uses a wider J-range (x_max + 3) to include the boundary face on the right.
        #pragma omp for
        for (int k = y_min - 2; k <= y_max + 2; ++k) {
            int k_idx = k - (y_min - 2);
            for (int j = x_min - 2; j <= x_max + 3; ++j) {
                int j_idx = j - (x_min - 2);
                xarea[k_idx * xarea_width + j_idx] = dy;
            }
        }
    }
}