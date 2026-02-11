#include "initialise_chunk_kernel.h"
#include <omp.h>
#include <cstdio>

void initialise_chunk_kernel(
    int x_min, int x_max, 
    int y_min, int y_max,
    double xmin, double ymin, 
    double dx, double dy,
    double* vertexx, 
    double* vertexdx,
    double* vertexy, 
    double* vertexdy,
    double* cellx, 
    double* celldx,
    double* celly, 
    double* celldy,
    double* volume,
    double* xarea, 
    double* yarea) 
{
    //calculate offsets and strides
    
    // In Fortran, arrays have negative indices (e.g. x_min-2).
    // In C++, index 0 corresponds to that lowest bound.
    // We calculate "offsets" so that: C_Index = Loop_Counter - Offset
    
    int x_off_2 = x_min - 2; // Offset for arrays starting at x_min-2
    int y_off_2 = y_min - 2; // Offset for arrays starting at y_min-2

    int vol_width   = (x_max + 2) - (x_min - 2) + 1; // For volume (ends at x_max+2)
    int xarea_width = (x_max + 3) - (x_min - 2) + 1; // For xarea (ends at x_max+3)
    int yarea_width = (x_max + 2) - (x_min - 2) + 1; // For yarea (ends at x_max+2)

    // ------------------------------------------------------------------------
    // 2. Parallel Region
    // ------------------------------------------------------------------------
    #pragma omp parallel
    {
        // --- Vertex X & DX ---
        // Range: x_min-2 to x_max+3
        #pragma omp for nowait
        for (int j = x_min - 2; j <= x_max + 3; ++j) {
            int idx = j - x_off_2; 
            vertexx[idx]  = xmin + dx * (double)(j - x_min);
            vertexdx[idx] = dx;
        }

        // --- Vertex Y & DY ---
        // Range: y_min-2 to y_max+3
        #pragma omp for
        for (int k = y_min - 2; k <= y_max + 3; ++k) {
            int idx = k - y_off_2;
            vertexy[idx]  = ymin + dy * (double)(k - y_min);
            vertexdy[idx] = dy;
        } 
        // Note: Implicit barrier here ensures vertices are ready for next steps

        // --- Cell X & DX ---
        // Range: x_min-2 to x_max+2
        #pragma omp for nowait
        for (int j = x_min - 2; j <= x_max + 2; ++j) {
            int idx = j - x_off_2;
            // Uses vertexx(j) and vertexx(j+1)
            cellx[idx]  = 0.5 * (vertexx[idx] + vertexx[idx + 1]);
            celldx[idx] = dx;
        }

        // --- Cell Y & DY ---
        // Range: y_min-2 to y_max+2
        #pragma omp for
        for (int k = y_min - 2; k <= y_max + 2; ++k) {
            int idx = k - y_off_2;
            // Uses vertexy(k) and vertexy(k+1)
            celly[idx]  = 0.5 * (vertexy[idx] + vertexy[idx + 1]);
            celldy[idx] = dy;
        }
        // Barrier here ensures cell dims are ready for Volume/Area

        // --- Volume ---
        // Range: X[-2, +2], Y[-2, +2]
        #pragma omp for nowait
        for (int k = y_min - 2; k <= y_max + 2; ++k) {
            int k_idx = k - y_off_2;
            for (int j = x_min - 2; j <= x_max + 2; ++j) {
                int j_idx = j - x_off_2;
                volume[k_idx * vol_width + j_idx] = dx * dy;
            }
        }

        // --- X Area ---
        // Range: X[-2, +2], Y[-2, +2] (Wait, check Fortran logic carefully)
        // Fortran: xarea loops j=x_min-2, x_max+2 ?? 
        // Actually, Fortran declaration is x_max+3 for xarea, but the loop 
        // in the snippet provided only goes to x_max+2.
        // We follow the provided loop logic precisely.
        #pragma omp for nowait
        for (int k = y_min - 2; k <= y_max + 2; ++k) {
            int k_idx = k - y_off_2;
            double cdy = celldy[k_idx]; // Pre-load to register
            
            for (int j = x_min - 2; j <= x_max + 2; ++j) {
                int j_idx = j - x_off_2;
                // Note: xarea stride is wider (x_max+3) even if loop is shorter
                xarea[k_idx * xarea_width + j_idx] = cdy;
            }
        }

        // --- Y Area ---
        // Range: X[-2, +2], Y[-2, +2]
        #pragma omp for
        for (int k = y_min - 2; k <= y_max + 2; ++k) {
            int k_idx = k - y_off_2;
            for (int j = x_min - 2; j <= x_max + 2; ++j) {
                int j_idx = j - x_off_2;
                yarea[k_idx * yarea_width + j_idx] = celldx[j_idx];
            }
        }

    } // End Parallel
}