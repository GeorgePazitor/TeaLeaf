#include "include/kernels/update_halo_kernel.h"
#include "include/data.h" 
#include "include/definitions.h"
#include <omp.h>

using namespace TeaLeaf;

/**
 * Handles the boundary conditions for the external faces of the domain.
 * If a tile face is on the global boundary (EXTERNAL_FACE), this routine 
 * implements "reflective" boundary conditions by mirroring the interior 
 * values into the halo (ghost) zones.
 */
void update_halo_cell(
    int x_min, int x_max, int y_min, int y_max,
    int halo_exchange_depth,
    const std::array<int, 4>& chunk_neighbours,
    const std::array<int, 4>& tile_neighbours,
    double* mesh, int depth)
{
    // Local stride calculation to map 2D coordinates to 1D memory layout
    int width = (x_max - x_min + 1) + 2 * halo_exchange_depth;
    #define IDX(x,y) ((y+halo_exchange_depth)*width + (x+halo_exchange_depth))

    // Left/Right Reflective Boundaries
    // If both the MPI chunk and the local tile identify this as a global edge
    if(chunk_neighbours[CHUNK_LEFT]==EXTERNAL_FACE && tile_neighbours[CHUNK_LEFT]==EXTERNAL_FACE){
        #pragma omp for collapse(2) nowait
        for(int k=y_min-depth; k<=y_max+depth; ++k)
            for(int j=1; j<=depth; ++j)
                // Mirror the interior cell value (x_min+j-1) to the halo (x_min-j)
                mesh[IDX(x_min-j,k)] = mesh[IDX(x_min+j-1,k)];
    }
    
    if(chunk_neighbours[CHUNK_RIGHT]==EXTERNAL_FACE && tile_neighbours[CHUNK_RIGHT]==EXTERNAL_FACE){
        #pragma omp for collapse(2) nowait
        for(int k=y_min-depth; k<=y_max+depth; ++k)
            for(int j=1; j<=depth; ++j)
                mesh[IDX(x_max+j,k)] = mesh[IDX(x_max-j+1,k)];
    }

    // Ensure all threads finish Horizontal reflections before starting Vertical 
    // to correctly populate corner ghost zones.
    #pragma omp barrier

    // Top/Bottom Reflective Boundaries
    if(chunk_neighbours[CHUNK_BOTTOM]==EXTERNAL_FACE && tile_neighbours[CHUNK_BOTTOM]==EXTERNAL_FACE){
        #pragma omp for collapse(2) nowait
        for(int k=1; k<=depth; ++k)
            for(int j=x_min-depth; j<=x_max+depth; ++j)
                mesh[IDX(j,y_min-k)] = mesh[IDX(j,y_min+k-1)];
    }
    
    if(chunk_neighbours[CHUNK_TOP]==EXTERNAL_FACE && tile_neighbours[CHUNK_TOP]==EXTERNAL_FACE){
        #pragma omp for collapse(2) nowait
        for(int k=1; k<=depth; ++k)
            for(int j=x_min-depth; j<=x_max+depth; ++j)
                mesh[IDX(j,y_max+k)] = mesh[IDX(j,y_max-k+1)];
    }

    #undef IDX
}

/**
 * Kernel wrapper to dispatch boundary updates for all active simulation fields.
 * Encapsulates the OpenMP parallel region to minimize fork/join overhead.
 */
void update_halo_kernel(
    int x_min, int x_max, int y_min, int y_max,
    int halo_exchange_depth,
    const std::array<int, 4>& chunk_neighbours,
    const std::array<int, 4>& tile_neighbours,
    double* density, double* energy0, double* energy1, double* u, double* p,
    double* sd, double* r, double* z, double* kx, double* ky, double* di,
    const int* fields, int depth)
{
    #pragma omp parallel
    {
        // For each field, if the solver requires it, perform the reflection.
        // Each call contains its own #pragma omp for loops.
        if(fields[FIELD_DENSITY]) update_halo_cell(x_min,x_max,y_min,y_max,halo_exchange_depth,chunk_neighbours,tile_neighbours,density,depth);
        if(fields[FIELD_ENERGY0]) update_halo_cell(x_min,x_max,y_min,y_max,halo_exchange_depth,chunk_neighbours,tile_neighbours,energy0,depth);
        if(fields[FIELD_ENERGY1]) update_halo_cell(x_min,x_max,y_min,y_max,halo_exchange_depth,chunk_neighbours,tile_neighbours,energy1,depth);
        if(fields[FIELD_U])       update_halo_cell(x_min,x_max,y_min,y_max,halo_exchange_depth,chunk_neighbours,tile_neighbours,u,depth);
        if(fields[FIELD_P])       update_halo_cell(x_min,x_max,y_min,y_max,halo_exchange_depth,chunk_neighbours,tile_neighbours,p,depth);
        if(fields[FIELD_SD])      update_halo_cell(x_min,x_max,y_min,y_max,halo_exchange_depth,chunk_neighbours,tile_neighbours,sd,depth);
        if(fields[FIELD_R])       update_halo_cell(x_min,x_max,y_min,y_max,halo_exchange_depth,chunk_neighbours,tile_neighbours,r,depth);
        if(fields[FIELD_Z])       update_halo_cell(x_min,x_max,y_min,y_max,halo_exchange_depth,chunk_neighbours,tile_neighbours,z,depth);
        if(fields[FIELD_KX])      update_halo_cell(x_min,x_max,y_min,y_max,halo_exchange_depth,chunk_neighbours,tile_neighbours,kx,depth);
        if(fields[FIELD_KY])      update_halo_cell(x_min,x_max,y_min,y_max,halo_exchange_depth,chunk_neighbours,tile_neighbours,ky,depth);
        if(fields[FIELD_DI])      update_halo_cell(x_min,x_max,y_min,y_max,halo_exchange_depth,chunk_neighbours,tile_neighbours,di,depth);
    }
}