#include "include/tea.h"
#include "include/data.h"
#include "include/definitions.h"
#include "include/kernels/generate_chunk_kernel.h"
#include <vector>
#include <omp.h>
#include <algorithm>

/**
 * Populates the grid with physical properties (density, energy, temperature) 
 * based on the geometric states defined in the input file.
 */
void generate_chunk() {
    using namespace TeaLeaf;
    
    // Data Preparation
    // Flatten state structures into contiguous vectors for efficient kernel access.
    // This mapping converts the OO-style 'states' vector into arrays of primitives.
    int num_states = number_of_states; 
    std::vector<double> state_dens(num_states + 1), state_ener(num_states + 1);
    std::vector<double> state_x_min(num_states + 1), state_x_max(num_states + 1);
    std::vector<double> state_y_min(num_states + 1), state_y_max(num_states + 1);
    std::vector<double> state_rad(num_states + 1);
    std::vector<int>    state_geo(num_states + 1);

    for (int s = 1; s <= num_states; ++s) {
        state_dens[s]  = states[s].density;
        state_ener[s]  = states[s].energy;
        state_x_min[s] = states[s].xmin;
        state_x_max[s] = states[s].xmax;
        state_y_min[s] = states[s].ymin;
        state_y_max[s] = states[s].ymax;
        state_rad[s]   = states[s].radius;
        state_geo[s]   = states[s].geometry;
    }

    // Parallel Field Generation
    // Process each tile in parallel to determine which state occupies which cell.
    #pragma omp parallel for
    for (int t = 0; t < tiles_per_task; ++t) {
        auto& tile = chunk.tiles[t];
        
        // The kernel evaluates geometric tests (rectangle, circle, point) 
        // for every cell in the tile.
        generate_chunk_kernel(
            tile.field.x_min, tile.field.x_max, 
            tile.field.y_min, tile.field.y_max, 
            chunk.halo_exchange_depth,
            tile.field.vertexx.data(), tile.field.vertexy.data(),
            tile.field.cellx.data(), tile.field.celly.data(),
            tile.field.density.data(), tile.field.energy0.data(),
            tile.field.u0.data(),
            num_states, 
            state_dens.data(), state_ener.data(),
            state_x_min.data(), state_x_max.data(),
            state_y_min.data(), state_y_max.data(),
            state_rad.data(), state_geo.data(),
            g_rect, g_circ, g_point
        );

        // Field Synchronization
        // Ensure the initial temperature field 'u' matches 'u0' before the solver starts.
        // This is a critical step for the first iteration of the heat equation.
        std::copy(tile.field.u0.begin(), tile.field.u0.end(), tile.field.u.begin());
    }
}