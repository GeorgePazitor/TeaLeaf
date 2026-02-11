#include "tea.h"
#include "data.h"
#include "definitions.h"
#include "generate_chunk_kernel.h"
#include <vector>
#include <omp.h>

void generate_chunk() {
    using namespace TeaLeaf;
    
    // The kernel expects structure of arrays (SoA).
    // We also convert 1-based global states to 0-based kernel Arrays.
    // global states[1] (Background) -> local_arrays[0]
    
    int num_states = number_of_states; 

    std::vector<double> state_dens(num_states);
    std::vector<double> state_ener(num_states);
    std::vector<double> state_x_min(num_states);
    std::vector<double> state_x_max(num_states);
    std::vector<double> state_y_min(num_states);
    std::vector<double> state_y_max(num_states);
    std::vector<double> state_rad(num_states);
    std::vector<int>    state_geo(num_states);

    for (int s = 0; s < num_states; ++s) {
        // Shift: s=0 fills from states[1]
        int global_idx = s + 1; 

        state_dens[s]  = states[global_idx].density;
        state_ener[s]  = states[global_idx].energy;
        state_x_min[s] = states[global_idx].xmin;
        state_x_max[s] = states[global_idx].xmax;
        state_y_min[s] = states[global_idx].ymin;
        state_y_max[s] = states[global_idx].ymax;
        state_rad[s]   = states[global_idx].radius;
        state_geo[s]   = states[global_idx].geometry;
    }


    #pragma omp parallel for
    for (int t = 0; t < tiles_per_task; ++t) {
        
        // Alias for cleaner code
        auto& tile = chunk.tiles[t];

        generate_chunk_kernel(
            // Field Bounds
            tile.field.x_min,
            tile.field.x_max,
            tile.field.y_min,
            tile.field.y_max,
            chunk.halo_exchange_depth,
            
            // Geometry Arrays (Pointers)
            tile.field.vertexx.data(),
            tile.field.vertexy.data(),
            tile.field.cellx.data(),
            tile.field.celly.data(),
            
            // Field Arrays to be populated (Pointers)
            tile.field.density.data(),
            tile.field.energy0.data(),
            tile.field.u.data(), // Note: Maps to 'u0' argument in kernel
            
            // State Data (Pointers to our local flattened vectors)
            num_states, 
            state_dens.data(),
            state_ener.data(),
            state_x_min.data(),
            state_x_max.data(),
            state_y_min.data(),
            state_y_max.data(),
            state_rad.data(),
            state_geo.data(),
            
            // Constants
            g_rect,
            g_circ,
            g_point
        );
    }
}