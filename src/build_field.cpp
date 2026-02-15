#include "include/build_field.h"

#include "include/tea.h"
#include "include/data.h"
#include "include/definitions.h"
#include <vector>
#include <algorithm>
#include <omp.h>

using namespace TeaLeaf;

//allocates and initializes all memory buffers for the simulation fields  
void build_field() {
    
    #pragma omp parallel for
    for (int t = 0; t < tiles_per_task; ++t) {
        
        auto& tile = chunk.tiles[t];
        auto& field = tile.field;
        int depth = chunk.halo_exchange_depth;

        field.x_min = 0;
        field.y_min = 0;
        field.x_max = tile.x_cells - 1; //inclusive upper bound
        field.y_max = tile.y_cells - 1;

        // main physical fields require extra space around the edges for MPI communication (halos)
        int main_width  = tile.x_cells + 2 * depth;
        int main_height = tile.y_cells + 2 * depth;
        size_t main_size = (size_t)main_width * main_height;

        field.density.resize(main_size, 0.0);
        field.energy0.resize(main_size, 0.0);
        field.energy1.resize(main_size, 0.0);
        field.u.resize(main_size, 0.0);
        field.u0.resize(main_size, 0.0);
        
        //solver specific vectors 
        field.vector_p.resize(main_size, 0.0);
        field.vector_r.resize(main_size, 0.0);
        field.vector_r_store.resize(main_size, 0.0);
        field.vector_Mi.resize(main_size, 0.0);
        field.vector_w.resize(main_size, 0.0);
        field.vector_z.resize(main_size, 0.0);
        field.vector_utemp.resize(main_size, 0.0);
        field.vector_rtemp.resize(main_size, 0.0);
        field.vector_Di.resize(main_size, 0.0);
        field.vector_Kx.resize(main_size, 0.0);
        field.vector_Ky.resize(main_size, 0.0);
        field.vector_sd.resize(main_size, 0.0);
        field.row_sums.resize(main_size, 0.0);

        //solver internal fields no halo for temporary calculations during matrix inversions
        size_t tri_size = (size_t)tile.x_cells * tile.y_cells;
        field.tri_cp.resize(tri_size, 0.0);
        field.tri_bfp.resize(tri_size, 0.0);

        //geometric and coordinate fields
        //specific padding to safely handle stencil operations that reach beyond cell centers (-2, +2 or -2, +3 offsets).
        int pad_x_4 = tile.x_cells + 4; // padding for cell-centered geometry
        int pad_y_4 = tile.y_cells + 4; 
        
        int pad_x_5 = tile.x_cells + 5; // padding for vertex / face-centered geometry
        int pad_y_5 = tile.y_cells + 5; 

        field.cellx.resize(pad_x_4, 0.0);
        field.celldx.resize(pad_x_4, 0.0);
        field.celly.resize(pad_y_4, 0.0);
        field.celldy.resize(pad_y_4, 0.0);

        field.vertexx.resize(pad_x_5, 0.0);
        field.vertexdx.resize(pad_x_5, 0.0);
        field.vertexy.resize(pad_y_5, 0.0);
        field.vertexdy.resize(pad_y_5, 0.0);

        //volume for cells
        field.volume.resize((size_t)pad_x_4 * pad_y_4, 0.0);

        //face areas, xarea is defined on vertical faces, yarea on horizontal faces
        field.xarea.resize((size_t)pad_x_5 * pad_y_4, 0.0);
        field.yarea.resize((size_t)pad_x_4 * pad_y_5, 0.0);
    }
}