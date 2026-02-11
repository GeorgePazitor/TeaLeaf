#include "build_field.h"

#include "tea.h"
#include "data.h"
#include "definitions.h"
#include <vector>
#include <algorithm>
#include <omp.h>
using namespace TeaLeaf;
void build_field() {
    //using namespace TeaLeaf;

    #pragma omp parallel for
    for (int t = 0; t < tiles_per_task; ++t) {
        
        auto& tile = chunk.tiles[t];
        auto& field = tile.field;
        int depth = chunk.halo_exchange_depth;

        field.x_min = 0;
        field.y_min = 0;
        // inclusive bounds
        field.x_max = tile.x_cells - 1;
        field.y_max = tile.y_cells - 1;

        // 
        // allocate main fields (with jalo)
        
        // fortran: x_min-depth : x_max+depth
        // C++ size: x_cells + 2*depth
        
        int main_width  = tile.x_cells + 2 * depth;
        int main_height = tile.y_cells + 2 * depth;
        size_t main_size = (size_t)main_width * main_height;

        field.density.resize(main_size, 0.0);
        field.energy0.resize(main_size, 0.0);
        field.energy1.resize(main_size, 0.0);
        field.u.resize(main_size, 0.0);
        field.u0.resize(main_size, 0.0);
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

        // allocate triangle fields (no halo)
        // fortran: x_min : x_max
        size_t tri_size = (size_t)tile.x_cells * tile.y_cells;
        
        field.tri_cp.resize(tri_size, 0.0);
        field.tri_bfp.resize(tri_size, 0.0);

        // allocate geometric fields (extra padding)
        // fortran bounds: -2 to +2 (implies size + 4)
        // fortran bounds: -2 to +3 (implies size + 5)

        int pad_x_4 = tile.x_cells + 4; // -2 to +2
        int pad_y_4 = tile.y_cells + 4; // -2 to +2
        
        int pad_x_5 = tile.x_cells + 5; // -2 to +3
        int pad_y_5 = tile.y_cells + 5; // -2 to +3

        // 1D vectors
        field.cellx.resize(pad_x_4, 0.0);
        field.celldx.resize(pad_x_4, 0.0);
        field.celly.resize(pad_y_4, 0.0);
        field.celldy.resize(pad_y_4, 0.0);

        field.vertexx.resize(pad_x_5, 0.0);
        field.vertexdx.resize(pad_x_5, 0.0);
        field.vertexy.resize(pad_y_5, 0.0);
        field.vertexdy.resize(pad_y_5, 0.0);

        // 2D vectors
        // volume: -2..+2 (both dims)
        field.volume.resize((size_t)pad_x_4 * pad_y_4, 0.0);

        // xarea: X (-2..+3), Y (-2..+2)
        field.xarea.resize((size_t)pad_x_5 * pad_y_4, 0.0);

        // yarea: X (-2..+2), Y (-2..+3)
        field.yarea.resize((size_t)pad_x_4 * pad_y_5, 0.0);
    }
}