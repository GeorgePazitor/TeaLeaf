#include "tea.h"
#include "data.h"
#include "definitions.h"
#include "kernels/initialise_chunk_kernel.h" 
#include <omp.h>

void initialise_chunk() {
    using namespace TeaLeaf;

    double dx = (grid.xmax - grid.xmin) / (double)grid.x_cells;
    double dy = (grid.ymax - grid.ymin) / (double)grid.y_cells;

    #pragma omp parallel for
    for (int t = 0; t < tiles_per_task; ++t) {
        
        auto& tile = chunk.tiles[t];

       double tile_xmin = grid.xmin + dx * (double)(tile.left);
        double tile_ymin = grid.ymin + dy * (double)(tile.bottom);

        initialise_chunk_kernel(
            tile.field.x_min, 
            tile.field.x_max,
            tile.field.y_min, 
            tile.field.y_max,
            tile_xmin, 
            tile_ymin, 
            dx, 
            dy,
            tile.field.vertexx.data(),
            tile.field.vertexdx.data(),
            tile.field.vertexy.data(),
            tile.field.vertexdy.data(),
            tile.field.cellx.data(),
            tile.field.celldx.data(),
            tile.field.celly.data(),
            tile.field.celldy.data(),
            tile.field.volume.data(),
            tile.field.xarea.data(),
            tile.field.yarea.data()
        );
    }
}