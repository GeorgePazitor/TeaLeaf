#include "set_field.h"
#include "tea.h"
#include "data.h"
#include "definitions.h"
#include "set_field_kernel.h"
#include <mpi.h>
#include <omp.h>

void set_field() {
    using namespace TeaLeaf;

    double start_time = 0.0;
    if (profiler_on) start_time = MPI_Wtime();

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int t = 0; t < tiles_per_task; ++t) {
            
            auto& tile = chunk.tiles[t];

            set_field_kernel(
                tile.field.x_min,
                tile.field.x_max,
                tile.field.y_min,
                tile.field.y_max,
                chunk.halo_exchange_depth,
                tile.field.energy0.data(),
                tile.field.energy1.data()
            );
        }
    }

    if (profiler_on) {
        profiler.set_field += (MPI_Wtime() - start_time);
    }
}