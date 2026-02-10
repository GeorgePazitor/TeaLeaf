#include "pack.h"
#include <iostream>
#include <cstdio>

namespace TeaLeaf {

void tea_pack_buffers(int* fields, int depth, int face, double* mpi_buffer, int* offsets) {
    call_packing_functions(fields, depth, face, true, mpi_buffer, offsets);
}

void tea_unpack_buffers(int* fields, int depth, int face, double* mpi_buffer, int* offsets) {
    call_packing_functions(fields, depth, face, false, mpi_buffer, offsets);
}

void call_packing_functions(int* fields, int depth, int face, bool packing, double* mpi_buffer, int* offsets) {
    
    // tile_offset doit être privé à chaque thread OpenMP
    #pragma omp parallel
    {
        int tile_offset = 0;

        #pragma omp for nowait
        for (int t = 0; t < tiles_per_task; ++t) {
            
            // 1. Calcul des Offsets
            switch (face) {
                case CHUNK_LEFT:
                case CHUNK_RIGHT:
                    tile_offset = (chunk.tiles[t].bottom - chunk.bottom) * depth;
                    break;
                
                case CHUNK_BOTTOM:
                case CHUNK_TOP:
                    tile_offset = (chunk.tiles[t].left - chunk.left) * depth;
                    if (tile_offset != 0) {
                        tile_offset += (depth * depth);
                    }
                    break;
                
                default:
                    #pragma omp critical
                    {
                        fprintf(stderr, "In pack_module.cpp : Invalid face passed to buffer packing\n");
                        std::exit(1);
                    }
            }

            // 2. IMPORTANT : Vérification du voisin (Logique Fortran)
            // On ne pack/unpack QUE si la tuile est sur une frontière MPI
            if (chunk.tiles[t].tile_neighbours[face] != EXTERNAL_FACE) {
                continue; 
            }

            // 3. Appel du Kernel avec .data()
            pack_all(
                chunk.tiles[t].field.x_min,
                chunk.tiles[t].field.x_max,
                chunk.tiles[t].field.y_min,
                chunk.tiles[t].field.y_max,
                chunk.halo_exchange_depth,
                chunk.tiles[t].tile_neighbours, 
                chunk.tiles[t].field.density.data(),
                chunk.tiles[t].field.energy0.data(),
                chunk.tiles[t].field.energy1.data(),
                chunk.tiles[t].field.u.data(),
                chunk.tiles[t].field.vector_p.data(),
                chunk.tiles[t].field.vector_sd.data(),
                chunk.tiles[t].field.vector_rtemp.data(),
                chunk.tiles[t].field.vector_z.data(),
                chunk.tiles[t].field.vector_Kx.data(),
                chunk.tiles[t].field.vector_Ky.data(),
                chunk.tiles[t].field.vector_Di.data(),
                fields,
                depth,
                face,
                packing,
                mpi_buffer,
                offsets,
                tile_offset
            );
        }
    }
}

} // namespace TeaLeaf