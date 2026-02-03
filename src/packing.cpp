#include "tealeaf.hpp"
void call_packing_functions(int* fields, int depth, int face, bool packing, double* mpi_buffer, int* offsets);

void tea_pack_buffers(int* fields, int depth, int face, double* mpi_buffer, int* offsets) {
    call_packing_functions(fields, depth, face, true, mpi_buffer, offsets);
}

void tea_unpack_buffers(int* fields, int depth, int face, double* mpi_buffer, int* offsets) {
    call_packing_functions(fields, depth, face, false, mpi_buffer, offsets);
}

void call_packing_functions(int* fields, int depth, int face, bool packing, double* mpi_buffer, int* offsets) {
    
    int tile_offset;

    #ifdef OMP
    #pragma omp parallel private(tile_offset)
    {
    
        #pragma omp for nowait
    #endif
        for (int t = 0; t < chunk.tiles_per_task; ++t) {
            
            // 1. Calculate Offsets based on Face direction
            switch (face) {
                case CHUNK_LEFT:
                case CHUNK_RIGHT:
                    tile_offset = (chunk.tiles[t].bottom - chunk.bottom) * depth;
                    break;
                
                case CHUNK_BOTTOM:
                case CHUNK_TOP:
                    tile_offset = (chunk.tiles[t].left - chunk.left) * depth;
                    
                    // Specific adjustment from the Fortran logic
                    if (tile_offset != 0) {
                        tile_offset = tile_offset + (depth * depth);
                    }
                    break;
                
                default:
                    // In C++, we typically throw exceptions or call a dedicated error handler
                    fprintf(stderr, "In pack_module.cpp : Invalid face passed to buffer packing");
                    std::exit(1);
            }

            // 2. Neighbor Check (Skip if no neighbor exists on this face)
            // Note: Fortran arrays are 1-based, C++ usually 0-based. 
            // Ensure 'face' index matches your definition in definitions.h
            if (chunk.tiles[t].tile_neighbours[face] == EXTERNAL_FACE) {
                continue; 
            }

            // 3. Call the Kernel
            // Passing pointers to the raw data vectors inside the tile structure
            pack_all(
                chunk.tiles[t].field.x_min,
                chunk.tiles[t].field.x_max,
                chunk.tiles[t].field.y_min,
                chunk.tiles[t].field.y_max,
                chunk.halo_exchange_depth,
                chunk.tiles[t].tile_neighbours, // Pass the array/pointer
                chunk.tiles[t].field.density,
                chunk.tiles[t].field.energy0,
                chunk.tiles[t].field.energy1,
                chunk.tiles[t].field.u,
                chunk.tiles[t].field.vector_p,
                chunk.tiles[t].field.vector_sd,
                chunk.tiles[t].field.vector_rtemp,
                chunk.tiles[t].field.vector_z,
                chunk.tiles[t].field.vector_kx,
                chunk.tiles[t].field.vector_ky,
                chunk.tiles[t].field.vector_di,
                fields,
                depth,
                face,
                packing,
                mpi_buffer,
                offsets,
                tile_offset
            );
        }
    #ifdef OMP
    }
    #endif
}