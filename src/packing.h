#ifndef PACKING_H
#define PACKING_H

#include "kernels/pack_kernel.h"
#include "data.h"
#include "definitions.h"
#include <mpi.h>

void call_packing_functions(int* fields, int depth, int face, bool packing, double* mpi_buffer, int* offsets);

void tea_pack_buffers(int* fields, int depth, int face, double* mpi_buffer, int* offsets);

void tea_unpack_buffers(int* fields, int depth, int face, double* mpi_buffer, int* offsets);

void call_packing_functions(int* fields, int depth, int face, bool packing, double* mpi_buffer, int* offsets); 

#endif