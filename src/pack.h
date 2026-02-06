#ifndef PACKING_H
#define PACKING_H

#include <definitions.h>
#include "data.h"
#include <mpi.h>

#include "kernels/pack_kernel.h"


void call_packing_functions(int* fields, int depth, int face, bool packing, std::vector<double>& mpi_buffer, int* offsets);

void tea_pack_buffers(int* fields, int depth, int face, std::vector<double>& mpi_buffer, int* offsets);

void tea_unpack_buffers(int* fields, int depth, int face, std::vector<double>& mpi_buffer, int* offsets);

void call_packing_functions(int* fields, int depth, int face, bool packing, std::vector<double>& mpi_buffer, int* offsets); 

#endif