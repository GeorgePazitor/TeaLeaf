#pragma once

void call_packing_functions(const int* fields, int depth, int face, bool packing, double* mpi_buffer, int* offsets);

void tea_pack_buffers(const int* fields, int depth, int face, double* mpi_buffer, int* offsets);

void tea_unpack_buffers(const int* fields, int depth, int face, double* mpi_buffer, int* offsets);

