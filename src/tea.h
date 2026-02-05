#ifndef TEA_MODULE_HPP
#define TEA_MODULE_HPP

#include <vector>
#include <mpi.h>

#include "definitions.h"
#include "data.h"
#include "packing.h"


void tea_init_comms();
void tea_finalize();
void tea_decompose(int x_cells, int y_cells);
void tea_decompose_tiles(int x_cells, int y_cells);
void tea_allocate_buffers();
void tea_exchange(const std::vector<int>& fields, int depth);

void tea_send_recv_message_left(double* snd_buf, double* rcv_buf, int size, int tag_send, int tag_recv, MPI_Request* req_send, MPI_Request* req_recv);
void tea_send_recv_message_right(double* snd_buf, double* rcv_buf, int size, int tag_send, int tag_recv, MPI_Request* req_send, MPI_Request* req_recv);
void tea_send_recv_message_top(double* snd_buf, double* rcv_buf, int size, int tag_send, int tag_recv, MPI_Request* req_send, MPI_Request* req_recv);
void tea_send_recv_message_bottom(double* snd_buf, double* rcv_buf, int size, int tag_send, int tag_recv, MPI_Request* req_send, MPI_Request* req_recv);

#endif