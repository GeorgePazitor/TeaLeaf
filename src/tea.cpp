#include "include/tea.h"
#include "include/data.h"
#include "include/definitions.h"
#include "include/pack.h"
#include <algorithm>
#include <vector>

using namespace TeaLeaf;    

/**
 * Initializes the MPI environment and creates a 2D Cartesian grid 
 * for domain decomposition.
 */
void tea_init_comms() {
    int err, rank, size;
    int periodic[2] = {0, 0}; // Non-periodic boundaries

    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Automatically determines the best 2D processor grid (e.g., 4x4 for 16 ranks)
    MPI_Dims_create(size, 2, mpi_dims);
    
    // Create a communicator with Cartesian topology for easy neighbor shifting
    MPI_Cart_create(MPI_COMM_WORLD, 2, mpi_dims, periodic, 1, &mpi_cart_comm);

    MPI_Comm_rank(mpi_cart_comm, &rank);
    MPI_Comm_size(mpi_cart_comm, &size);
    MPI_Cart_coords(mpi_cart_comm, rank, 2, mpi_coords);

    parallel.task = rank;
    parallel.boss_task = 0;
    parallel.max_task = size;
    parallel.boss = (rank == 0);
}

/**
 * Decomposes the global problem grid into local chunks for each MPI rank.
 * Handles the mapping between Fortran-style 1-based logic and C++ 0-based indexing.
 */
void tea_decompose(int x_cells, int y_cells) {
    int err;
    
    // Finding neighbors using MPI_Cart_shift. 
    // Note: Direction 0/1 swap is often used to reconcile Fortran column-major 
    // with C++ row-major memory layouts.
    MPI_Cart_shift(mpi_cart_comm, 1, 1, 
                    &chunk.chunk_neighbours[CHUNK_BOTTOM], 
                    &chunk.chunk_neighbours[CHUNK_TOP]);
    MPI_Cart_shift(mpi_cart_comm, 0, 1, 
                    &chunk.chunk_neighbours[CHUNK_LEFT], 
                    &chunk.chunk_neighbours[CHUNK_RIGHT]);

    // Cleanup: Replace MPI_PROC_NULL with our internal boundary constant
    for(int i = 0; i < 4; ++i) {
        if(chunk.chunk_neighbours[i] == MPI_PROC_NULL) {
            chunk.chunk_neighbours[i] = EXTERNAL_FACE;
        }
    }

    int chunk_x = mpi_dims[0];
    int chunk_y = mpi_dims[1];

    int delta_x = x_cells / chunk_x;
    int delta_y = y_cells / chunk_y;
    int mod_x = x_cells % chunk_x;
    int mod_y = y_cells % chunk_y;

    // Distribute remaining cells (mod_x/y) across the first few ranks
    chunk.left = mpi_coords[0] * delta_x + std::min(mpi_coords[0], mod_x);
    chunk.right = chunk.left + delta_x - 1;
    if (mpi_coords[0] < mod_x) chunk.right++;

    chunk.bottom = mpi_coords[1] * delta_y + std::min(mpi_coords[1], mod_y);
    chunk.top = chunk.bottom + delta_y - 1;
    if (mpi_coords[1] < mod_y) chunk.top++;

    if (parallel.boss) {
        *g_out << "\n Mesh ratio of " << (double)x_cells/y_cells << "\n";
        *g_out << " Decomposing the mesh into " << chunk_x << " by " << chunk_y << " chunks\n";
    }
}

/**
 * Further divides the local MPI chunk into tiles for OpenMP multithreading.
 * Attempts to find a tiling geometry that minimizes the surface-to-volume ratio.
 */
void tea_decompose_tiles(int x_cells, int y_cells) {
    int tiles_x, tiles_y, mod_x, mod_y;
    int delta_x, delta_y;
    double best_fit_v = 0.0;
    int best_fit_i = 0;

    // Optimization loop to find tile dimensions closest to a square
    for (int i = 1; i <= tiles_per_task; ++i) {
        if (tiles_per_task % i != 0) continue;
        int j = tiles_per_task / i;
        double fit_v = (double)std::min(x_cells / i, y_cells / j) /
                       (double)std::max(x_cells / i, y_cells / j);
        if (fit_v > best_fit_v) {
            best_fit_v = fit_v;
            best_fit_i = i;
        }
    }

    if (best_fit_i == 0) { exit(1); }

    chunk.tile_dims[0] = best_fit_i;
    chunk.tile_dims[1] = tiles_per_task / best_fit_i;
    tiles_x = chunk.tile_dims[0];
    tiles_y = chunk.tile_dims[1];

    // Similar logic for sub-tiling (if applicable)
    best_fit_v = 0.0;
    best_fit_i = 0;
    double xs_cells = static_cast<double>(x_cells) / tiles_x;
    double ys_cells = static_cast<double>(y_cells) / tiles_y;

    for (int i = 1; i <= sub_tiles_per_tile; ++i) {
        if (sub_tiles_per_tile % i != 0) continue;
        int j = sub_tiles_per_tile / i;
        double fit_v = std::min(xs_cells / i, ys_cells / j) /
                       std::max(xs_cells / i, ys_cells / j);
        if (fit_v > best_fit_v) {
            best_fit_v = fit_v;
            best_fit_i = i;
        }
    }

    if (best_fit_i == 0) { exit(1); }

    chunk.sub_tile_dims[0] = best_fit_i;
    chunk.sub_tile_dims[1] = sub_tiles_per_tile / best_fit_i;

    delta_x = x_cells / tiles_x;
    delta_y = y_cells / tiles_y;
    mod_x = x_cells % tiles_x;
    mod_y = y_cells % tiles_y;

    chunk.tiles.resize(tiles_per_task);

    // Populate tile structures with local coordinate offsets
    for (int j = 0; j < tiles_x; ++j) {
        for (int k = 0; k < tiles_y; ++k) {
            int t = j * tiles_y + k;
            chunk.tiles[t].tile_coords[0] = j;
            chunk.tiles[t].tile_coords[1] = k;
            chunk.tiles[t].left   = chunk.left + j * delta_x + std::min(j, mod_x);
            chunk.tiles[t].right  = chunk.tiles[t].left + delta_x - 1;
            if (j < mod_x) chunk.tiles[t].right++;
            chunk.tiles[t].bottom = chunk.bottom + k * delta_y + std::min(k, mod_y);
            chunk.tiles[t].top    = chunk.tiles[t].bottom + delta_y - 1;
            if (k < mod_y) chunk.tiles[t].top++;

            // Internal tile adjacency within the MPI rank
            std::fill(chunk.tiles[t].tile_neighbours.begin(), chunk.tiles[t].tile_neighbours.end(), EXTERNAL_FACE);
            if (j > 0)           chunk.tiles[t].tile_neighbours[CHUNK_LEFT]   = (j - 1) * tiles_y + k;
            if (j < tiles_x - 1) chunk.tiles[t].tile_neighbours[CHUNK_RIGHT]  = (j + 1) * tiles_y + k;
            if (k > 0)           chunk.tiles[t].tile_neighbours[CHUNK_BOTTOM] = j * tiles_y + (k - 1);
            if (k < tiles_y - 1) chunk.tiles[t].tile_neighbours[CHUNK_TOP]    = j * tiles_y + (k + 1);
        }
    }
}

/**
 * Resizes MPI buffers based on the halo depth and number of fields to be exchanged.
 */
void tea_allocate_buffers() {
    int allocate_extra_size = std::max(2, chunk.halo_exchange_depth);
    int lr_size = NUM_FIELDS * (chunk.y_cells + 2 * allocate_extra_size) * chunk.halo_exchange_depth;
    int bt_size = NUM_FIELDS * (chunk.x_cells + 2 * allocate_extra_size) * chunk.halo_exchange_depth;

    chunk.left_snd_buffer.resize(lr_size, 0.0);
    chunk.left_rcv_buffer.resize(lr_size, 0.0);
    chunk.right_snd_buffer.resize(lr_size, 0.0);
    chunk.right_rcv_buffer.resize(lr_size, 0.0);
    chunk.bottom_snd_buffer.resize(bt_size, 0.0);
    chunk.bottom_rcv_buffer.resize(bt_size, 0.0);
    chunk.top_snd_buffer.resize(bt_size, 0.0);
    chunk.top_rcv_buffer.resize(bt_size, 0.0);
}

/**
 * Main halo exchange routine. Performs a 2-step (Horizontal then Vertical) 
 * non-blocking exchange to synchronize ghost cells between MPI ranks.
 */
void tea_exchange(const int* fields, int depth) {
    bool no_neighbours = true;
    for(int i=0; i<4; ++i) if(chunk.chunk_neighbours[i] != EXTERNAL_FACE) no_neighbours = false;
    if (no_neighbours) return;

    int exchange_size_lr = depth * (chunk.y_cells + 2 * depth);
    int exchange_size_ud = depth * (chunk.x_cells + 2 * depth);

    std::vector<int> lr_offset(NUM_FIELDS, 0);
    std::vector<int> bt_offset(NUM_FIELDS, 0);
    int end_pack_lr = 0, end_pack_bt = 0;

    // Calculate offsets for packing multiple fields into a single message
    for (int f = 0; f < NUM_FIELDS; ++f) {
        if (fields[f] == 1) {
            lr_offset[f] = end_pack_lr;
            bt_offset[f] = end_pack_bt;
            end_pack_lr += exchange_size_lr;
            end_pack_bt += exchange_size_ud;
        }
    }

    MPI_Request requests[4];
    int msg_count = 0;

    // --- PHASE 1 : HORIZONTAL EXCHANGE (LEFT / RIGHT) ---
    if (chunk.chunk_neighbours[CHUNK_LEFT] != EXTERNAL_FACE) {
        tea_pack_buffers(fields, depth, CHUNK_LEFT, chunk.left_snd_buffer.data(), lr_offset.data());
        tea_send_recv_message_left(chunk.left_snd_buffer.data(), chunk.left_rcv_buffer.data(), 
                                    end_pack_lr, 1, 2, &requests[msg_count], &requests[msg_count+1]);
        msg_count += 2;
    }
    if (chunk.chunk_neighbours[CHUNK_RIGHT] != EXTERNAL_FACE) {
        tea_pack_buffers(fields, depth, CHUNK_RIGHT, chunk.right_snd_buffer.data(), lr_offset.data());
        tea_send_recv_message_right(chunk.right_snd_buffer.data(), chunk.right_rcv_buffer.data(), 
                                     end_pack_lr, 2, 1, &requests[msg_count], &requests[msg_count+1]);
        msg_count += 2;
    }

    if (msg_count > 0) {
        MPI_Waitall(msg_count, requests, MPI_STATUSES_IGNORE);
        if (chunk.chunk_neighbours[CHUNK_LEFT] != EXTERNAL_FACE)
            tea_unpack_buffers(fields, depth, CHUNK_LEFT, chunk.left_rcv_buffer.data(), lr_offset.data());
        if (chunk.chunk_neighbours[CHUNK_RIGHT] != EXTERNAL_FACE)
            tea_unpack_buffers(fields, depth, CHUNK_RIGHT, chunk.right_rcv_buffer.data(), lr_offset.data());
    }

    // --- PHASE 2 : VERTICAL EXCHANGE (BOTTOM / TOP) ---
    msg_count = 0; 
    if (chunk.chunk_neighbours[CHUNK_BOTTOM] != EXTERNAL_FACE) {
        tea_pack_buffers(fields, depth, CHUNK_BOTTOM, chunk.bottom_snd_buffer.data(), bt_offset.data());
        tea_send_recv_message_bottom(chunk.bottom_snd_buffer.data(), chunk.bottom_rcv_buffer.data(), 
                                      end_pack_bt, 3, 4, &requests[msg_count], &requests[msg_count+1]);
        msg_count += 2;
    }
    if (chunk.chunk_neighbours[CHUNK_TOP] != EXTERNAL_FACE) {
        tea_pack_buffers(fields, depth, CHUNK_TOP, chunk.top_snd_buffer.data(), bt_offset.data());
        tea_send_recv_message_top(chunk.top_snd_buffer.data(), chunk.top_rcv_buffer.data(), 
                                  end_pack_bt, 4, 3, &requests[msg_count], &requests[msg_count+1]);
        msg_count += 2;
    }

    if (msg_count > 0) {
        MPI_Waitall(msg_count, requests, MPI_STATUSES_IGNORE);
        if (chunk.chunk_neighbours[CHUNK_BOTTOM] != EXTERNAL_FACE)
            tea_unpack_buffers(fields, depth, CHUNK_BOTTOM, chunk.bottom_rcv_buffer.data(), bt_offset.data());
        if (chunk.chunk_neighbours[CHUNK_TOP] != EXTERNAL_FACE)
            tea_unpack_buffers(fields, depth, CHUNK_TOP, chunk.top_rcv_buffer.data(), bt_offset.data());
    }
}

// Low-level MPI Isend/Irecv wrappers for each cardinal direction
void tea_send_recv_message_left(double* snd, double* rcv, int size, int tag_s, int tag_r, MPI_Request* req_s, MPI_Request* req_r) {
    int dest = chunk.chunk_neighbours[CHUNK_LEFT];
    MPI_Isend(snd, size, MPI_DOUBLE, dest, tag_s, mpi_cart_comm, req_s);
    MPI_Irecv(rcv, size, MPI_DOUBLE, dest, tag_r, mpi_cart_comm, req_r);
}

void tea_send_recv_message_right(double* snd_buf, double* rcv_buf, int size, int tag_send, int tag_recv, MPI_Request* req_send, MPI_Request* req_recv) {
    int right_task = chunk.chunk_neighbours[CHUNK_RIGHT];
    MPI_Isend(snd_buf, size, MPI_DOUBLE, right_task, tag_send, mpi_cart_comm, req_send);
    MPI_Irecv(rcv_buf, size, MPI_DOUBLE, right_task, tag_recv, mpi_cart_comm, req_recv);
}

void tea_send_recv_message_top(double* snd_buf, double* rcv_buf, int size, int tag_send, int tag_recv, MPI_Request* req_send, MPI_Request* req_recv) {
    int top_task = chunk.chunk_neighbours[CHUNK_TOP];
    MPI_Isend(snd_buf, size, MPI_DOUBLE, top_task, tag_send, mpi_cart_comm, req_send);
    MPI_Irecv(rcv_buf, size, MPI_DOUBLE, top_task, tag_recv, mpi_cart_comm, req_recv);
}

void tea_send_recv_message_bottom(double* snd_buf, double* rcv_buf, int size, int tag_send, int tag_recv, MPI_Request* req_send, MPI_Request* req_recv) {
    int bottom_task = chunk.chunk_neighbours[CHUNK_BOTTOM];
    MPI_Isend(snd_buf, size, MPI_DOUBLE, bottom_task, tag_send, mpi_cart_comm, req_send);
    MPI_Irecv(rcv_buf, size, MPI_DOUBLE, bottom_task, tag_recv, mpi_cart_comm, req_recv);
}

void tea_finalize() {
    std::cout.flush();
    std::cerr.flush();
    MPI_Finalize();
}