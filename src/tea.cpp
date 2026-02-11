#include "tea.h"
#include "data.h"
#include "definitions.h"
#include "pack.h"

#include <vector>


using namespace TeaLeaf;
// using namespace  global_mpi_module

void tea_init_comms(){
  int err, rank, size;
  int periodic[2] = {0,0};
    
  mpi_dims[0]=0;
  mpi_dims[1]=0;

  rank=0;
  size=1;

  MPI_Init(NULL, NULL);

  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Dims_create(size, 2, mpi_dims);
  MPI_Cart_create(MPI_COMM_WORLD, 2, mpi_dims, periodic, 1, &mpi_cart_comm);

  MPI_Comm_rank(mpi_cart_comm, &rank);
  MPI_Comm_size(mpi_cart_comm, &size);
  MPI_Cart_coords(mpi_cart_comm, rank, 2, mpi_coords);

  if (rank == 0) {
      parallel.boss = true;
  }

  parallel.task = rank;
  parallel.boss_task = 0;
  parallel.max_task = size;
}


void tea_finalize() {
    std::cout.flush();
    std::cerr.flush();    
    MPI_Finalize();
}


void tea_decompose(int x_cells, int y_cells) {
    int delta_x, delta_y;
    int chunk_x, chunk_y, mod_x, mod_y;
    int err;

    //This decomposes the mesh into a number of chunks.

    // Get neighbors (Shift along X and Y axes)
    // 0 = Y direction (Rows), 1 = X direction (Cols) in C Row-Major? 
    // MPI_Cart_shift directions depend on how dims were mapped.
    // Assuming dim 0 = Y, dim 1 = X for standard 2D C arrays.
    
    // Fortran usually maps (1) -> X, (2) -> Y.
    // C++ usually maps [0] -> Y (rows), [1] -> X (cols).
    
    // Top/Bottom (Y-axis, dim 0)
    MPI_Cart_shift(mpi_cart_comm, 0, 1, 
                   &chunk.chunk_neighbours[CHUNK_TOP],    // source 
                   &chunk.chunk_neighbours[CHUNK_BOTTOM]); // dest 

    // Left/Right (X-axis, dim 1)
    MPI_Cart_shift(mpi_cart_comm, 1, 1, 
                   &chunk.chunk_neighbours[CHUNK_LEFT], 
                   &chunk.chunk_neighbours[CHUNK_RIGHT]);

    // Handle Boundary Conditions (MPI_PROC_NULL -> EXTERNAL_FACE)
    for(int i=0; i<4; ++i) {
        if(chunk.chunk_neighbours[i] == MPI_PROC_NULL) {
            chunk.chunk_neighbours[i] = EXTERNAL_FACE;
        }
    }

    chunk_x = mpi_dims[1]; // Columns
    chunk_y = mpi_dims[0]; // Rows

    delta_x = x_cells / chunk_x;
    delta_y = y_cells / chunk_y;
    mod_x = x_cells % chunk_x;
    mod_y = y_cells % chunk_y;

    // X-Bounds
    chunk.left = mpi_coords[1] * delta_x; 
    if (mpi_coords[1] < mod_x) {
        chunk.left += mpi_coords[1];
    } else {
        chunk.left += mod_x;
    }
    
    chunk.right = chunk.left + delta_x - 1;
    if (mpi_coords[1] < mod_x) {
        chunk.right += 1;
    }

    // Y-Bounds
    chunk.bottom = mpi_coords[0] * delta_y; 
    if (mpi_coords[0] < mod_y) {
        chunk.bottom += mpi_coords[0];
    } else {
        chunk.bottom += mod_y;
    }
    
    chunk.top = chunk.bottom + delta_y - 1;
    if (mpi_coords[0] < mod_y) {
        chunk.top += 1;
    }

    if (parallel.boss) {
        std::cout << "\nMesh ratio of " << (double)x_cells / (double)y_cells << std::endl;
        std::cout << "Decomposing the mesh into " << chunk_x << " by " << chunk_y << " chunks" << std::endl;
    }
}

void tea_decompose_tiles(int x_cells, int y_cells) {

    double best_fit_v = 0.0;
    int best_fit_i = 0;
    
    //lookinf for the first divisor of tiles_per_task that gives the best fit for the tile dimensions  
    for (int i = 1; i <= tiles_per_task; ++i) {
        if (tiles_per_task % i != 0) continue;
        int j = tiles_per_task / i; // i * j = total tiles

        double dim_x = (double)x_cells / i;
        double dim_y = (double)y_cells / j;
        double fit_v = std::min(dim_x, dim_y) / std::max(dim_x, dim_y);

        if (fit_v > best_fit_v) {
            best_fit_v = fit_v;
            best_fit_i = i;
        }
    }

    if (best_fit_i == 0) {
        std::cerr << "No fit found - tiles_per_task=" << tiles_per_task << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    chunk.tile_dims[0] = best_fit_i; // tiles_x
    chunk.tile_dims[1] = tiles_per_task / best_fit_i; // tiles_y

    //Calculate Sub-Tile Dimensions 
    // ----------------------------------------------------
    best_fit_v = 0.0;
    best_fit_i = 0;
    
    //Assuming simple assignment for now:
    chunk.sub_tile_dims[0] = 1; 
    chunk.sub_tile_dims[1] = 1;

    int tiles_x = chunk.tile_dims[0];
    int tiles_y = chunk.tile_dims[1];

    int delta_x = x_cells / tiles_x;
    int delta_y = y_cells / tiles_y;
    int mod_x = x_cells % tiles_x;
    int mod_y = y_cells % tiles_y;

    // Resize the flat vector of tiles
    chunk.tiles.resize(tiles_per_task);

    for (int j = 0; j < tiles_x; ++j) { 
        for (int k = 0; k < tiles_y; ++k) {             
           
            int t = j * tiles_y + k; 

            // Store Coordinates
            chunk.tiles[t].tile_coords[0] = j;
            chunk.tiles[t].tile_coords[1] = k;

            // --- X Calculation ---
            chunk.tiles[t].left = chunk.left + j * delta_x;
            if (j <= mod_x) {
                // If we are past the 'mod' boundary, add full mod_x, else add current index
                chunk.tiles[t].left += (j < mod_x) ? j : mod_x; // Logic adjusted for 0-base
            } else {
                 chunk.tiles[t].left += mod_x;
            }
            // (Re-verifying exact Fortran logic translation requires care with <= vs <)
            // Simplified logic: Distribute remainder 1 pixel per tile for first 'mod_x' tiles.
            int offset_x = (j < mod_x) ? j : mod_x;
            chunk.tiles[t].left = chunk.left + j * delta_x + offset_x;
            
            int my_width = delta_x + (j < mod_x ? 1 : 0);
            chunk.tiles[t].right = chunk.tiles[t].left + my_width - 1;

            // --- Y Calculation ---
            int offset_y = (k < mod_y) ? k : mod_y;
            chunk.tiles[t].bottom = chunk.bottom + k * delta_y + offset_y;
            
            int my_height = delta_y + (k < mod_y ? 1 : 0);
            chunk.tiles[t].top = chunk.tiles[t].bottom + my_height - 1;

            // --- Neighbors (Internal Tiling) ---
            // Initialize to External
            for(int n=0; n<4; ++n) chunk.tiles[t].tile_neighbours[n] = EXTERNAL_FACE;

            // Left
            if (j > 0) 
                chunk.tiles[t].tile_neighbours[CHUNK_LEFT] = (j - 1) * tiles_y + k;
            // Right
            if (j < tiles_x - 1) 
                chunk.tiles[t].tile_neighbours[CHUNK_RIGHT] = (j + 1) * tiles_y + k;
            // Bottom
            if (k > 0) 
                chunk.tiles[t].tile_neighbours[CHUNK_BOTTOM] = j * tiles_y + (k - 1);
            // Top
            if (k < tiles_y - 1) 
                chunk.tiles[t].tile_neighbours[CHUNK_TOP] = j * tiles_y + (k + 1);
        }
    }

    if (parallel.boss) {
        std::cout << "Decomposing each chunk into " << tiles_x << " by " << tiles_y << " tiles\n";
    }
}


void tea_allocate_buffers() {
    int allocate_extra_size = std::max(2, chunk.halo_exchange_depth);

    // X-direction buffers (Left/Right) need Y-dimension size
    int lr_size = NUM_FIELDS * (chunk.y_cells + 2 * allocate_extra_size) * chunk.halo_exchange_depth;
    
    // Y-direction buffers (Top/Bottom) need X-dimension size
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



void tea_exchange(const int* fields, int depth) {
    // halo exchange driver

    // Check if fully isolated (no neighbors)
    bool isolated = true;
    for(int i=0; i<4; ++i) {
        if(chunk.chunk_neighbours[i] != EXTERNAL_FACE) isolated = false;
    }
    if (isolated) return;

    int exchange_size_lr = depth * (chunk.y_cells + 2 * depth);
    int exchange_size_ud = depth * (chunk.x_cells + 2 * depth);

    MPI_Request request_lr[4];
    MPI_Request request_ud[4];
    MPI_Status status_array[4];
    
    int msg_count_lr = 0;
    int msg_count_ud = 0;

    // Offsets for packing multiple fields into one buffer
    std::vector<int> offsets_lr(NUM_FIELDS, 0);
    std::vector<int> offsets_ud(NUM_FIELDS, 0);
    
    int end_pack_idx_lr = 0;
    int end_pack_idx_ud = 0;

    // Calculate packing offsets
    for (int f = 0; f < NUM_FIELDS; ++f) {
        if (fields[f] == 1) { // Active field
            offsets_lr[f] = end_pack_idx_lr;
            offsets_ud[f] = end_pack_idx_ud;
            end_pack_idx_lr += exchange_size_lr;
            end_pack_idx_ud += exchange_size_ud;
        }
    }

    // --- Left/Right Phase ---

    if (chunk.chunk_neighbours[CHUNK_LEFT] != EXTERNAL_FACE) {
        // Pack
        tea_pack_buffers(fields, depth, CHUNK_LEFT, chunk.left_snd_buffer.data(), offsets_lr.data());
        
        // Send/Recv
        tea_send_recv_message_left(chunk.left_snd_buffer.data(), chunk.left_rcv_buffer.data(),
                                   end_pack_idx_lr, 1, 2, 
                                   &request_lr[msg_count_lr], &request_lr[msg_count_lr+1]);
        msg_count_lr += 2;
    }

    if (chunk.chunk_neighbours[CHUNK_RIGHT] != EXTERNAL_FACE) {
        // Pack
        tea_pack_buffers(fields, depth, CHUNK_RIGHT, chunk.right_snd_buffer.data(), offsets_lr.data());
        
        // Send/Recv
        tea_send_recv_message_right(chunk.right_snd_buffer.data(), chunk.right_rcv_buffer.data(),
                                    end_pack_idx_lr, 2, 1, 
                                    &request_lr[msg_count_lr], &request_lr[msg_count_lr+1]);
        msg_count_lr += 2;
    }

    // Wait for Left/Right
    if (depth == 1) {
        // Optimization for depth 1 (Testing?) - Kept logic from Fortran
        int flag = 0;
        MPI_Testall(msg_count_lr, request_lr, &flag, status_array);
        if (!flag) {
            MPI_Waitall(msg_count_lr, request_lr, status_array);
        }
    } else {
        MPI_Waitall(msg_count_lr, request_lr, status_array);
    }

    // Unpack Left/Right
    if (chunk.chunk_neighbours[CHUNK_LEFT] != EXTERNAL_FACE) {
        tea_unpack_buffers(fields, depth, CHUNK_LEFT, chunk.left_rcv_buffer.data(), offsets_lr.data());
    }
    if (chunk.chunk_neighbours[CHUNK_RIGHT] != EXTERNAL_FACE) {
        tea_unpack_buffers(fields, depth, CHUNK_RIGHT, chunk.right_rcv_buffer.data(), offsets_lr.data());
    }

    // --- Top/Bottom Phase ---

    if (chunk.chunk_neighbours[CHUNK_BOTTOM] != EXTERNAL_FACE) {
        tea_pack_buffers(fields, depth, CHUNK_BOTTOM, chunk.bottom_snd_buffer.data(), offsets_ud.data());
        
        tea_send_recv_message_bottom(chunk.bottom_snd_buffer.data(), chunk.bottom_rcv_buffer.data(),
                                     end_pack_idx_ud, 3, 4,
                                     &request_ud[msg_count_ud], &request_ud[msg_count_ud+1]);
        msg_count_ud += 2;
    }

    if (chunk.chunk_neighbours[CHUNK_TOP] != EXTERNAL_FACE) {
        tea_pack_buffers(fields, depth, CHUNK_TOP, chunk.top_snd_buffer.data(), offsets_ud.data());

        tea_send_recv_message_top(chunk.top_snd_buffer.data(), chunk.top_rcv_buffer.data(),
                                  end_pack_idx_ud, 4, 3,
                                  &request_ud[msg_count_ud], &request_ud[msg_count_ud+1]);
        msg_count_ud += 2;
    }

    // Wait for Top/Bottom
    MPI_Waitall(msg_count_ud, request_ud, status_array);

    // Unpack Top/Bottom
    if (chunk.chunk_neighbours[CHUNK_TOP] != EXTERNAL_FACE) {
        tea_unpack_buffers(fields, depth, CHUNK_TOP, chunk.top_rcv_buffer.data(), offsets_ud.data());
    }
    if (chunk.chunk_neighbours[CHUNK_BOTTOM] != EXTERNAL_FACE) {
        tea_unpack_buffers(fields, depth, CHUNK_BOTTOM, chunk.bottom_rcv_buffer.data(), offsets_ud.data());
    }
}

// MPI Low-Level Wrappers

void tea_send_recv_message_left(double* snd, double* rcv, int size, int tag_s, int tag_r, MPI_Request* req_s, MPI_Request* req_r) {
    int dest = chunk.chunk_neighbours[CHUNK_LEFT];
    MPI_Isend(snd, size, MPI_DOUBLE, dest, tag_s, mpi_cart_comm, req_s);
    MPI_Irecv(rcv, size, MPI_DOUBLE, dest, tag_r, mpi_cart_comm, req_r);
}

void tea_send_recv_message_right(double* snd, double* rcv, int size, int tag_s, int tag_r, MPI_Request* req_s, MPI_Request* req_r) {
    int dest = chunk.chunk_neighbours[CHUNK_RIGHT];
    MPI_Isend(snd, size, MPI_DOUBLE, dest, tag_s, mpi_cart_comm, req_s);
    MPI_Irecv(rcv, size, MPI_DOUBLE, dest, tag_r, mpi_cart_comm, req_r);
}

void tea_send_recv_message_top(double* snd, double* rcv, int size, int tag_s, int tag_r, MPI_Request* req_s, MPI_Request* req_r) {
    int dest = chunk.chunk_neighbours[CHUNK_TOP];
    MPI_Isend(snd, size, MPI_DOUBLE, dest, tag_s, mpi_cart_comm, req_s);
    MPI_Irecv(rcv, size, MPI_DOUBLE, dest, tag_r, mpi_cart_comm, req_r);
}

void tea_send_recv_message_bottom(double* snd, double* rcv, int size, int tag_s, int tag_r, MPI_Request* req_s, MPI_Request* req_r) {
    int dest = chunk.chunk_neighbours[CHUNK_BOTTOM];
    MPI_Isend(snd, size, MPI_DOUBLE, dest, tag_s, mpi_cart_comm, req_s);
    MPI_Irecv(rcv, size, MPI_DOUBLE, dest, tag_r, mpi_cart_comm, req_r);
}
