#include "tea.h"
#include "data.h"
#include "definitions.h"
#include "pack.h"

#include <vector>

namespace TeaLeaf {
    
void tea_init_comms() {
    int err, rank, size;
    int periodic[2] = {0, 0};

    // Initialisation MPI (si pas déjà faite dans le main)
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Création de la grille cartésienne 2D
    MPI_Dims_create(size, 2, mpi_dims);
    // 1 signifie que MPI peut réorganiser les rangs pour optimiser
    MPI_Cart_create(MPI_COMM_WORLD, 2, mpi_dims, periodic, 1, &mpi_cart_comm);

    MPI_Comm_rank(mpi_cart_comm, &rank);
    MPI_Comm_size(mpi_cart_comm, &size);
    MPI_Cart_coords(mpi_cart_comm, rank, 2, mpi_coords);

    parallel.task = rank;
    parallel.boss_task = 0;
    parallel.max_task = size;
    parallel.boss = (rank == 0);
}

void tea_decompose(int x_cells, int y_cells) {
    int err;
    
    // Décalages pour trouver les voisins (Shift)
    // Direction 1 (Y en Fortran), Direction 0 (X en Fortran)
    MPI_Cart_shift(mpi_cart_comm, 1, 1, 
                   &chunk.chunk_neighbours[CHUNK_BOTTOM], 
                   &chunk.chunk_neighbours[CHUNK_TOP]);
    MPI_Cart_shift(mpi_cart_comm, 0, 1, 
                   &chunk.chunk_neighbours[CHUNK_LEFT], 
                   &chunk.chunk_neighbours[CHUNK_RIGHT]);

    // Conversion MPI_PROC_NULL vers EXTERNAL_FACE
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

    // Calcul des limites locales (conversion logic 1-based Fortran vers 0-based C++)
    // X logic
    chunk.left = mpi_coords[0] * delta_x;
    chunk.left += std::min(mpi_coords[0], mod_x);
    
    chunk.right = chunk.left + delta_x - 1;
    if (mpi_coords[0] < mod_x) chunk.right++;

    // Y logic
    chunk.bottom = mpi_coords[1] * delta_y;
    chunk.bottom += std::min(mpi_coords[1], mod_y);
    
    chunk.top = chunk.bottom + delta_y - 1;
    if (mpi_coords[1] < mod_y) chunk.top++;

    if (parallel.boss) {
        std::cout << "\nMesh ratio of " << (double)x_cells/y_cells << std::endl;
        std::cout << "Decomposing the mesh into " << chunk_x << " by " << chunk_y << " chunks" << std::endl;
    }
}

void tea_decompose_tiles(int x_cells, int y_cells) {
    int tiles_x, tiles_y, mod_x, mod_y;
    int delta_x, delta_y;
    double best_fit_v = 0.0;
    int best_fit_i = 0;

    // 1. Trouver la meilleure division pour les tiles par tâche
    for (int i = 1; i <= tiles_per_task; ++i) {
        if (tiles_per_task % i != 0) continue;
        int j = tiles_per_task / i;

        // On cherche le ratio le plus proche de 1 (carré)
        double fit_v = (double)std::min(x_cells / i, y_cells / j) / 
                       (double)std::max(x_cells / i, y_cells / j);

        if (fit_v > best_fit_v) {
            best_fit_v = fit_v;
            best_fit_i = i;
        }
    }

    if (best_fit_i == 0) {
        std::cerr << "No fit found - tiles_per_task=" << tiles_per_task << std::endl;
        exit(1);
    }

    chunk.tile_dims[0] = best_fit_i;
    chunk.tile_dims[1] = tiles_per_task / best_fit_i;

    // 2. Calcul des dimensions de base
    tiles_x = chunk.tile_dims[0];
    tiles_y = chunk.tile_dims[1];
    delta_x = x_cells / tiles_x;
    delta_y = y_cells / tiles_y;
    mod_x = x_cells % tiles_x;
    mod_y = y_cells % tiles_y;

    // Redimensionner le vecteur de tiles (0-based indexing)
    chunk.tiles.resize(tiles_per_task);

    // 3. Boucle de création des tiles
    for (int j = 0; j < tiles_x; ++j) {
        for (int k = 0; k < tiles_y; ++k) {
            int t = j * tiles_y + k; // Index 0 à tiles_per_task-1

            chunk.tiles[t].tile_coords[0] = j;
            chunk.tiles[t].tile_coords[1] = k;

            // Calcul des bords de la tile (logique Fortran adaptée au 0-based)
            chunk.tiles[t].left = chunk.left + j * delta_x + std::min(j, mod_x);
            chunk.tiles[t].right = chunk.tiles[t].left + delta_x - 1;
            if (j < mod_x) chunk.tiles[t].right++;

            chunk.tiles[t].bottom = chunk.bottom + k * delta_y + std::min(k, mod_y);
            chunk.tiles[t].top = chunk.tiles[t].bottom + delta_y - 1;
            if (k < mod_y) chunk.tiles[t].top++;

            // Voisins des tiles
            std::fill(chunk.tiles[t].tile_neighbours, chunk.tiles[t].tile_neighbours + 4, EXTERNAL_FACE);
            
            if (j > 0)           chunk.tiles[t].tile_neighbours[CHUNK_LEFT]   = (j - 1) * tiles_y + k;
            if (j < tiles_x - 1) chunk.tiles[t].tile_neighbours[CHUNK_RIGHT]  = (j + 1) * tiles_y + k;
            if (k > 0)           chunk.tiles[t].tile_neighbours[CHUNK_BOTTOM] = j * tiles_y + (k - 1);
            if (k < tiles_y - 1) chunk.tiles[t].tile_neighbours[CHUNK_TOP]    = j * tiles_y + (k + 1);
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

    std::vector<int> lr_offset(NUM_FIELDS, 0);
    std::vector<int> bt_offset(NUM_FIELDS, 0);
    int end_pack_lr = 0, end_pack_bt = 0;

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

    // --- PHASE 1 : ÉCHANGE HORIZONTAL (LEFT / RIGHT) ---
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

    // --- PHASE 2 : ÉCHANGE VERTICAL (BOTTOM / TOP) ---
    msg_count = 0; // On réinitialise le compteur pour les requêtes verticales
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

void tea_send_recv_message_right(double* snd_buf, double* rcv_buf, int size, 
                                 int tag_send, int tag_recv, 
                                 MPI_Request* req_send, MPI_Request* req_recv) {
    int right_task = chunk.chunk_neighbours[CHUNK_RIGHT];
    MPI_Isend(snd_buf, size, MPI_DOUBLE, right_task, tag_send, mpi_cart_comm, req_send);
    MPI_Irecv(rcv_buf, size, MPI_DOUBLE, right_task, tag_recv, mpi_cart_comm, req_recv);
}

void tea_send_recv_message_top(double* snd_buf, double* rcv_buf, int size, 
                               int tag_send, int tag_recv, 
                               MPI_Request* req_send, MPI_Request* req_recv) {
    int top_task = chunk.chunk_neighbours[CHUNK_TOP];
    MPI_Isend(snd_buf, size, MPI_DOUBLE, top_task, tag_send, mpi_cart_comm, req_send);
    MPI_Irecv(rcv_buf, size, MPI_DOUBLE, top_task, tag_recv, mpi_cart_comm, req_recv);
}

void tea_send_recv_message_bottom(double* snd_buf, double* rcv_buf, int size, 
                                  int tag_send, int tag_recv, 
                                  MPI_Request* req_send, MPI_Request* req_recv) {
    int bottom_task = chunk.chunk_neighbours[CHUNK_BOTTOM];
    MPI_Isend(snd_buf, size, MPI_DOUBLE, bottom_task, tag_send, mpi_cart_comm, req_send);
    MPI_Irecv(rcv_buf, size, MPI_DOUBLE, bottom_task, tag_recv, mpi_cart_comm, req_recv);
}

void tea_finalize() {
    MPI_Finalize();
}

} // namespace TeaLeaf