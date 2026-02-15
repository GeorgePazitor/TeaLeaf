#include "include/tea.h"
#include "include/data.h"
#include "include/definitions.h"
#include "include/kernels/tea_leaf_cg_kernel.h"

namespace TeaLeaf {

void tea_leaf_cg_init(double& error) {
    
    double total_tile_error = 0.0 ;

    #pragma omp parallel reduction(+:total_tile_error)
    {
        #pragma omp for
        for (int t = 0; t < tiles_per_task; ++t) {
            double tile_error = 0.0;
            auto& f = chunk.tiles[t].field;

            tea_leaf_cg_init_kernel(
                f.x_min, f.x_max, f.y_min, f.y_max,
                chunk.halo_exchange_depth,
                f.vector_p, f.vector_r, f.vector_Mi, f.vector_z,
                f.vector_Kx, f.vector_Ky, f.vector_Di,
                f.tri_cp, f.tri_bfp, f.rx, f.ry,
                tile_error,
                tl_preconditioner_type
            );

            total_tile_error += tile_error;
        }
    }
    error = total_tile_error;
}

void tea_leaf_cg_calc_w(double& pw) {
  
    double total_tile_pw = 0.0 ;

    #pragma omp parallel reduction(+:total_tile_pw)
    {
        #pragma omp for
        for (int t = 0; t < tiles_per_task; ++t) {
            double tile_pw = 0.0;
            auto& f = chunk.tiles[t].field;

            tea_leaf_cg_calc_w_kernel(
                f.x_min, f.x_max, f.y_min, f.y_max,
                chunk.halo_exchange_depth,
                f.vector_p, f.vector_w, 
                f.vector_Kx, f.vector_Ky, f.vector_Di,
                f.rx, f.ry,
                tile_pw
            );

            total_tile_pw += tile_pw;
        }
    }
    pw = total_tile_pw;
}

void tea_leaf_cg_calc_ur(double alpha, double& error) {
  
    double total_tile_error = 0.0 ;

    #pragma omp parallel reduction(+:total_tile_error)
    {
        #pragma omp for
        for (int t = 0; t < tiles_per_task; ++t) {
            double tile_error = 0.0;
            auto& f = chunk.tiles[t].field;

            tea_leaf_cg_calc_ur_kernel(
                f.x_min, f.x_max, f.y_min, f.y_max,
                chunk.halo_exchange_depth,
                f.u, f.vector_p, f.vector_r, f.vector_Mi, f.vector_w, f.vector_z,
                f.tri_cp, f.tri_bfp,
                f.vector_Kx, f.vector_Ky, f.vector_Di,
                f.rx, f.ry,
                alpha,
                tile_error,
                tl_preconditioner_type
            );

            total_tile_error += tile_error;
        }
    }
    error = total_tile_error;
}

void tea_leaf_cg_calc_p(double beta) {

    #pragma omp parallel 
    {
        #pragma omp for
        for (int t = 0; t < tiles_per_task; ++t) {
            auto& f = chunk.tiles[t].field;

            tea_leaf_cg_calc_p_kernel(
                f.x_min, f.x_max, f.y_min, f.y_max,
                chunk.halo_exchange_depth,
                f.vector_p, f.vector_r, f.vector_z,
                beta,
                tl_preconditioner_type
            );
        }
    }
}
}