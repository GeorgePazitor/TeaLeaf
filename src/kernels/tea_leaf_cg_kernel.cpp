#include <cmath>
#include <vector>
#include "include/data.h"
#include "include/definitions.h"
namespace TeaLeaf {


/*void tea_block_solve(int x_min, int x_max, int y_min, int y_max, int halo_depth,
                     const std::vector<double>& r, std::vector<double>& z,
                     const std::vector<double>& cp, const std::vector<double>& bfp,
                     const std::vector<double>& Kx, const std::vector<double>& Ky,
                     const std::vector<double>& Di, double rx, double ry);

void tea_diag_solve(int x_min, int x_max, int y_min, int y_max, int halo_depth, int step,
                    const std::vector<double>& r, std::vector<double>& z,
                    const std::vector<double>& Mi);
*/

void tea_leaf_cg_init_kernel(
                int x_min, int x_max, int y_min, int y_max,
                int halo_depth,
                std::vector<double>& p, 
                const std::vector<double>& r, 
                const std::vector<double>& Mi, 
                std::vector<double>& z,
                const std::vector<double>& Kx,
                const std::vector<double>& Ky,
                const std::vector<double>& Di,
                const std::vector<double>& cp, const std::vector<double>& bfp, 
                double rx, double ry,
                double& error,
                int preconditioner_type) 
{
    const int x_width = (x_max - x_min + 1) + 2 * halo_depth;
    #define IDX(j, k) (((k) - y_min + halo_depth) * x_width + ((j) - x_min + halo_depth))

    double local_rro = 0.0;

    for (int k = y_min; k <= y_max; ++k) {
        for (int j = x_min; j <= x_max; ++j) {
            p[IDX(j, k)] = 0.0;
            z[IDX(j, k)] = 0.0;
        }
    }

    /*if (preconditioner_type != TL_PREC_NONE) {
        if (preconditioner_type == TL_PREC_JAC_BLOCK) {
            tea_block_solve(x_min, x_max, y_min, y_max, halo_depth, r, z, cp, bfp, Kx, Ky, Di, rx, ry);
        } 
        else if (preconditioner_type == TL_PREC_JAC_DIAG) {
            tea_diag_solve(x_min, x_max, y_min, y_max, halo_depth, 0, r, z, Mi);
        }

        for (int k = y_min; k <= y_max; ++k) {
            for (int j = x_min; j <= x_max; ++j) {
                p[IDX(j, k)] = z[IDX(j, k)];
            }
        }
        } else {*/
        
        for (int k = y_min; k <= y_max; ++k) {
            for (int j = x_min; j <= x_max; ++j) {
                p[IDX(j, k)] = r[IDX(j, k)];
            }
        }
    //}

    
    for (int k = y_min; k <= y_max; ++k) {
        for (int j = x_min; j <= x_max; ++j) {
            local_rro += r[IDX(j, k)] * p[IDX(j, k)];
        }
    }

    error = local_rro;
    #undef IDX
}

void tea_leaf_cg_calc_w_kernel(
                int x_min, int x_max, int y_min, int y_max,
                int halo_depth,
                const std::vector<double>& p, 
                std::vector<double>& w, 
                const std::vector<double>& Kx,
                const std::vector<double>& Ky,
                const std::vector<double>& Di,
                double rx, double ry,
                double& pw) 
{
    const int x_width = (x_max - x_min + 1) + 2 * halo_depth;
    #define IDX(j, k) (((k) - y_min + halo_depth) * x_width + ((j) - x_min + halo_depth))

    double local_pw = 0.0;

    for (int k = y_min; k <= y_max; ++k) {
        for (int j = x_min; j <= x_max; ++j) {
            //matrix-vector product
            w[IDX(j, k)] = Di[IDX(j, k)] * p[IDX(j, k)]
                - ry * (Ky[IDX(j, k + 1)] * p[IDX(j, k + 1)] + Ky[IDX(j, k)] * p[IDX(j, k - 1)])
                - rx * (Kx[IDX(j + 1, k)] * p[IDX(j + 1, k)] + Kx[IDX(j, k)] * p[IDX(j - 1, k)]);
            
            
            local_pw += w[IDX(j, k)] * p[IDX(j, k)];
        }
    }

    pw = local_pw;
    #undef IDX
}

void tea_leaf_cg_calc_ur_kernel(
                int x_min, int x_max, int y_min, int y_max,
                int halo_depth,
                std::vector<double>& u,
                const std::vector<double>& p, 
                std::vector<double>& r, 
                const std::vector<double>& Mi, 
                const std::vector<double>& w, 
                std::vector<double>& z,
                const std::vector<double>& cp, const std::vector<double>& bfp,
                const std::vector<double>& Kx,
                const std::vector<double>& Ky,
                const std::vector<double>& Di,
                double rx, double ry,
                double alpha,
                double& error,
                int preconditioner_type)
{
    const int x_width = (x_max - x_min + 1) + 2 * halo_depth;
    #define IDX(j, k) (((k) - y_min + halo_depth) * x_width + ((j) - x_min + halo_depth))

    double local_rrn = 0.0;

    /*if (preconditioner_type != TL_PREC_NONE) {
        if (preconditioner_type == TL_PREC_JAC_DIAG) {
            for (int k = y_min; k <= y_max; ++k) {
                for (int j = x_min; j <= x_max; ++j) {
                    u[IDX(j, k)] += alpha * p[IDX(j, k)];
                    r[IDX(j, k)] -= alpha * w[IDX(j, k)];
                    z[IDX(j, k)] = Mi[IDX(j, k)] * r[IDX(j, k)];
                    local_rrn += r[IDX(j, k)] * z[IDX(j, k)];
                }
            }
        } 
        else if (preconditioner_type == TL_PREC_JAC_BLOCK) {
            for (int k = y_min; k <= y_max; ++k) {
                for (int j = x_min; j <= x_max; ++j) {
                    u[IDX(j, k)] += alpha * p[IDX(j, k)];
                    r[IDX(j, k)] -= alpha * w[IDX(j, k)];
                }
            }

            tea_block_solve(x_min, x_max, y_min, y_max, halo_depth, r, z, cp, bfp, Kx, Ky, Di, rx, ry);

            for (int k = y_min; k <= y_max; ++k) {
                for (int j = x_min; j <= x_max; ++j) {
                    local_rrn += r[IDX(j, k)] * z[IDX(j, k)];
                }
            }
        }
    } else {*/

        for (int k = y_min; k <= y_max; ++k) {
            for (int j = x_min; j <= x_max; ++j) {
                u[IDX(j, k)] += alpha * p[IDX(j, k)];
                r[IDX(j, k)] -= alpha * w[IDX(j, k)];
                local_rrn += r[IDX(j, k)] * r[IDX(j, k)];
            }
        }
    //}

    error = local_rrn;
    #undef IDX
}

void tea_leaf_cg_calc_p_kernel(int x_min, int x_max, int y_min, int y_max,
                int halo_depth,
                std::vector<double>& p, 
                const std::vector<double>& r, 
                const std::vector<double>& z,
                double beta,
                int preconditioner_type
            )
{
    const int x_width = (x_max - x_min + 1) + 2 * halo_depth;
    #define IDX(j, k) (((k) - y_min + halo_depth) * x_width + ((j) - x_min + halo_depth))

    /*if (preconditioner_type != TL_PREC_NONE || tl_ppcg_active) {
        for (int k = y_min; k <= y_max; ++k) {
            for (int j = x_min; j <= x_max; ++j) {
                p[IDX(j, k)] = z[IDX(j, k)] + beta * p[IDX(j, k)];
            }
        }
    }else {*/
        for (int k = y_min; k <= y_max; ++k) {
            for (int j = x_min; j <= x_max; ++j) {
                p[IDX(j, k)] = r[IDX(j, k)] + beta * p[IDX(j, k)];
            }
        }
    //}

    #undef IDX
}

} // namespace TeaLeaf