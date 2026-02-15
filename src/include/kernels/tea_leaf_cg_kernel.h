#pragma once

#include <vector>

namespace TeaLeaf {


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
                const std::vector<double>& cp, 
                const std::vector<double>& bfp, 
                double rx, double ry,
                double& error,
                int preconditioner_type
            );

    void tea_leaf_cg_calc_w_kernel(
                int x_min, int x_max, int y_min, int y_max,
                int halo_depth,
                const std::vector<double>& p, 
                std::vector<double>& w, 
                const std::vector<double>& Kx,
                const std::vector<double>& Ky,
                const std::vector<double>& Di,
                double rx, double ry,
                double& pw
            );

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
                int preconditioner_type
            );

    void tea_leaf_cg_calc_p_kernel(
                int x_min, int x_max, int y_min, int y_max,
                int halo_depth,
                std::vector<double>& p, 
                const std::vector<double>& r, 
                const std::vector<double>& z,
                double beta,
                int preconditioner_type
            );

} // namespace TeaLeaf