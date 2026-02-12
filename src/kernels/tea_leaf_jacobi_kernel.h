#pragma once

#include <vector>

namespace TeaLeaf {

    void tea_leaf_jacobi_solve_kernel(
        int x_min, int x_max, int y_min, int y_max,
        int halo_depth, double rx, double ry,
        const std::vector<double>& Kx,
        const std::vector<double>& Ky,
        double& error,
        const std::vector<double>& u0,
        std::vector<double>& u1,
        std::vector<double>& un);

} // namespace TeaLeaf
