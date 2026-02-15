#pragma once

#include <vector>
#include <array>
#include "../data.h"

namespace TeaLeaf {

    void tea_leaf_common_init_kernel(
        int x_min, int x_max, int y_min, int y_max, int halo,
        const std::array<bool, 4>& zero_boundary, bool reflective_boundary,
        const std::vector<double>& density, const std::vector<double>& energy,
        std::vector<double>& u, std::vector<double>& u0,
        std::vector<double>& r, std::vector<double>& w,
        std::vector<double>& Kx, std::vector<double>& Ky,
        std::vector<double>& Di, std::vector<double>& cp,
        std::vector<double>& bfp, std::vector<double>& Mi,
        double rx, double ry, int preconditioner_type, int coef);

    void tea_leaf_kernel_finalise(
        int x_min, int x_max, int y_min, int y_max, int halo,
        std::vector<double>& energy, const std::vector<double>& density,
        const std::vector<double>& u);

    void tea_leaf_calc_residual_kernel(
        int x_min, int x_max, int y_min, int y_max, int halo,
        const std::vector<double>& u, const std::vector<double>& u0,
        std::vector<double>& r, const std::vector<double>& Kx,
        const std::vector<double>& Ky, const std::vector<double>& Di,
        double rx, double ry);

    void tea_leaf_calc_2norm_kernel(
        int x_min, int x_max, int y_min, int y_max, int halo,
        const std::vector<double>& arr, double& norm);

    void tea_diag_init(
        int x_min, int x_max, int y_min, int y_max, int halo,
        std::vector<double>& Mi, const std::vector<double>& Kx, 
        const std::vector<double>& Ky, const std::vector<double>& Di, 
        double rx, double ry);

} // namespace TeaLeaf

