#pragma once

namespace TeaLeaf {

    void tea_leaf_init_common();
    void tea_leaf_calc_residual();
    void tea_leaf_calc_2norm(int norm_array, double& norm);
    void tea_leaf_finalise();

} // namespace TeaLeaf
