#pragma once
namespace TeaLeaf {

    void tea_leaf_cg_init(double& error);
    void tea_leaf_cg_calc_w(double& pw);
    void tea_leaf_cg_calc_ur(double alpha, double& error);
    void tea_leaf_cg_calc_p(double beta);

} // namespace TeaLeaf
