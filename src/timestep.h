#ifndef TIMESTEP_H
#define TIMESTEP_H

namespace TeaLeaf {
    
    void timestep();

    void calc_dt_kernel(int tile_idx, double& dtlp);


    void tea_min(double& val);

} // namespace TeaLeaf

#endif