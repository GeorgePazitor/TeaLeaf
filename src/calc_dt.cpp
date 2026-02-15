#include "include/calc_dt.h"
#include "include/data.h"
#include "include/definitions.h"

namespace TeaLeaf {

    //can serve to compute a stable time step based on the current state of the simulation but for now we just set it to the initial value.
    void calc_dt(double& local_dt) {
        
        
        local_dt = dtinit;
        
    }

} // namespace TeaLeaf