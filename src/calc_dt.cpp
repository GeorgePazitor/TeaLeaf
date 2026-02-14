#include "include/calc_dt.h"
#include "include/data.h"
#include "include/definitions.h"

namespace TeaLeaf {

    /**
     * Calculates the time step size (dt) for the current iteration.
     * In this implementation, it sets the local time step to the initial 
     * user-defined value.
     */
    void calc_dt(double& local_dt) {
        
        // Assign the initial time step value defined during input parsing.
        // In more complex hydrodynamics codes, this would involve CFL 
        // stability checks across all grid cells.
        local_dt = dtinit;
        
    }

} // namespace TeaLeaf