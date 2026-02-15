#include "include/definitions.h"
#include <chrono>

namespace TeaLeaf {

    // State handling
    // Stores the physical properties (density, energy) for different regions of the grid
    std::vector<State_type> states;
    int number_of_states = 0;

    // Grid
    // Global grid metadata including physical domain bounds and cell counts
    Grid_type grid{};

    // Global control variables
    int step = 0;               // Current simulation time step index
    int error_condition = 0;    // Global error flag for MPI synchronization
    int test_problem = 0;       // Identifier for specific QA test cases
    bool complete = false;      // Flag indicating if the end criteria are met

    // Flags for selecting the numerical solver algorithm
    bool use_fortran_kernels = false;
    bool tl_use_chebyshev = false;
    bool tl_use_cg = false;
    bool tl_use_ppcg = false;
    bool tl_use_dpcg = false;
    bool tl_use_jacobi = false;
    bool tl_ppcg_active = false;
    bool verbose_on = false;

    // Solver convergence parameters
    int max_iters = 0;          // Maximum allowable iterations for the linear solver
    double eps = 0.0;           // Convergence tolerance (epsilon)
    int coefficient = 0;        // Material coefficient type (e.g., Conductivity)

    // Chebyshev / CG options
    double tl_ch_cg_epslim = 0.0;
    int tl_ch_cg_presteps = 0;
    bool tl_check_result = false;

    // PPCG options
    int tl_ppcg_inner_steps = 0;
    double tl_ppcg_steps_eigmin = 0.0;

    int tl_ppcg_inner_coarse_steps = 0;
    double tl_ppcg_coarse_eigmin = 0.0;

    // Boundary and preconditioning
    bool reflective_boundary = false;
    int tl_preconditioner_type = 0;

    bool use_vector_loops = false;
    bool profiler_on = false;

    // Coarse solve
    // Settings for multi-grid or coarse-level acceleration
    int coarse_solve_max_iters = 0;
    double corase_solve_max_eps = 0.0;
    bool coarse_solve_ppcg = false;

    // Profiler
    // Tracks accumulated wall-clock time for various kernels and MPI communications
    Profiler_type profiler{
        0.0, // timestep
        0.0, // visit
        0.0, // summary
        0.0, // tea_init
        0.0, // tea_solve
        0.0, // tea_reset
        0.0, // set_field
        0.0, // dot_product
        0.0, // halo_update
        0.0, // internal_halo_update
        0.0  // halo_exchange
    };

    // --- Time control ---
    double end_time = 0.0;      // Simulation physical time limit
    int end_step = 0;           // Simulation iteration limit

    double dt = 0.0;            // Current time step size (delta T)
    double timee = 0.0;         // Current elapsed physical simulation time
    double dtinit = 0.0;        // Initial user-requested time step

    int visit_frequency = 0;    // Steps between VTK visualization dumps
    int summary_frequency = 0;  // Steps between text log summaries

    int jdt = 0;                // Index of the cell determining the timestep limit (X)
    int kdt = 0;                // Index of the cell determining the timestep limit (Y)

    // Chunk
    // The primary data structure containing the local MPI rank's tiles and fields
    Chunk_type chunk{};

    /**
     * High-resolution timer utility.
     * @return Current steady-clock time in seconds.
     */
    double timer() {
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch());
        return duration.count() * 1e-9;
    }

}