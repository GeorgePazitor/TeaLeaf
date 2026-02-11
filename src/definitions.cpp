#include "definitions.h"

namespace TeaLeaf {

    // --- State handling ---
    std::vector<State_type> states;
    int number_of_states = 0;

    // --- Grid ---
    Grid_type grid{};

    // --- Global control variables ---
    int step = 0;
    int error_condition = 0;
    int test_problem = 0;
    bool complete = false;

    bool use_fortran_kernels = false;
    bool tl_use_chebyshev = false;
    bool tl_use_cg = false;
    bool tl_use_ppcg = false;
    bool tl_use_dpcg = false;
    bool tl_use_jacobi = false;
    bool tl_ppcg_active = false;
    bool verbose_on = false;

    int max_iters = 0;
    double eps = 0.0;
    int coefficient = 0;

    // --- Chebyshev / CG options ---
    double tl_ch_cg_epslim = 0.0;
    int tl_ch_cg_presteps = 0;
    bool tl_check_result = false;

    // --- PPCG options ---
    int tl_ppcg_inner_steps = 0;
    double tl_ppcg_steps_eigmin = 0.0;

    int tl_ppcg_inner_coarse_steps = 0;
    double tl_ppcg_coarse_eigmin = 0.0;

    // --- Boundary and preconditioning ---
    bool reflective_boundary = false;
    int tl_preconditioner_type = 0;

    bool use_vector_loops = false;
    bool profiler_on = false;

    // --- Coarse solve ---
    int coarse_solve_max_iters = 0;
    double corase_solve_max_eps = 0.0;
    bool coarse_solve_ppcg = false;

    // --- Profiler ---
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
    double end_time = 0.0;
    int end_step = 0;

    double dt = 0.0;
    double timee = 0.0;
    double dtinit = 0.0;

    int visit_frequency = 0;
    int summary_frequency = 0;

    int jdt = 0;
    int kdt = 0;

    // --- Chunk ---
    Chunk_type chunk{};

}
