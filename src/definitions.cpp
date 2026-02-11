#include "definitions.h"
#include <vector>
#include <array>

namespace TeaLeaf {

    int number_of_states;
    
    int step;
    int error_condition;
    int test_problem;
    bool complete;

    bool use_fortran_kernels;
    bool tl_use_chebyshev;
    bool tl_use_cg;
    bool tl_use_ppcg;
    bool tl_use_dpcg;
    bool tl_use_jacobi;
    bool tl_ppcg_active;
    bool verbose_on;
    int max_iters;

    double eps;
    int coefficient;


    // error to run cg to before calculating eigenvalues
    double tl_ch_cg_epslim;
    // do b-Ax after finishing to make sure solver actually converged
    int tl_ch_cg_presteps;

    bool tl_check_result;
    int tl_ppcg_inner_steps;
    double tl_ppcg_steps_eigmin;

    int tl_ppcg_inner_coarse_steps;
    double tl_ppcg_coarse_eigmin;

    bool reflective_boundary;

    int tl_preconditioner_type;

    bool use_vector_loops;

    bool profiler_on;

    int coarse_solve_max_iters;
    double corase_solve_max_eps;
    bool coarse_solve_ppcg;
    
    
    double end_time;

    int end_step;

    double dt, timee, dtinit;

    int visit_frequency, summary_frequency;
    int jdt, kdt;
    
    std::vector<State_type> states;

    Profiler_type profiler;

    Chunk_type chunk;

    Grid_type grid;
}
