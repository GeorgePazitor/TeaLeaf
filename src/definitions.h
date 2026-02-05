#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <vector>
#include <array>

namespace TeaLeaf {

    struct State_type{
        bool defined;

        double density, energy;
        int geometry;
        double xmin, xmax, ymin, ymax, radius;

    };

    int number_of_states;

    struct Grid_type{
        double xmin, xmax, ymin, ymax;

        int x_cells, y_cells;
    };

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

    struct Profiler_type{
        double timestep       
                ,visit           
                ,summary         
                ,tea_init                                  
                ,tea_solve       
                ,tea_reset       
                ,set_field       
                ,dot_product     
                ,halo_update     
                ,internal_halo_update     
                ,halo_exchange;
    };

    Profiler_type profiler;

    double end_time;

    int end_step;

    double dt, time, dtinit;

    int visit_frequency, summary_frequency;
    int jdt, kdt;

    struct Field_type{

        // 2D arrays stored as 1D vectors in row-major order
        std::vector<double> density, 
                            energy0, 
                            energy1, 
                            u, 
                            u0, 
                            vector_p, 
                            vector_r, 
                            vector_r_store,
                            vector_Mi, 
                            vector_w, 
                            vector_z, 
                            vector_utemp, 
                            vector_rtemp, 
                            vector_Di, 
                            vector_Kx, 
                            vector_Ky, 
                            vector_sd, 
                            tri_cp, 
                            tri_bfp, 
                            row_sums;

        std::vector<double> cellx, celly, vertexx, vertexy, celldx, celldy, vertexdx, vertexdy;
        
        // 2D arrays stored as 1D vectors in row-major order
        std::vector<double> volume, xarea, yarea;

        int xmin, xmax, ymin, ymax;
        double rx, ry;
    };

    struct Tile_type{

        Field_type field;

        int left, right, bottom, top;

        int x_cells, y_cells;

        std::array<int, 2> tile_neighbours;
        std::array<int, 2> tile_coords;       
    };

    struct Chunk_type{
        int task;

        int chunk_x_min, chunk_y_min, chunk_x_max, chunk_y_max;

        int left, right, bottom, top;

        int x_cells, y_cells;

        std::array<int, 4> chunk_neighbours;

        std::vector<double> left_rcv_buffer, right_rcv_buffer, bottom_rcv_buffer,top_rcv_buffer;
        std::vector<double> left_snd_buffer, right_snd_buffer, bottom_snd_buffer,top_snd_buffer;

        std::vector<Tile_type> tiles;
        
        std::array<int, 2> tile_dims;

        std::array<int, 2> sub_tile_dims;

        int halo_exchange_depth;
    };

    Chunk_type chunk;

    Grid_type grid;
}

#endif