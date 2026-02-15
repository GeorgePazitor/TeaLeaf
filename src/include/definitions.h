#pragma once

#include <vector>
#include <array>

namespace TeaLeaf {

    struct State_type{
        bool defined;

        double density, energy;
        int geometry;
        double xmin, xmax, ymin, ymax, radius;

    };

    extern int number_of_states;

    struct Grid_type{
        double xmin, xmax, ymin, ymax;

        int x_cells, y_cells;
    };

    extern int step;
    extern int error_condition;
    extern int test_problem;
    extern bool complete;

    extern bool use_fortran_kernels; // TODO  : remove this flag and associated code
    extern bool tl_use_chebyshev;
    extern bool tl_use_cg;
    extern bool tl_use_ppcg;
    extern bool tl_use_dpcg;
    extern bool tl_use_jacobi;
    extern bool tl_ppcg_active;
    extern bool verbose_on;
    extern int max_iters;

    extern double eps;
    extern int coefficient;

    extern double tl_ch_cg_epslim;

    extern int tl_ch_cg_presteps;

    extern bool tl_check_result;
    extern int tl_ppcg_inner_steps;
    extern double tl_ppcg_steps_eigmin;

    extern int tl_ppcg_inner_coarse_steps;
    extern double tl_ppcg_coarse_eigmin;

    extern  bool reflective_boundary;

    extern int tl_preconditioner_type;

    extern bool use_vector_loops;

    extern bool profiler_on;

    extern int coarse_solve_max_iters;
    extern double corase_solve_max_eps;
    extern bool coarse_solve_ppcg;

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

    extern double end_time;

    extern int end_step;

    extern double dt, timee, dtinit;

    extern int visit_frequency, summary_frequency;
    extern int jdt, kdt;

    struct Field_type{

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

        // vertexx, vertexy,  vertices coordinates define the corners of the cells.
        // cellx, celly, cell coordinates, they are located halfway between vertices.
        // celldx, celldy, define the grid spacing in each direction (constant if uniform grid)
        std::vector<double> cellx, celly, vertexx, vertexy, celldx, celldy, vertexdx, vertexdy;
        
        std::vector<double> volume, xarea, yarea;

        int x_min, x_max, y_min, y_max;
        double rx, ry;
    };

    struct Tile_type{

        Field_type field;

        int left, right, bottom, top;

        int x_cells, y_cells;

        std::array<int, 4> tile_neighbours; 
        std::array<int, 4> tile_coords;       
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

    extern std::vector<State_type> states;
    
    extern Profiler_type profiler;

    extern Chunk_type chunk;

    extern Grid_type grid;

    extern double timer();
 
}
