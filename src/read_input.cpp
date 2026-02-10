#include "read_input.h"
using namespace TeaLeaf;

class InputParser {

std::ifstream file;
std::istringstream current_line_ss;
std::string current_line;

public:
    InputParser(const std::string& filename) {
        file.open(filename);
        if (!file.is_open()) {
            std::cerr <<"read_input: " << "Could not open input file: " << filename;
        }
    }

    void reset() {
        file.clear();
        file.seekg(0, std::ios::beg);
    }

    bool next_line() {
        if (std::getline(file, current_line)) {
            // Replace '=' with space to handle "key=value" formats easily
            std::replace(current_line.begin(), current_line.end(), '=', ' ');
            
            current_line_ss.clear();
            current_line_ss.str(current_line);
            return true;
        }
        return false;
    }

    std::string get_word() {
        std::string word;
        if (current_line_ss >> word) {
            return word;
        }
        return "";
    }

    int get_int() {
        std::string w = get_word();
        try { return std::stoi(w); }
        catch (...) { return 0; }
    }

    double get_double() {
        std::string w = get_word();
        try { return std::stod(w); }
        catch (...) { return 0.0; }
    }
};

void read_input() {
    using namespace TeaLeaf; 
    
    test_problem = 0;
    int state_max = 0;

    grid.xmin = 0.0;
    grid.ymin = 0.0;
    grid.xmax = 100.0;
    grid.ymax = 100.0;
    grid.x_cells = 10;
    grid.y_cells = 10;

    end_time = 10.0;
    end_step = g_ibig;
    complete = false;

    visit_frequency = 0;
    summary_frequency = 10;
    dtinit = 0.1;

    max_iters = 1000;
    eps = 1.0e-10;

    use_fortran_kernels = true; 
    coefficient = CONDUCTIVITY;
    
    profiler_on = false;
    // Assuming Profiler_type has a default constructor or we zero it manually:
    profiler = Profiler_type(); 

    tiles_per_task = 1;
    sub_tiles_per_tile = 1;

    // OpenMP Thread Check
    #pragma omp parallel
    {
        #pragma omp master
        {
            #ifdef _OPENMP
            tiles_per_task = omp_get_num_threads();
            #endif
        }
    }

    tl_ch_cg_presteps = 25;
    tl_ch_cg_epslim = 1.0;
    tl_check_result = false;
    tl_preconditioner_type = TL_PREC_NONE;
    reflective_boundary = false;
    tl_ppcg_inner_steps = -1;

    tl_use_chebyshev = false;
    tl_use_cg = false;
    tl_use_ppcg = false;
    tl_use_jacobi = true;
    verbose_on = false;

    chunk.halo_exchange_depth = 1;

    if (parallel.boss) {
        *g_out << "Reading input file\n\n";
    }

    
    //scan states to know how many states exist to allocate memory
    
    InputParser parser("tea.in.tmp");

    while (parser.next_line()) {
        while (true) {
            std::string word = parser.get_word();
            if (word == "") break;
            
            if (word == "state") {
                int val = parser.get_int();
                if (val > state_max) state_max = val;
                // Move to next line after finding state ID
                break; 
            }
        }
    }

    number_of_states = state_max;
    if (number_of_states < 1) {
        std::cerr <<"read_input: " << "No states defined.";
    }

    // 1-based indexing to match Fortran ID logic
    states.resize(number_of_states + 1);

    for(auto& s : states) {
        s.defined = false;
        s.energy = 0.0;
        s.density = 0.0;
    }

    // actual reading of the states
    
    parser.reset();

    while (parser.next_line()) {
        while (true) {
            std::string word = parser.get_word();
            if (word == "") break;

            if (word == "initial_timestep") {
                dtinit = parser.get_double();
                if (parallel.boss) *g_out << " initial_timestep        " << dtinit << "\n";
            }
            else if (word == "end_time") {
                end_time = parser.get_double();
                if (parallel.boss) *g_out << " end_time                " << end_time << "\n";
            }
            else if (word == "end_step") {
                end_step = parser.get_int();
                if (parallel.boss) *g_out << " end_step                " << end_step << "\n";
            }
            else if (word == "xmin") {
                grid.xmin = parser.get_double();
                if (parallel.boss) *g_out << " xmin                    " << grid.xmin << "\n";
            }
            else if (word == "xmax") {
                grid.xmax = parser.get_double();
                if (parallel.boss) *g_out << " xmax                    " << grid.xmax << "\n";
            }
            else if (word == "ymin") {
                grid.ymin = parser.get_double();
                if (parallel.boss) *g_out << " ymin                    " << grid.ymin << "\n";
            }
            else if (word == "ymax") {
                grid.ymax = parser.get_double();
                if (parallel.boss) *g_out << " ymax                    " << grid.ymax << "\n";
            }
            else if (word == "x_cells") {
                grid.x_cells = parser.get_int();
                if (parallel.boss) *g_out << " x_cells                 " << grid.x_cells << "\n";
            }
            else if (word == "y_cells") {
                grid.y_cells = parser.get_int();
                if (parallel.boss) *g_out << " y_cells                 " << grid.y_cells << "\n";
            }
            else if (word == "visit_frequency") {
                visit_frequency = parser.get_int();
                if (parallel.boss) *g_out << " visit_frequency         " << visit_frequency << "\n";
            }
            else if (word == "summary_frequency") {
                summary_frequency = parser.get_int();
                if (parallel.boss) *g_out << " summary_frequency       " << summary_frequency << "\n";
            }
            else if (word == "tiles_per_task") {
                tiles_per_task = parser.get_int();
                if (parallel.boss) *g_out << " tiles_per_task          " << tiles_per_task << "\n";
            }
            else if (word == "sub_tiles_per_tile") {
                sub_tiles_per_tile = parser.get_int();
                if (parallel.boss) *g_out << " sub_tiles_per_tile      " << sub_tiles_per_tile << "\n";
            }
            else if (word == "tl_max_iters") {
                max_iters = parser.get_int();
            }
            else if (word == "tl_eps") {
                eps = parser.get_double();
            }
            else if (word == "tl_use_jacobi") {
                tl_use_jacobi = true;
                tl_use_cg = false;
                tl_use_chebyshev = false;
                tl_use_ppcg = false;
            }
            else if (word == "tl_use_cg") {
                tl_use_cg = true;
                tl_use_jacobi = false;
                tl_use_chebyshev = false;
                tl_use_ppcg = false;
            }
            else if (word == "tl_use_ppcg") {
                tl_use_ppcg = true;
                tl_use_cg = false;
                tl_use_jacobi = false;
                tl_use_chebyshev = false;
            }
            else if (word == "tl_use_chebyshev") {
                tl_use_chebyshev = true;
                tl_use_cg = false;
                tl_use_jacobi = false;
                tl_use_ppcg = false;
            }
            else if (word == "profiler_on") {
                profiler_on = true;
                if (parallel.boss) *g_out << " Profiler on\n";
            }
            else if (word == "state") {
                // Read State ID
                int id = parser.get_int();
                
                if (parallel.boss) *g_out << " Reading specification for state " << id << "\n";
                
                if (id > number_of_states || id < 1) std::cerr << "read_input: " << "Invalid state ID";
                if (states[id].defined) std::cerr << "read_input: " << "State defined twice";

                states[id].defined = true;

                // Inner Loop for State Properties
                while(true) {
                    std::string sw = parser.get_word();
                    if (sw == "") break; // End of line

                    if (sw == "xmin") {
                        states[id].xmin = parser.get_double();
                        if (parallel.boss) *g_out << " state xmin              " << states[id].xmin << "\n";
                    }
                    else if (sw == "xmax") {
                        states[id].xmax = parser.get_double();
                        if (parallel.boss) *g_out << " state xmax              " << states[id].xmax << "\n";
                    }
                    else if (sw == "ymin") {
                        states[id].ymin = parser.get_double();
                        if (parallel.boss) *g_out << " state ymin              " << states[id].ymin << "\n";
                    }
                    else if (sw == "ymax") {
                        states[id].ymax = parser.get_double();
                        if (parallel.boss) *g_out << " state ymax              " << states[id].ymax << "\n";
                    }
                    else if (sw == "radius") {
                        states[id].radius = parser.get_double();
                        if (parallel.boss) *g_out << " state radius            " << states[id].radius << "\n";
                    }
                    else if (sw == "density") {
                        states[id].density = parser.get_double();
                        if (parallel.boss) *g_out << " state density           " << states[id].density << "\n";
                    }
                    else if (sw == "energy") {
                        states[id].energy = parser.get_double();
                        if (parallel.boss) *g_out << " state energy            " << states[id].energy << "\n";
                    }
                    else if (sw == "geometry") {
                        std::string geom = parser.get_word();
                        if (geom == "rectangle") {
                            states[id].geometry = g_rect;
                            if (parallel.boss) *g_out << " state geometry rectangular\n";
                        } else if (geom == "circle") {
                            states[id].geometry = g_circ;
                            if (parallel.boss) *g_out << " state geometry circular\n";
                        } else if (geom == "point") {
                            states[id].geometry = g_point;
                            if (parallel.boss) *g_out << " state geometry point\n";
                        }
                    }
                }
                if (parallel.boss) *g_out << "\n";
            }
        }
    }

    // --- 4. LOGIC CHECKS & ADJUSTMENTS ---

    // Heuristic for PPCG steps if not set
    if (tl_ppcg_inner_steps == -1 && tl_use_ppcg) {
        double total_cells = (double)grid.x_cells * (double)grid.y_cells;
        tl_ppcg_inner_steps = 4 * (int)std::sqrt(std::sqrt(total_cells));
        if (parallel.boss) *g_out << " tl_ppcg_inner_steps     " << tl_ppcg_inner_steps << "\n";
    }

    if (chunk.halo_exchange_depth > 1 && tl_preconditioner_type == TL_PREC_JAC_BLOCK) {
        std::cerr <<"read_input: " << "Unable to use nonstandard halo depth with block jacobi preconditioner";
    }

    if (parallel.boss) {
        *g_out << " tiles per task          " << tiles_per_task << "\n\n";
        if (use_fortran_kernels) {
            *g_out << " Using Fortran Kernels\n";
        }
        *g_out << "\n Input read finished.\n\n";
        g_out->flush();
    }

    // Adjust State Boundaries to avoid floating point errors on cell edges
    double dx = (grid.xmax - grid.xmin) / (double)grid.x_cells;
    double dy = (grid.ymax - grid.ymin) / (double)grid.y_cells;

    // Fortran loop was DO n=2,number_of_states. 
    // State 1 is background, usually covers everything, so no need to shrink it.
    for (int n = 2; n <= number_of_states; ++n) {
        states[n].xmin += (dx / 100.0);
        states[n].ymin += (dy / 100.0);
        states[n].xmax -= (dx / 100.0);
        states[n].ymax -= (dy / 100.0);
    }
}