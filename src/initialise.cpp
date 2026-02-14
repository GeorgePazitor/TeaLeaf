#include "initialise.h"

#include <iostream>
#include <fstream>
//#include <mpi.h>
#include <omp.h>

#include "data.h"
#include "definitions.h"
#include "read_input.h"
#include "start.h"


std::ofstream g_file_stream;

//check file existence
bool file_exists(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

//strip comments and copy (replaces parse_module logic for this snippet)
void clean_input_file(const std::string& input_file, const std::string& output_file) {
    std::ifstream infile(input_file);
    std::ofstream outfile(output_file);
    
    if (!infile.is_open()) {
        std::cerr << "initialise: " << "Error opening input file for cleaning";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (!outfile.is_open())  {
        std::cerr << "initialise: " << "Error opening tmp file for writing";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::string line;
    while (std::getline(infile, line)) {
        //strip comments starting with ! or #
        size_t comment_pos = line.find_first_of("!#");
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }
        
        if (!line.empty()) {
            outfile << line << "\n";
        }
    }
}

void initialise() {
    using namespace TeaLeaf;

    if (parallel.boss) {
        g_file_stream.open("tea.out");
        if (!g_file_stream.is_open()) {
            std::cerr <<"initialise" << "Error opening tea.out file.";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        g_out = &g_file_stream;
    } else {
        //non-boss ranks write to stdout
        g_out = &std::cout; 
    }

    #pragma omp parallel
    {
        // only master rank prints
        #pragma omp master
        {
            if (parallel.boss) {
                *g_out << "\n";
                *g_out << "Tea Version      " << g_version << "\n";
                *g_out << "MPI Version\n";    // TODO - add mpi and open mpi versions            
                *g_out << "OpenMP Version\n";
                *g_out << "Task Count       " << parallel.max_task << "\n";                
                *g_out << "Thread Count:    " << omp_get_num_threads() << "\n";
                *g_out << "\n";
                *g_out << "Output file tea.out opened. All output will go there." << std::endl;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (parallel.boss) {
        *g_out << "Tea will run from the following input:-\n\n";
    }

    if (parallel.boss) {
        //check if tea.in exists
        if (!file_exists("../src/tea.in")) {
            // Generate default input file
            std::cout << "\nNo input file found. Generating default tea.in\n";
            std::ofstream out_unit("../src/tea.in");
            if (!out_unit.is_open()) {
                std::cerr << "initialise: " << "Error creating default tea.in";
                MPI_Abort(MPI_COMM_WORLD, 1);
            
            }

            out_unit << "*tea\n";
            out_unit << "state 1 density=100.0 energy=0.0001\n";
            out_unit << "state 2 density=0.1 energy=25.0 geometry=rectangle xmin=0.0 xmax=1.0 ymin=1.0 ymax=2.0\n";
            out_unit << "state 3 density=0.1 energy=0.1 geometry=rectangle xmin=1.0 xmax=6.0 ymin=1.0 ymax=2.0\n";
            out_unit << "state 4 density=0.1 energy=0.1 geometry=rectangle xmin=5.0 xmax=6.0 ymin=1.0 ymax=8.0\n";
            out_unit << "state 5 density=0.1 energy=0.1 geometry=rectangle xmin=5.0 xmax=10.0 ymin=7.0 ymax=8.0\n\n";
            out_unit << "x_cells=1000\n";
            out_unit << "y_cells=1000\n\n";
            out_unit << "xmin=0.0\n";
            out_unit << "ymin=0.0\n";
            out_unit << "xmax=10.0\n";
            out_unit << "ymax=10.0\n\n";
            out_unit << "initial_timestep=0.004\n";
            out_unit << "end_step=10\n\n";
            out_unit << "tl_max_iters=1000\n";
            out_unit << " test_problem 1\n";
            out_unit << "tl_use_jacobi\n";
            out_unit << "tl_eps=1.0e-15\n\n";
            out_unit << "*endtea\n";
            out_unit.close();
        }else{
            std::cout << "\nFile found, using the specified tea.in\n";
        }

        // Parse/Clean the input file to tea.in.tmp
        clean_input_file("../src/tea.in", "../src/tea.in.tmp");
    }

    // wait for boss to finish file i/o
    MPI_Barrier(MPI_COMM_WORLD);

    // boss reads the original file just to echo it to the log
    if (parallel.boss) {
        std::ifstream uin("../src/tea.in");
        if (uin.is_open()) {
            std::string ltmp;
            while (std::getline(uin, ltmp)) {
                *g_out << ltmp << "\n";
            }
        }
    }

    if (parallel.boss) {
        *g_out << "\nInitialising and generating\n\n";
    }

    // 
    // check tea.in, generate if missing, clean to tea.in.tmp, broadcast/read

    // reads from tea.in.tmp 
    read_input();

    MPI_Barrier(MPI_COMM_WORLD);

    step = 0;

    start();

    MPI_Barrier(MPI_COMM_WORLD);

    if (parallel.boss) {
        *g_out << "Starting the calculation\n";
    }

}