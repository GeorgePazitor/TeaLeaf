#include "include/initialise.h"
#include <iostream>
#include <fstream>
#include <omp.h>
#include "include/data.h"
#include "include/definitions.h"
#include "include/read_input.h"
#include "include/start.h"

std::ofstream g_file_stream;

/**
 * Checks if a file exists and is readable.
 */
bool file_exists(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

/**
 * Pre-processes the input file: removes comments and empty lines.
 * Outputs to a temporary file for easier parsing.
 */
void clean_input_file(const std::string& input_file, const std::string& output_file) {
    std::ifstream infile(input_file);
    std::ofstream outfile(output_file);
    
    if (!infile.is_open()) {
        std::cerr << "initialise: Error opening input file for cleaning\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (!outfile.is_open())  {
        std::cerr << "initialise: Error opening tmp file for writing\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::string line;
    while (std::getline(infile, line)) {
        // Strip comments starting with ! (Fortran style) or # (Script style)
        size_t comment_pos = line.find_first_of("!#");
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }
        
        // Only write non-empty lines to the temp file
        if (!line.empty()) {
            outfile << line << "\n";
        }
    }
}

/**
 * Main initialization sequence. Sets up I/O, parses inputs, 
 * and triggers the domain generation via start().
 */
void initialise() {
    using namespace TeaLeaf;

    // Set up global output stream
    if (parallel.boss) {
        g_file_stream.open("tea.out");
        if (!g_file_stream.is_open()) {
            std::cerr << "initialise: Error opening tea.out file.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        g_out = &g_file_stream; // Master rank writes to file
    } else {
        g_out = &std::cout;     // Slave ranks write to standard output
    }

    // Print banner and versioning info
    #pragma omp parallel
    {
        #pragma omp master
        {
            if (parallel.boss) {
                *g_out << "\nTea Version       " << g_version << "\n";
                *g_out << "Task Count        " << parallel.max_task << "\n";                
                *g_out << "Thread Count:     " << omp_get_num_threads() << "\n";
                *g_out << "\nOutput file tea.out opened.\n";
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Input file handling: Generate default if missing
    if (parallel.boss) {
        if (!file_exists("../src/tea.in")) {
            std::cout << "\nNo input file found. Generating default tea.in\n";
            std::ofstream out_unit("../src/tea.in");
            if (!out_unit.is_open()) {
                std::cerr << "initialise: Error creating default tea.in\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // Write a basic physical problem (states, grid size, solver params)
            out_unit << "*tea\n"
                     << "state 1 density=100.0 energy=0.0001\n"
                     << "state 2 density=0.1 energy=25.0 geometry=rectangle xmin=0.0 xmax=1.0 ymin=1.0 ymax=2.0\n"
                     << "x_cells=1000\ny_cells=1000\n"
                     << "xmin=0.0\nymin=0.0\nxmax=10.0\nymax=10.0\n"
                     << "initial_timestep=0.004\nend_step=10\n"
                     << "tl_max_iters=1000\ntest_problem 1\ntl_use_jacobi\ntl_eps=1.0e-15\n"
                     << "*endtea\n";
            out_unit.close();
        } else {
            std::cout << "\nFile found, using the specified tea.in\n";
        }

        // Create the clean version for the parser
        clean_input_file("../src/tea.in", "../src/tea.in.tmp");
    }

    // Synchronize so all ranks wait for the boss to finish I/O
    MPI_Barrier(MPI_COMM_WORLD);

    // Echo the input file content to the log for traceability
    if (parallel.boss) {
        std::ifstream uin("../src/tea.in");
        if (uin.is_open()) {
            std::string ltmp;
            while (std::getline(uin, ltmp)) {
                *g_out << ltmp << "\n";
            }
        }
        *g_out << "\nInitialising and generating\n\n";
    }

    // Read parameters from the cleaned temp file
    read_input();

    MPI_Barrier(MPI_COMM_WORLD);

    step = 0;

    // Start memory allocation and initial field generation (geometry, states)
    start();

    MPI_Barrier(MPI_COMM_WORLD);

    if (parallel.boss) {
        *g_out << "Starting the calculation\n";
    }
}