#include "include/global_mpi.h"
#include "include/data.h" 
#include "include/definitions.h"
#include <mpi.h>
#include <algorithm> // For std::max if needed

using namespace TeaLeaf;

/**
 * Reduces a local value to a total sum on rank 0 only.
 * Uses MPI_Reduce to minimize communication overhead when the result 
 * is only needed by the master process.
 */
void tea_sum(double& value) {
    double total = 0.0;
    
    // Perform the reduction sum targeting the root process (rank 0)
    MPI_Reduce(&value, &total, 1, MPI_DOUBLE, MPI_SUM, 0, mpi_cart_comm);

    // Note: 'total' is only valid on rank 0 after this call
}

/**
 * Global sum reduction where the result is synchronized across all ranks.
 * Essential for global convergence checks and energy conservation.
 */
void tea_allsum(double& value) {
    double total = 0.0;
    double start_time = 0.0;

    if (profiler_on) start_time = MPI_Wtime();

    // Sum across all processes and broadcast the result back to everyone
    MPI_Allreduce(&value, &total, 1, MPI_DOUBLE, MPI_SUM, mpi_cart_comm);

    if (profiler_on) {
        profiler.dot_product += (MPI_Wtime() - start_time);
    }

    value = total;
}

/**
 * Optimized global sum reduction for two values simultaneously.
 * Bundling values into a single MPI call reduces network latency overhead.
 */
void tea_allsum2(double& value1, double& value2) {
    double values[2] = {value1, value2};
    double totals[2] = {0.0, 0.0};
    double start_time = 0.0;

    if (profiler_on) start_time = MPI_Wtime();

    // Reduce 2 doubles in one packet to save latency
    MPI_Allreduce(values, totals, 2, MPI_DOUBLE, MPI_SUM, mpi_cart_comm);

    if (profiler_on) {
        profiler.dot_product += (MPI_Wtime() - start_time);
    }

    value1 = totals[0];
    value2 = totals[1];
}

/**
 * Synchronizes the minimum value across the entire computational domain.
 */
void tea_min(double& value) {
    double minimum = 0.0;
    MPI_Allreduce(&value, &minimum, 1, MPI_DOUBLE, MPI_MIN, mpi_cart_comm);
    value = minimum;
}

/**
 * Synchronizes the maximum value across the entire computational domain.
 */
void tea_max(double& value) {
    double maximum = 0.0;
    MPI_Allreduce(&value, &maximum, 1, MPI_DOUBLE, MPI_MAX, mpi_cart_comm);
    value = maximum;
}

/**
 * Gathers a single value from every process into a global vector.
 * Used for collecting status or performance data from all MPI ranks.
 */
void tea_allgather(double value, std::vector<double>& values) {

    // Ensure the vector is big enough to gather from every rank
    if (values.size() < (size_t)parallel.max_task) {
        values.resize(parallel.max_task);
    }

    values[0] = value; 
    MPI_Allgather(&value, 1, MPI_DOUBLE, 
                  values.data(), 1, MPI_DOUBLE, 
                  mpi_cart_comm);
}

/**
 * Checks for errors across all ranks by finding the global maximum error code.
 * If any rank has a non-zero error, all ranks will receive that error code.
 */
void tea_check_error(int& error) {
    int maximum = 0;
    MPI_Allreduce(&error, &maximum, 1, MPI_INT, MPI_MAX, mpi_cart_comm);
    error = maximum;
}

/**
 * Blocks execution until all processes in the communicator have reached this point.
 */
void tea_barrier() {
    MPI_Barrier(mpi_cart_comm);
}

/**
 * Forcefully terminates the MPI environment in the event of a critical error.
 */
void tea_abort() {
    MPI_Abort(mpi_cart_comm, 1);
}