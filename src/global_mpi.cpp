#include "global_mpi.h"
#include "data.h" 
#include "definitions.h"
#include <mpi.h>
#include <algorithm> // For std::max if needed

using namespace TeaLeaf;

// reduces value to rank 0
void tea_sum(double& value) {
    double total = 0.0;
    
    //MPI_Reduce(sendbuf, recvbuf, ..)
    MPI_Reduce(&value, &total, 1, MPI_DOUBLE, MPI_SUM, 0, mpi_cart_comm);

    // Update the input value (only meaningfull on rank 0)
}

// global sum reduction 
void tea_allsum(double& value) {
    double total = 0.0;
    double start_time = 0.0;

    if (profiler_on) start_time = MPI_Wtime();

    MPI_Allreduce(&value, &total, 1, MPI_DOUBLE, MPI_SUM, mpi_cart_comm);

    if (profiler_on) {
        profiler.dot_product += (MPI_Wtime() - start_time);
    }

    value = total;
}

// ----------------------------------------------------------------------------
// tea_allsum2: Global Sum Reduction for 2 values at once (Optimization)
// ----------------------------------------------------------------------------
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

void tea_min(double& value) {
    double minimum = 0.0;
    MPI_Allreduce(&value, &minimum, 1, MPI_DOUBLE, MPI_MIN, mpi_cart_comm);
    value = minimum;
}

void tea_max(double& value) {
    double maximum = 0.0;
    MPI_Allreduce(&value, &maximum, 1, MPI_DOUBLE, MPI_MAX, mpi_cart_comm);
    value = maximum;
}


void tea_allgather(double value, std::vector<double>& values) {

    //ensure the vector is big enough to gather from every rank
    if (values.size() < (size_t)parallel.max_task) {
        values.resize(parallel.max_task);
    }

    values[0] = value; 
    MPI_Allgather(&value, 1, MPI_DOUBLE, 
                  values.data(), 1, MPI_DOUBLE, 
                  mpi_cart_comm);
}

void tea_check_error(int& error) {
    int maximum = 0;
    MPI_Allreduce(&error, &maximum, 1, MPI_INT, MPI_MAX, mpi_cart_comm);
    error = maximum;
}

void tea_barrier() {
    MPI_Barrier(mpi_cart_comm);
}

void tea_abort() {
    MPI_Abort(mpi_cart_comm, 1);
}