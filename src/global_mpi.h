#pragma once

#include <vector>

void tea_sum(double& value);
void tea_allsum(double& value);
void tea_allsum2(double& value1, double& value2);
void tea_min(double& value);
void tea_max(double& value);
void tea_allgather(double value, std::vector<double>& values);
void tea_check_error(int& error);
void tea_barrier();
void tea_abort();

