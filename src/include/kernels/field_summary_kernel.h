#pragma once

void field_summary_kernel(
    int x_min, int x_max, 
    int y_min, int y_max, 
    int halo_exchange_depth,
    double* volume, 
    double* density, 
    double* energy1, 
    double* u, 
    double& vol, 
    double& mass, 
    double& ie, 
    double& temp
);

