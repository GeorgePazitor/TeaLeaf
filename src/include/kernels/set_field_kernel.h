#pragma once

void set_field_kernel(
    int x_min, int x_max, 
    int y_min, int y_max, 
    int halo_exchange_depth,
    double* energy0, 
    double* energy1
);
