# pragma once

void initialise_chunk_kernel(
    int x_min, int x_max, 
    int y_min, int y_max,
    double xmin, double ymin, 
    double dx, double dy,
    double* vertexx, 
    double* vertexdx,
    double* vertexy, 
    double* vertexdy,
    double* cellx, 
    double* celldx,
    double* celly, 
    double* celldy,
    double* volume,
    double* xarea, 
    double* yarea
);
