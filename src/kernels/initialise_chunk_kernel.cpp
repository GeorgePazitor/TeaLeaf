void initialise_chunk_kernel(
    int x_min, int x_max, 
    int y_min, int y_max,
    double xmin, double ymin, 
    double dx, double dy,
    double* vertexx, double* vertexdx,
    double* vertexy, double* vertexdy,
    double* cellx, double* celldx,
    double* celly, double* celldy,
    double* volume, double* xarea, double* yarea) 
{
    // Calcul des largeurs exactes basées sur les déclarations Fortran
    int vol_width   = (x_max + 2) - (x_min - 2) + 1; 
    int xarea_width = (x_max + 3) - (x_min - 2) + 1; 
    int yarea_width = (x_max + 2) - (x_min - 2) + 1; 

    #pragma omp parallel
    {
        // 1D arrays (Vertex/Cell) - Pas de changement ici
        #pragma omp for nowait
        for (int j = x_min - 2; j <= x_max + 3; ++j) {
            int idx = j - (x_min - 2);
            vertexx[idx]  = xmin + dx * (double)(j - x_min);
            vertexdx[idx] = dx;
        }

        #pragma omp for
        for (int k = y_min - 2; k <= y_max + 3; ++k) {
            int idx = k - (y_min - 2);
            vertexy[idx]  = ymin + dy * (double)(k - y_min);
            vertexdy[idx] = dy;
        }

        #pragma omp for nowait
        for (int j = x_min - 2; j <= x_max + 2; ++j) {
            int idx = j - (x_min - 2);
            cellx[idx]  = 0.5 * (vertexx[idx] + vertexx[idx + 1]);
            celldx[idx] = dx;
        }

        #pragma omp for
        for (int k = y_min - 2; k <= y_max + 2; ++k) {
            int idx = k - (y_min - 2);
            celly[idx]  = 0.5 * (vertexy[idx] + vertexy[idx + 1]);
            celldy[idx] = dy;
        }

        // 2D Fields - Utilisation des largeurs spécifiques pour éviter le SegFault
        #pragma omp for
        for (int k = y_min - 2; k <= y_max + 2; ++k) {
            int k_idx = k - (y_min - 2);
            for (int j = x_min - 2; j <= x_max + 2; ++j) {
                int j_idx = j - (x_min - 2);
                
                volume[k_idx * vol_width + j_idx] = dx * dy;
                yarea[k_idx * yarea_width + j_idx] = dx;
            }
        }

        // Xarea a une plage J plus large (x_max + 3)
        #pragma omp for
        for (int k = y_min - 2; k <= y_max + 2; ++k) {
            int k_idx = k - (y_min - 2);
            for (int j = x_min - 2; j <= x_max + 3; ++j) {
                int j_idx = j - (x_min - 2);
                xarea[k_idx * xarea_width + j_idx] = dy;
            }
        }
    }
}