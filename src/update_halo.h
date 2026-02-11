#pragma once
#include <array>

void update_halo(const int* fields, int depth);
void update_boundary(const int* fields, int depth);
void update_tile_boundary(const int*  fields, int depth);

