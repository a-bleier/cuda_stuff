#pragma once
#define TILE_DIM_X 32
#define TILE_DIM_Y 32
__global__ void transpose_matrix_naive(int *in, int *out, int m, int n);
template <typename T> __global__ void transpose_matrix_shared(T *in, T*out, int m, int n);
