#include "matrix_transpose.cuh"


__global__ void transpose_matrix_naive(int *in, int *out, int m, int n){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;


    if(row < m && col < n){
        int pos = n * row + col;
        int trans_pos =  m* col + row;
        out[trans_pos] = in[pos];
    }
}

template <typename T>
__global__ void transpose_matrix_shared(T *in, T *out, int m, int n){
    __shared__ T tile[TILE_DIM_Y][TILE_DIM_X];

    int row = blockIdx.y * TILE_DIM_Y + threadIdx.y;
    int col = blockIdx.x * TILE_DIM_X + threadIdx.x;

    if(row < m && col < n){
        int pos = n*row + col;
        int pos_trans = m * col + row;
        tile[threadIdx.y][threadIdx.x] = in[pos];


        __syncthreads();


        out[pos_trans] = tile[threadIdx.y][threadIdx.x];
    }

}



template __global__ void transpose_matrix_shared<int>(int *in, int *out, int m, int n);