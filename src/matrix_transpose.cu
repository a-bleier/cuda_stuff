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