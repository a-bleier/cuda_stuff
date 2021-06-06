//
// Created by adrian on 31.03.21.
//
#include "cudaDmy.cuh"
#include "cudaMultiplier_standard.cuh"
#include "common.cuh"
#include <iostream>

__global__ void launch_multiplication_standard(int *m1, int *m2_trans, int *res, int m, int n, int k){


    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < m && col < k){
        for(int i=0; i < n; i++){

            res[row * k + col] += m1[row * n + i] * m2_trans[n * col + i];
        }
    }
}

__global__ void transpose_matrix(int *in, int *out, int m, int n){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;


    if(row < m && col < n){
        int pos = n * row + col;
        int trans_pos =  m* col + row;
        out[trans_pos] = in[pos];
    }
}



void cuda_multiply_standard(int **m1, int **m2, int **res, uint m, uint n, uint k) {

    //Allocate device memory
//    printf("Setting up device mem\n");
    int *d_m1, *d_m2, *d_m2_trans, *d_res;;
    CHECK(cudaMalloc((int**)&d_m1, m*n*sizeof(int)));
    CHECK(cudaMalloc((int**)&d_m2, n*k*sizeof(int)));
    CHECK(cudaMalloc((int**)&d_m2_trans, n*k*sizeof(int)));
    CHECK(cudaMalloc((int**)&d_res, m*k*sizeof(int)));

    //Copy to device
//    printf("Copy data to device\n");
    for(int i=0; i<m; i++){
        CHECK(cudaMemcpy(&d_m1[i * n], m1[i], n*sizeof(int), cudaMemcpyHostToDevice));
    }
    for(int i=0; i<n; i++){
        CHECK(cudaMemcpy(&d_m2[i * k], m2[i], k*sizeof(int), cudaMemcpyHostToDevice));
    }
    //Null the result matrix
    cudaMemset(d_res, 0, m*k);


    //Transpose matrix
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_trans(ceilf((float) k/(float)BLOCK_SIZE),ceilf((float) n/(float)BLOCK_SIZE));
    CHECK(cudaDeviceSynchronize());
    transpose_matrix<<<grid_trans, block>>>(d_m2, d_m2_trans, n,k);
    //Grid for multiplication
    dim3 grid(ceilf((float) k/(float)BLOCK_SIZE),ceilf((float) m/(float)BLOCK_SIZE));

    double start, elapse;

    CHECK(cudaDeviceSynchronize());
    start = seconds();

    launch_multiplication_standard<<<grid, block>>>(d_m1, d_m2_trans, d_res, m, n, k);

    CHECK(cudaDeviceSynchronize());
    elapse = seconds() - start;
    printf("CUDA matrix multplication WITHOUT OVERHEAD: %f\n", elapse);
    CHECK(cudaGetLastError());

    //Copy result back to host memory
    for(int i=0; i<m; i++){
            CHECK(cudaMemcpy(res[i], &d_res[k*i], k*sizeof(int), cudaMemcpyDeviceToHost));
    }

    //Free device memory
    CHECK(cudaFree(d_m1));
    CHECK(cudaFree(d_m2));
    CHECK(cudaFree(d_res));
    CHECK(cudaFree(d_m2_trans));
}
//n x k matrix on device
void check_trans(int **trans_cuda, int n, int k){

    //Check transpose matrix
    int **trans = new int*[n];
    for(int i = 0; i< k; i++){
        trans[i] = new int[n];
         CHECK(cudaMemcpy(trans[i], &trans_cuda[n*i], n*sizeof(int), cudaMemcpyDeviceToHost))
    }
    std::cout << "Print transposed matrix " << n << " " << k << std::endl;
    printMatrix(trans, k, n);

    destroyMatrix(trans, n, k);
}