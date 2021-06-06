//
// Created by adrian on 31.03.21.
//
#include "cudaDmy.cuh"
#include "cudaMultiplier_own.cuh"
#include "common.cuh"

__global__ void launch_multiplication_own(int *m1, int *m2, int *res, int *buff){
    /* dimensions
     * M -> gridDim.y
     * N -> blockDim.x
     * K -> gridDim.x
     *
     * Coords:
     * m -> blockIdx.y
     * n -> threadIdx.x
     * k -> blockIdx.x
     */

    int buff_index = gridDim.x * blockDim.x * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    buff[buff_index] = m1[blockIdx.y * blockDim.x + threadIdx.x] * m2[threadIdx.x * gridDim.x + blockIdx.x];

    int idx = gridDim.x * blockDim.x * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;

    int old_stride = blockDim.x;
    int limit = old_stride/2;
    int stride=old_stride-limit;
    do{
        __syncthreads();
        if(threadIdx.x < limit){
            buff[idx] += buff[idx + stride];
        }
        __syncthreads();
        old_stride -= limit;
        limit = old_stride/2;
        stride=old_stride-limit;
    }while(limit > 0);// The number of loop iterations are the same among all threads

    __syncthreads();

    if(threadIdx.x == 0){

        res[blockIdx.y * gridDim.x + blockIdx.x]  = buff[gridDim.x * blockDim.x * blockIdx.y + blockDim.x * blockIdx.x + 0];
    }
}



void cuda_multiply_own(int **m1, int **m2, int **res, uint m, uint n, uint k) {

    //Allocate device memory
//    printf("Setting up device mem\n");
    int *d_m1, *d_m2, *d_res, *d_buff;
    CHECK(cudaMalloc((int**)&d_m1, m*n*sizeof(int)));
    CHECK(cudaMalloc((int**)&d_m2, n*k*sizeof(int)));
    CHECK(cudaMalloc((int**)&d_res, m*k*sizeof(int)));
    CHECK(cudaMalloc((int**)&d_buff, m*n*k*sizeof(int))); //This one is for storing products; needs maybe some rework later on

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
    cudaMemset(d_buff, 0, m*k*n);


    //Setup dimensions: grid like the resulting matrix, block resembles the number of multiplications
    dim3 grid(k,m);
    dim3 block(n);

    //Launch the kernel
//    printf("Launching the kernel\n");
    double start, elapse;

    CHECK(cudaDeviceSynchronize());
    start = seconds();
    launch_multiplication_own<<<grid, block>>>(d_m1, d_m2, d_res, d_buff);
    CHECK(cudaDeviceSynchronize());
    elapse = seconds() - start;
    printf("CUDA matrix multplication WITHOUT OVERHEAD: %f\n", elapse);
    CHECK(cudaGetLastError());

    //Copy result back to host memory
//    printf("Copy back the result\n");
    for(int i=0; i<m; i++){
            CHECK(cudaMemcpy(res[i], &d_res[k*i], k*sizeof(int), cudaMemcpyDeviceToHost));
    }

    //Free device memory
//    printf("Free device memory\n");
    CHECK(cudaFree(d_m1));
    CHECK(cudaFree(d_m2));
    CHECK(cudaFree(d_res));
    CHECK(cudaFree(d_buff));
}

