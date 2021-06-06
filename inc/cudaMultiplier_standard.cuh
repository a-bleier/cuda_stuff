//
// Created by adrian on 31.03.21.
//

#ifndef MATRIXMUL_CUDAMULTIPLIER_CUH
#define MATRIXMUL_CUDAMULTIPLIER_CUH

#define BLOCK_SIZE 32
#include "main.cuh"
void cuda_multiply_standard(int** m1, int**m2, int**res, uint m, uint n, uint k);
#endif //MATRIXMUL_CUDAMULTIPLIER_CUH
