//
// Created by adrian on 31.03.21.
//

#ifndef MATRIXMUL_CUDAMULTIPLIER_CUH
#define MATRIXMUL_CUDAMULTIPLIER_CUH

void cuda_multiply_own(int** m1, int**m2, int**res, uint m, uint n, uint k);
#endif //MATRIXMUL_CUDAMULTIPLIER_CUH
