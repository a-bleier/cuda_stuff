//
// Created by adrian on 30.03.21.
//

#ifndef MATRIXMUL_MAIN_CUH
#define MATRIXMUL_MAIN_CUH

#include <cstdlib>
#include <ctime>
#include "cudaMultiplier_standard.cuh"

//generates a mxn Matrix with random integer values
int** generateRandomMatrix(uint m, uint n);
void destroyMatrix(int** mat, int m, int n);
//prints a mxn Matrix
void printMatrix(int**mat, uint m, uint n);
//Matrix multiplication
int** matrixMultiplicationOnHost(int **m1, int **m2, int m, int n, int k);
//check two matrices for equality
bool checkMatrixEqual(int**mat1, int**mat2, int M, int N);
#endif //MATRIXMUL_MAIN_CUH
