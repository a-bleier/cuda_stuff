#include <iostream>
#include <stdio.h>
#include "main.cuh"
#include "common.cuh"

int main() {

    const int M = 2245;
    const int N = 3333;
    const int K = 2121;

    printf("Multiplying a %dx%d with %dx%d\n", M, N, N, K);
    int** m1 = generateRandomMatrix(M, N);
//    printMatrix(m1, M, N);
    int** m2 = generateRandomMatrix(N, K);
//    printMatrix(m2, N, K);


    double start, elapse_cu, elapse_h;
    start = seconds();
    int** res = matrixMultiplicationOnHost(m1, m2, M, N, K);
    elapse_h = seconds() - start;
//    printMatrix(res, M, K);

    int **res_cuda = generateRandomMatrix(M,K);
    start = seconds();
    cuda_multiply_standard(m1, m2, res_cuda, (uint) M,(uint) N,(uint) K);
    elapse_cu = seconds() - start;

    //Ceck for equality
    if(checkMatrixEqual(res, res_cuda, M, K)){
        printf("SUCCESS: Host result and CUDA result are equal\n");
    } else {

        printf("ERROR: Host result and CUDA result differ\n");
        printMatrix(res, M, K);
        printMatrix(res_cuda, M, K);
    }

    printf("CPU matrix multplication: %f\n", elapse_h);

    printf("CUDA matrix multplication WITH OVERHEAD: %f\n", elapse_cu);



    destroyMatrix(m1, M, N);
    destroyMatrix(m2, N, K);
    destroyMatrix(res, M,K);
    destroyMatrix(res_cuda, M,K);
    return 0;
}

int **generateRandomMatrix(uint m, uint n) {
    srand(time(NULL));
    int** mat = new int*[m];
    for(uint i = 0; i < m; i++){
        mat[i] = new int[n];
        for(uint k = 0; k < n; k++){
            mat[i][k] = rand() % 1000;
            // mat[i][k] = 1;
        }
    }
    return mat;
}

void printMatrix(int **mat, uint m, uint n) {
    printf("\n");
    for(uint i = 0; i < m; i++){
        for(uint k = 0; k < n; k++){
            printf("%d ",mat[i][k] );
        }
        printf("\n");
    }
}
//Matrix Multiplication of a mxn and nxk Matrix
int** matrixMultiplicationOnHost(int **m1, int **m2, int m, int n, int k) {
    int** mat = new int*[m];
    for(int i=0; i < m; i++){
        mat[i] = new int[k];
        for(int l=0; l < k; l++){
            int sum = 0;
            for(int r = 0; r < n; r++){
               sum += m1[i][r] * m2[r][l];
            }
            mat[i][l] = sum;
        }
    }
    return mat;
}

void destroyMatrix(int** mat, int m, int n) {
    for(int i=0; i<m; i++){
        delete[] mat[i];
    }
    delete[] mat;
}

bool checkMatrixEqual(int**mat1, int**mat2, int M, int N){
    for(int m=0; m<M; m++){
        for(int n=0; n<N; n++){
            if(mat1[m][n] != mat2[m][n]){
                return false;
            }
        }
    }

    return true;
}