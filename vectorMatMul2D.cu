#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <chrono>

// using namespace std;
#define BLOCK_SIZE 16
#define ELEMENTS 4096

typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

__device__ void SetElement(Matrix A, int row, int col, float value)
{
    A.elements[row * A.stride + col] = value;
}

__device__ Matrix GetSubMatrix(Matrix A, int row, int col){
    Matrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.width = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}

__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // allocate A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    
    auto start = std::chrono::system_clock::now();
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed Time: " << elapsed_seconds.count() << "sec" << std::endl;

    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // non-shared memory
    // int row = blockDim.y * blockIdx.y + threadIdx.y;
    // int col = blockDim.x * blockIdx.x + threadIdx.x;
    // for (int e=0; e< A.width; ++e)
    //     Cvalue += A.elements[row * A.width + e] 
    //             * B.elements[e *B.width + col];
    // C.elements[row * C.width + col] = Cvalue;

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    float Cvalue = 0.0;

    int row = threadIdx.y;
    int col = threadIdx.x;

    for(int m=0; m< (A.width / BLOCK_SIZE); ++m){
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        __syncthreads();
        for (int e=0; e<BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];
        __syncthreads();
    }

    SetElement(Csub, row, col, Cvalue);
}


int main(){
    Matrix h_A, h_B, h_C;
    size_t size = sizeof(float) * ELEMENTS;
    h_A.width  = h_B.height = 128;
    h_A.stride = h_B.stride = h_C.stride = 4;
    h_A.height = h_B.width = 32;
    h_C.width  = h_C.height = 32;
    float *mA, *mB;
    mA = (float *)malloc(size);
    mB = (float *)malloc(size);
    float *mC = NULL;
    for (int i=0; i<ELEMENTS; ++i){
            mA[i] = rand() / (float)RAND_MAX;
            mB[i] = rand() / (float)RAND_MAX;
    }
    h_A.elements = mA;
    h_B.elements = mB;
    h_C.elements = mC;

    MatMul(h_A, h_B, h_C);

    printf("Done\n");

    free(h_A.elements);
    free(h_B.elements);
    free(h_C.elements);
    return 0;
}