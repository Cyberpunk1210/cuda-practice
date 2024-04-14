#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define ELEMENTS 1024


__global__ void vectorAdd(const float *d_A, const float *d_B, float *d_C, int Elements)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < Elements){
        d_C[tid] = d_A[tid] + d_B[tid];
    }
}


int main()
{
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    size_t size = sizeof(float) * ELEMENTS;

    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);
    
    for(int i=0; i<ELEMENTS; ++i){
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlocks = 16;
    int blocksPerGrid = (ELEMENTS + threadsPerBlocks - 1) / threadsPerBlocks;
    vectorAdd<<<blocksPerGrid, threadsPerBlocks>>>(d_A, d_B, d_C, ELEMENTS);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    for (int j=0; j<ELEMENTS; ++j)
    {
        if (h_C[j] != h_A[j] + h_B[j])
            printf("computation error!\n");
    }

    printf("Done!\n");
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
