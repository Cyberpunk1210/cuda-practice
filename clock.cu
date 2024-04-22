#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

__global__ static void timeReduction(const float *input, float *output, clock_t *timer){
    extern __shared__ float shared[];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    if (tid==0) timer[bid] = clock();

    shared[tid] = input[tid];
    shared[tid + blockDim.x] = input[tid + blockDim.x];

    for (int d = blockDim.x; d > 0; d /= 2){
        __syncthreads();

        if (tid < d){
            float f0 = shared[tid];
            float f1 = shared[tid + d];
        
            if (f1 < f0){
                shared[tid] = f1;
            }
        }
    }

    if (tid == 0) output[bid] = shared[0];

    __syncthreads();

    if (tid == 0) timer[bid + gridDim.x] = clock();
}

#define NUM_BLOCKS 64
#define NUM_THREADS 256

int main(int argc, char **argv){
    printf("CUDA Clock sample\n");

    int dev = findCudaDevice(argc, (const char **)argv);

    float *dinput = NULL;
    float *doutput = NULL;
    clock_t *dtimer = NULL;

    clock_t timer[NUM_BLOCKS * 2];
    float input[NUM_THREADS * 2];

    for (int i=0; i < NUM_THREADS * 2; i++){
        input[i] = (float)i;
    }

    checkCudaErrors(
        cudaMalloc((void **)&dinput, sizeof(float) * NUM_THREADS * 2));
    checkCudaErrors(cudaMalloc((void **)&doutput, sizeof(float) * NUM_BLOCKS));
    checkCudaErrors(
        cudaMalloc((void **)&dtimer, sizeof(clock_t) * NUM_BLOCKS * 2));
    checkCudaErrors(cudaMemcpy(dinput, input, sizeof(float) * NUM_BLOCKS * 2,
                               cudaMemcpyHostToDevice));

    timeReduction<<<NUM_BLOCKS, NUM_THREADS, sizeof(float) * 2 * NUM_THREADS>>>(
        dinput, doutput, dtimer);
    
    checkCudaErrors(cudaMemcpy(timer, dtimer, sizeof(clock_t) * NUM_BLOCKS * 2,
                               cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(dinput));
    checkCudaErrors(cudaFree(doutput));
    checkCudaErrors(cudaFree(dtimer));

    long double avgElapsedClocks = 0;

    for (int i=0; i < NUM_BLOCKS; i++){
        avgElapsedClocks += (long double)(timer[i + NUM_BLOCKS] - timer[i]);
    }

    avgElapsedClocks = avgElapsedClocks / NUM_BLOCKS;
    printf("Average clocks/block = %Lf\n", avgElapsedClocks);

    return EXIT_SUCCESS;
}

