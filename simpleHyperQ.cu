#include <cooperative_groups.h>
#include <iostream>
#include <stdio.h>

namespace cg = cooperative_groups;
#include <helper_cuda.h>
#include <helper_functions.h>

const char *sSDKsample = "hyperQ";

__device__ void clock_block(clock_t *d_o, clock_t clock_count){
    unsigned int start_clock = (unsigned int)(clock)();

    clock_t clock_offset = 0;

    while(clock_offset < clock_count){
        unsigned int end_clock = (unsigned int)clock();

        // clock_offset - (clock_t)(end_clock > start_clock ?
        //                          end_clock - start_clock :
        //                          end_clock + (0xffffffffu - start_clock));
        clock_offset = (clock_t)(end_clock - start_clock);
    }
    d_o[0] = clock_offset;
}

__global__ void kernel_A(clock_t *d_o, clock_t clock_count){
    clock_block(d_o, clock_count);
}

__global__ void kernel_B(clock_t *d_o, clock_t clock_count){
    clock_block(d_o, clock_count);
}

__global__ void sum(clock_t *d_clocks, int N){
    cg::thread_block cta = cg::this_thread_block();
    __shared__ clock_t s_clocks[32];

    clock_t my_sum = 0;

    for (int i = threadIdx.x; i < N; i += blockDim.x){
        my_sum += d_clocks[i];
    }

    s_clocks[threadIdx.x] = my_sum;
    cg::sync(cta);

    for (int i = warpSize / 2; i > 0; i /= 2){
        if(threadIdx.x < i){
            s_clocks[threadIdx.x] += s_clocks[threadIdx.x + i];
        }

        cg::sync(cta);
    }

    if (threadIdx.x == 0){
        d_clocks[0] = s_clocks[0];
    }
}


int main(int argc, char **argv){
    int nstreams = 32;
    float kernel_time = 10;
    float elapsed_time;
    int cuda_device = 0;

    printf("starting %s...\n", sSDKsample);

    if(checkCmdLineFlag(argc, (const char **)argv, "nstreams")){
        nstreams = getCmdLineArgumentInt(argc, (const char **)argv, "nstreams");
    }

    cuda_device = findCudaDevice(argc, (const char **)argv);

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDevice(&cuda_device));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));

    if(deviceProp.major < 3 || (deviceProp.major == 3 && deviceProp.minor < 5)){
        if(deviceProp.concurrentKernels == 0){
            printf(
                "> GPU does not support concurrent kernel execution (SM 3.5 or "
                "higher required)\n");
            printf("  CUDA kernel runs will be serialized\n");
        }
    }
    printf(">Detected Computed SM %d.%d hardware with %d multi-processor\n",
           deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

    clock_t *a = 0;
    checkCudaErrors(cudaMallocHost((void **)&a, sizeof(clock_t)));

    clock_t *d_a = 0;
    checkCudaErrors(cudaMalloc((void **)&d_a, 2 * nstreams * sizeof(clock_t)));

    cudaStream_t * streams = (cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));

    for (int i = 0; i < nstreams; i++){
        checkCudaErrors(cudaStreamCreate(&(streams[i])));
    }

    cudaEvent_t start_event, stop_event;
    checkCudaErrors(cudaEventCreate(&start_event));
    checkCudaErrors(cudaEventCreate(&stop_event));

#if defined(__arm__) || defined(__aarch64__)
    clock_t time_clocks = (clock_t)(kernel_time * (deviceProp.clockRate / 100));
#else
    clock_t time_clocks = (clock_t)(kernel_time * deviceProp.clockRate);
#endif
    clock_t total_clocks = 0;

    checkCudaErrors(cudaEventRecord(start_event, 0));

    for(int i = 0; i < nstreams; ++i){
        kernel_A<<<1, 1, 0, streams[i]>>>(&d_a[2 * i], time_clocks);
        total_clocks += time_clocks;
        kernel_B<<<1, 1, 0, streams[i]>>>(&d_a[2 * i + 1], time_clocks);
        total_clocks += time_clocks;
    }

    checkCudaErrors(cudaEventRecord(stop_event, 0));

    sum<<<1, 32>>>(d_a, 2 * nstreams);
    checkCudaErrors(cudaMemcpy(a, d_a, sizeof(clock_t), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaEventSynchronize(stop_event));
    checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));

    printf(
        "Expected time for serial execution of %d sets of kernels is between "
        "approx. %.3fs and %.3fs\n",
        nstreams, (nstreams + 1) * kernel_time / 1000.0f,
        2 * nstreams * kernel_time / 1000.0f);
    printf(
        "Expected time for fully concurrent execution of %d sets of kernels is "
        "aprox. %.3fs\n",
        nstreams, 2 * kernel_time / 1000.0f);
    printf("Measured time for sample = %3.fs\n", elapsed_time / 1000.0f);

    bool bTestResult = (a[0] >= total_clocks);

    for (int i=0; i<nstreams; i++){
        cudaStreamDestroy(streams[i]);
    }
    free(streams);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaFreeHost(a);
    cudaFree(d_a);

    exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

