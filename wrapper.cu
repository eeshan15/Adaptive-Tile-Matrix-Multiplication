//wrapper.cu
#include <cuda_runtime.h>
#include "static_tile_matmul.cu"
#include "dynamic_tile_matmul.cu"


extern void runStaticMatMul(const float* A, const float* B, float* C, int wA, int wB, int tileSize);
extern void launchDynamicTiledKernel(float* A, float* B, float* C, int wA, int wB, int wC, int TILE_SIZE);

extern "C" {
    __declspec(dllexport) void run_matmul(float* A, float* B, float* C, int N, int kernel_type, int tile_size, float* time_ms) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    if (kernel_type == 0) {
        runStaticMatMul(A, B, C, N, N, tile_size);
    } else {
        launchDynamicTiledKernel(A, B, C, N, N, N, tile_size);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(time_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
}
