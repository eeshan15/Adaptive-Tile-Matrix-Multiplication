// dynamic_tile_matmul.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

using namespace std;

template <int TILE_WIDTH>
__global__ void matMulKernelDynamic(const float *A, const float *B, float *C, int wA, int wB, int wC) {
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float value = 0.0f;

    for (int t = 0; t < (wC + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        if (row < wA && t * TILE_WIDTH + threadIdx.x < wC)
            s_A[threadIdx.y][threadIdx.x] = A[row * wC + t * TILE_WIDTH + threadIdx.x];
        else
            s_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (t * TILE_WIDTH + threadIdx.y < wB && col < wB)
            s_B[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * wB + col];
        else
            s_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i)
            value += s_A[threadIdx.y][i] * s_B[i][threadIdx.x];

        __syncthreads();
    }

    if (row < wA && col < wB)
        C[row * wB + col] = value;
}

//  Public entry point callable from main.cu
extern "C" void launchDynamicTiledKernel(float *h_A, float *h_B, float *h_C, int wA, int wB, int wC, int tileSize) {
    float *d_A, *d_B, *d_C;

    size_t size_A = wA * wC * sizeof(float);
    size_t size_B = wC * wB * sizeof(float);
    size_t size_C = wA * wB * sizeof(float);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 dimBlock, dimGrid;

    if (tileSize == 8) {
        dimBlock = dim3(8, 8);
        dimGrid = dim3((wB + 7) / 8, (wA + 7) / 8);
        matMulKernelDynamic<8><<<dimGrid, dimBlock>>>(d_A, d_B, d_C, wA, wB, wC);
    }
    else if (tileSize == 16) {
        dimBlock = dim3(16, 16);
        dimGrid = dim3((wB + 15) / 16, (wA + 15) / 16);
        matMulKernelDynamic<16><<<dimGrid, dimBlock>>>(d_A, d_B, d_C, wA, wB, wC);
    }
    else if (tileSize == 32) {
        dimBlock = dim3(32, 32);
        dimGrid = dim3((wB + 31) / 32, (wA + 31) / 32);
        matMulKernelDynamic<32><<<dimGrid, dimBlock>>>(d_A, d_B, d_C, wA, wB, wC);
    } else {
        std::cerr << "Unsupported tile size: " << tileSize << std::endl;
        return;
    }

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
