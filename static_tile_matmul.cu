//static_tile_matmul.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

template <int TILE_SIZE>
__global__ void staticTileKernel(const float* A, const float* B, float* C, int wA, int wB) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0.0f;

    for (int t = 0; t < (wA + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < wA && (t * TILE_SIZE + threadIdx.x) < wA)
            tile_A[threadIdx.y][threadIdx.x] = A[row * wA + t * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        if ((t * TILE_SIZE + threadIdx.y) < wB && col < wB)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * wB + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i)
            value += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];

        __syncthreads();
    }

    if (row < wA && col < wB)
        C[row * wB + col] = value;
}

extern "C" {
void runStaticMatMul(const float* A, const float* B, float* C, int wA, int wB, int tileSize) {
    dim3 grid, block;

    if (tileSize == 8) {
        block = dim3(8, 8);
        grid = dim3((wB + 7) / 8, (wA + 7) / 8);
        staticTileKernel<8><<<grid, block>>>(A, B, C, wA, wB);
    }
    else if (tileSize == 16) {
        block = dim3(16, 16);
        grid = dim3((wB + 15) / 16, (wA + 15) / 16);
        staticTileKernel<16><<<grid, block>>>(A, B, C, wA, wB);
    }
    else if (tileSize == 32) {
        block = dim3(32, 32);
        grid = dim3((wB + 31) / 32, (wA + 31) / 32);
        staticTileKernel<32><<<grid, block>>>(A, B, C, wA, wB);
    }
    else {
        printf("Unsupported tile size %d for static tiling.\n", tileSize);
    }
}
}
