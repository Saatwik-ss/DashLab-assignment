#include <stdio.h>
#include <cuda_runtime.h>

#define TILE 16
#define RB   2   // Register block size: each thread computes 2x2 outputs

__global__ void matmul_regblock(const float *A, const float *B, float *C,
                                int M, int N, int K) {
    // Compute the base thread coordinates
    int ty = threadIdx.y * RB;
    int tx = threadIdx.x * RB;

    int rowBase = blockIdx.y * TILE + ty;
    int colBase = blockIdx.x * TILE + tx;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    float regC[RB][RB] = {0};

    // Loop over tiles
    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        // Load A tile
        for (int i = 0; i < RB; ++i) {
            int row = rowBase + i;
            int col = t * TILE + threadIdx.x;
            As[threadIdx.y * RB + i][threadIdx.x] =
                (row < M && col < N) ? A[row * N + col] : 0.0f;
        }

        // Load B tile
        for (int j = 0; j < RB; ++j) {
            int row = t * TILE + threadIdx.y;
            int col = colBase + j;
            Bs[threadIdx.y][threadIdx.x * RB + j] =
                (row < N && col < K) ? B[row * K + col] : 0.0f;
        }

        __syncthreads();

        // Compute partial results using register blocking
        for (int k = 0; k < TILE; ++k) {
            float aReg[RB];
            float bReg[RB];
            for (int i = 0; i < RB; ++i) aReg[i] = As[threadIdx.y * RB + i][k];
            for (int j = 0; j < RB; ++j) bReg[j] = Bs[k][threadIdx.x * RB + j];

            // Compute all 2x2 combinations
            for (int i = 0; i < RB; ++i)
                for (int j = 0; j < RB; ++j)
                    regC[i][j] += aReg[i] * bReg[j];
        }

        __syncthreads();
    }

    // Write results back to global memory
    for (int i = 0; i < RB; ++i) {
        int row = rowBase + i;
        if (row < M) {
            for (int j = 0; j < RB; ++j) {
                int col = colBase + j;
                if (col < K)
                    C[row * K + col] = regC[i][j];
            }
        }
    }
}

// Utility function to fill matrices
void fill(float *a, int m, int n) {
    for (int i = 0; i < m * n; i++) a[i] = (float)(rand() % 10);
}

int main() {
    int M = 1024, N = 1024, K = 1024;
    size_t sA = M * N * sizeof(float), sB = N * K * sizeof(float), sC = M * K * sizeof(float);

    float *hA = (float*)malloc(sA);
    float *hB = (float*)malloc(sB);
    float *hC = (float*)malloc(sC);

    fill(hA, M, N);
    fill(hB, N, K);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, sA);
    cudaMalloc(&dB, sB);
    cudaMalloc(&dC, sC);

    cudaMemcpy(dA, hA, sA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sB, cudaMemcpyHostToDevice);

    dim3 block(TILE / RB, TILE / RB);
    dim3 grid((K + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_regblock<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    double gflops = 2.0 * M * N * K / (ms / 1000.0) / 1e9;
    printf("Register-blocked: %.4f ms  %.2f GFLOPS\n", ms, gflops);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}


//nvcc -arch=sm_89 tiled.cu -o tiled.exe
//nvprof ./tiled.exe
