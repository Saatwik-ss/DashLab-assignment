#include <stdio.h>
#include <cuda_runtime.h>

#define TILE 8

__global__ void matmul_naive(const float *A,const float *B,float *C,int M,int N,int K){
 int row=blockIdx.y*blockDim.y+threadIdx.y;
 int col=blockIdx.x*blockDim.x+threadIdx.x;

#define RB 2  // register block (each thread computes 2x2 outputs)

__shared__ float As[TILE][TILE];
__shared__ float Bs[TILE][TILE];

float regC[RB][RB] = {0};

// Compute a 2x2 tile of C
for (int t = 0; t < (N + TILE - 1) / TILE; t++) {
    if (row < M && t*TILE + threadIdx.x < N)
        As[threadIdx.y][threadIdx.x] = A[row*N + t*TILE + threadIdx.x];
    else As[threadIdx.y][threadIdx.x] = 0;

    if (col < K && t*TILE + threadIdx.y < N)
        Bs[threadIdx.y][threadIdx.x] = B[(t*TILE + threadIdx.y)*K + col];
    else Bs[threadIdx.y][threadIdx.x] = 0;

    __syncthreads();

    for (int i = 0; i < TILE; i++) {
        float a0 = As[threadIdx.y][i];
        float a1 = As[threadIdx.y + 1][i];      // reuse of neighboring rows
        float b0 = Bs[i][threadIdx.x];
        float b1 = Bs[i][threadIdx.x + 1];      // reuse neighboring column
        regC[0][0] += a0* b0;
        regC[0][1] += a0 *b1;
        regC[1][0] += a1*b0;
        regC[1][1] += a1*b1;
    }
    __syncthreads();
}

// store results
int row0 = blockIdx.y*TILE + threadIdx.y;
int col0 = blockIdx.x*TILE + threadIdx.x;
if(row0<M && col0 <K) C[row0*K +col0] = regC[0][0];
if(row0 <M && col0+1 < K) C[row0*K+ col0+1] = regC[0][1];
if(row0+1< M && col0 < K)      C[(row0+1)*K+ col0] = regC[1][0];
if(row0+1<M && col0+1 < K)    C[(row0+1)*K+ col0+1] = regC[1][1];

 }



void fill(float *a,int m,int n){
  for(int i=0;i<m*n;i++) a[i]=(float)(rand()%10);
}

int main(){
 int M=1024,N=1024,K=1024;
 size_t sA=M*N*sizeof(float), sB=N*K*sizeof(float), sC=M*K*sizeof(float);
 float *hA=(float*)malloc(sA);
 float *hB=(float*)malloc(sB);
 float *hC=(float*)malloc(sC);
 fill(hA,M,N); fill(hB,N,K);
 float *dA,*dB,*dC;
 cudaMalloc(&dA,sA); cudaMalloc(&dB,sB); cudaMalloc(&dC,sC);
 cudaMemcpy(dA,hA,sA,cudaMemcpyHostToDevice);
 cudaMemcpy(dB,hB,sB,cudaMemcpyHostToDevice);
 dim3 block(TILE,TILE);
 dim3 grid((K+TILE-1)/TILE,(M+TILE-1)/TILE);

 cudaEvent_t start,stop;
 cudaEventCreate(&start); cudaEventCreate(&stop);

 cudaEventRecord(start);
 matmul_naive<<<grid,block>>>(dA,dB,dC,M,N,K);
 cudaEventRecord(stop);
 cudaEventSynchronize(stop);
 float ms1;
 cudaEventElapsedTime(&ms1,start,stop);
 double gflops1=2.0*M*N*K/(ms1/1000.0)/1e9;
 printf("Naive: %.4f ms  %.2f GFLOPS\n",ms1,gflops1);



 cudaFree(dA); cudaFree(dB); cudaFree(dC);
 free(hA); free(hB); free(hC);
 return 0;
}
