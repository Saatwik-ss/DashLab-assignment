
#include <iostream>
#include <cuda_runtime.h>
#define TILE 16

__global__ void matmul_naive(const float*A,const float*B,float*C,int M,int N,int K){
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<M&&col<N){
        float val=0.0f;
        for(int k=0;k<K;k++) val+=A[row*K+k]*B[k*N+col];
        C[row*N+col]=val;
    }
}

__global__ void matmul_tiled(const float*A,const float*B,float*C,int M,int N,int K){
    __shared__ float As[TILE][TILE],Bs[TILE][TILE];
    int row=blockIdx.y*TILE+threadIdx.y;
    int col=blockIdx.x*TILE+threadIdx.x;
    float val=0.0f;
    for(int t=0;t<(K+TILE-1)/TILE;t++){
        if(row<M&&t*TILE+threadIdx.x<K) As[threadIdx.y][threadIdx.x]=A[row*K+t*TILE+threadIdx.x];
        else As[threadIdx.y][threadIdx.x]=0.0f;
        if(col<N&&t*TILE+threadIdx.y<K) Bs[threadIdx.y][threadIdx.x]=B[(t*TILE+threadIdx.y)*N+col];
        else Bs[threadIdx.y][threadIdx.x]=0.0f;
        __syncthreads();
        for(int i=0;i<TILE;i++) val+=As[threadIdx.y][i]*Bs[i][threadIdx.x];
        __syncthreads();
    }
    if(row<M&&col<N) C[row*N+col]=val;
}

int main(){
    int M=1024,N=1024,K=1024;
    size_t sA=M*K*sizeof(float),sB=K*N*sizeof(float),sC=M*N*sizeof(float);
    float*hA=(float*)malloc(sA),*hB=(float*)malloc(sB),*hC=(float*)malloc(sC);
    for(int i=0;i<M*K;i++) hA[i]=1.0f;
    for(int i=0;i<K*N;i++) hB[i]=1.0f;
    float*dA,*dB,*dC;
    cudaMalloc(&dA,sA);cudaMalloc(&dB,sB);cudaMalloc(&dC,sC);
    cudaMemcpy(dA,hA,sA,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,hB,sB,cudaMemcpyHostToDevice);
    dim3 threads(TILE,TILE),blocks((N+TILE-1)/TILE,(M+TILE-1)/TILE);
    cudaEvent_t start,stop;
    cudaEventCreate(&start);cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_naive<<<blocks,threads>>>(dA,dB,dC,M,N,K);
    cudaEventRecord(stop);cudaEventSynchronize(stop);
    float ms1;cudaEventElapsedTime(&ms1,start,stop);
    double gflops1=2.0*M*N*K/(ms1/1000.0)/1e9;
    printf("Naive: %.4f ms  %.2f GFLOPS\n",ms1,gflops1);

    cudaMemset(dC,0,sC);
    cudaEventRecord(start);
    matmul_tiled<<<blocks,threads>>>(dA,dB,dC,M,N,K);
    cudaEventRecord(stop);cudaEventSynchronize(stop);
    float ms2;cudaEventElapsedTime(&ms2,start,stop);
    double gflops2=2.0*M*N*K/(ms2/1000.0)/1e9;
    printf("Tiled: %.4f ms  %.2f GFLOPS\n",ms2,gflops2);

    cudaFree(dA);cudaFree(dB);cudaFree(dC);
    free(hA);free(hB);free(hC);
    return 0;
}



//!nvcc -O3 -arch=sm_75 tiled_01.cu -o tiled_01.exe
//!nvprof ./tiled_01.exe
