# Background:
I have worked on modifying(not fully writing) CUDA kernels for SAiDL and have practiced some questions on LEETGPU, so i had some experience in reading and filling the code but not fully writing kernels from scratch.
---



# Naive CUDA Matrix Multiplication

## 1. Objective
The first step was to implement a simple matrix multiplier, something i have done in C++ and torch for some previous works, so wasn't very difficult to implement. Took the basic LeetGPU format of code and implemented the basic formula needed.

Implemented a method to multiply any two matrices of dimensions $(A × B)$ and $(B × C)$, which can be formulated as below.

---

## 2. Mathematical Background

For two matrices **A** and **B** of size $N × M$ and $(M × L)$, their product **C = A × B** is defined as:

$$
C[i][j] = \sum_{k=0}^{M-1} A[i][k] \times B[k][j]
$$

Each element $C[i][j]$ represents the **dot product** of the $i^{th}$ row of **A** and the $j^{th}$ column of **B**.

---

## 3. Implementation Concept

In this, each thread is assigned to compute one element of the output matrix C.

The global thread coordinates are computed as below:

$$
\text{row} = \text{blockIdx.y} \times \text{blockDim.y} + \text{threadIdx.y}
$$

$$
\text{col} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}
$$

Since 2D matrices are flattned in GPU, this formula works.

### Problems faced: 

Switched order of x and y threads over rows and columns.

<img width="1171" height="550" alt="image" src="https://github.com/user-attachments/assets/6bf0e174-ad33-48bf-a141-d38f93850379" />


Just looking at the profiler, it seemed as the gflops were much higher than expected. The GFLOPS for CuBlas infact are around 410 for the same compute, putting my effeciency around 56%, indicating some issues in profiling.

I then tested my kernel in kaggle and other machines too but the GFlops were around 230.

### Bottleneck:

The kernel shows a clear memory bandwidth bottleneck. About 68% of GPU time (≈9.17 ms) is spent inside matrixMulNaive, while global memory transfers (HtoD + DtoH) take another ~6 ms. Each thread performs N³ = 1024³ ≈ 1.07×10⁹ multiply-adds but also issues roughly 2×N³ = 2.1×10⁹ global memory reads/writes with almost no data reuse. The achieved throughput is ~230 GFLOPS, only around 55% of cuBLAS peak on a T4 GPU. The non-coalesced column access to matrix B and lack of shared-memory tiling make it heavily memory-bound rather than compute-bound.

     
---






