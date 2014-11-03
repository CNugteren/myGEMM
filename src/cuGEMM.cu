
// =================================================================================================
// Project: 
// Exploring the performance of general matrix-multiplication on an NVIDIA Tesla K40m GPU.
//
// File information:
// Institution.... SURFsara <www.surfsara.nl>
// Author......... Cedric Nugteren <cedric.nugteren@surfsara.nl>
// Changed at..... 2014-10-31
// License........ MIT license
// Tab-size....... 4 spaces
// Line length.... 100 characters
//
// =================================================================================================

// Common include
#include "common.h"

// Include kernel constants
#include "settings.h"

// =================================================================================================

// Configuration settings for the CUDA version (comment out if not desired)
#define USE_LDG         // Whether to use the __ldg() intrinsic
//#define USE_SHUFFLE   // Whether to use warp-shuffle instructions

// Include the OpenCL-to-CUDA header and the OpenCL kernel-code
#include "cl_to_cuda.h"
#include "kernels.cl"

// =================================================================================================

// Matrix-multiplication using a custom CUDA SGEMM kernel. This function also copies the input
// matrices to the GPU, runs SGEMM, and copies the output matrix back to the CPU.
void mycublas(float* A, float* B, float* C,
              int K, int M, int N,
              int timerID) {

    // In case of myGEMM10, compute matrix sizes K, M, N as rounded-up to form complete tiles
    #if KERNEL == 10
        int K_XL = CEIL_DIV(K, TSK) * TSK;
        int M_XL = CEIL_DIV(M, TSM) * TSM;
        int N_XL = CEIL_DIV(N, TSN) * TSN;
    #else
        int K_XL = K;
        int M_XL = M;
        int N_XL = N;
    #endif

    // Prepare CUDA memory objects
    float* bufA = 0;
    float* bufB = 0;
    float* bufB_TR = 0; // This is the transposed version of B
    float* bufC = 0;
    cudaMalloc((void**)&bufA,    M*K*sizeof(*A));
    cudaMalloc((void**)&bufB,    K*N*sizeof(*B));
    cudaMalloc((void**)&bufB_TR, N*K*sizeof(*B));
    cudaMalloc((void**)&bufC,    M*N*sizeof(*C));

    // Copy matrices to the GPU (memset C to erase the results of the previous run)
    cudaMemcpy((void*)bufA, (void*)A, M*K*sizeof(*A), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)bufB, (void*)B, K*N*sizeof(*B), cudaMemcpyHostToDevice);
    cudaMemset((void*)bufC, 0.0, M*N*sizeof(*C));

    // Create extra objects for rounded-up sizes (only needed in case of myGEMM10)
    float* bufA_XL = 0;
    float* bufB_TR_XL = 0;
    float* bufC_XL = 0;
    cudaMalloc((void**)&bufA_XL,    M_XL*K_XL*sizeof(*A));
    cudaMalloc((void**)&bufB_TR_XL, K_XL*N_XL*sizeof(*B));
    cudaMalloc((void**)&bufC_XL,    M_XL*N_XL*sizeof(*C));

    // Configure the local memory (banks of 8 bytes, 48KB local memory)
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    // Configure the thread/threadblock dimensions of the transpose kernel (only for certain myGEMMs)
    #if KERNEL == 5 || KERNEL == 6 || KERNEL == 7 || KERNEL == 8 || KERNEL == 9 || KERNEL == 10
        dim3 blocksTRP(CEIL_DIV(K,TRANSPOSEX), CEIL_DIV(N,TRANSPOSEY));
        dim3 threadsTRP(TRANSPOSEX, TRANSPOSEY);
    #endif

    // Configure the thread/threadblock dimensions of the padding kernels (only for myGEMM10)
    #if KERNEL == 10
        dim3 blocksA(CEIL_DIV(M_XL,PADDINGX), CEIL_DIV(K_XL,PADDINGY));
        dim3 threadsA(PADDINGX, PADDINGY);
        dim3 blocksB(CEIL_DIV(N_XL,PADDINGX), CEIL_DIV(K_XL,PADDINGY));
        dim3 threadsB(PADDINGX, PADDINGY);
        dim3 blocksC(CEIL_DIV(M,PADDINGX), CEIL_DIV(N,PADDINGY));
        dim3 threadsC(PADDINGX, PADDINGY);
    #endif

    // Configure the thread/threadblock dimensions of the myGEMM kernel
    #if KERNEL == 1 || KERNEL == 2
        dim3 blocks(M/TS, N/TS);
        dim3 threads(TS, TS);
    #elif KERNEL == 3 || KERNEL == 5
        dim3 blocks(M/TS, N/TS);
        dim3 threads(TS, TS/WPT);
    #elif KERNEL == 4
        dim3 blocks(M/TS, N/TS);
        dim3 threads(TS/WIDTH, TS);
    #elif KERNEL == 6 || KERNEL == 7 || KERNEL == 8 || KERNEL == 9
        dim3 blocks(M/TSM, N/TSN);
        dim3 threads(TSM/WPTM, TSN/WPTN);
    #elif KERNEL == 10
        dim3 blocks(M_XL/TSM, N_XL/TSN);
        dim3 threads(TSM/WPTM, TSN/WPTN);
    #endif

    // Start the timed loop
    double startTime = timer();
    for (int r=0; r<NUM_RUNS; r++) {

        // Run the transpose kernel first
        #if KERNEL == 5 || KERNEL == 6 || KERNEL == 7 || KERNEL == 8 || KERNEL == 9 || KERNEL == 10
            transpose<<<blocksTRP, threadsTRP>>>(K, N, bufB, bufB_TR);
        #endif

        // Make the inputs extra large with padded zeros
        #if KERNEL == 10
            paddingAddZeroes<<<blocksA, threadsA>>>(M, K, bufA, M_XL, K_XL, bufA_XL);
            paddingAddZeroes<<<blocksB, threadsB>>>(N, K, bufB_TR, N_XL, K_XL, bufB_TR_XL);
        #endif

        // Run the myGEMM kernel
        #if KERNEL == 1
            myGEMM1<<<blocks, threads>>>(M, N, K, bufA, bufB, bufC);
        #elif KERNEL == 2
            myGEMM2<<<blocks, threads>>>(M, N, K, bufA, bufB, bufC);
        #elif KERNEL == 3
            myGEMM3<<<blocks, threads>>>(M, N, K, bufA, bufB, bufC);
        #elif KERNEL == 4
            myGEMM4<<<blocks, threads>>>(M, N, K, (floatX*)bufA, (floatX*)bufB, (floatX*)bufC);
        #elif KERNEL == 5
            myGEMM5<<<blocks, threads>>>(M, N, K, bufA, bufB_TR, bufC);
        #elif KERNEL == 6
            myGEMM6<<<blocks, threads>>>(M, N, K, bufA, bufB_TR, bufC);
        #elif KERNEL == 7
            myGEMM7<<<blocks, threads>>>(M, N, K, (floatX*)bufA, (floatX*)bufB_TR, bufC);
        #elif KERNEL == 8
            myGEMM8<<<blocks, threads>>>(M, N, K, (floatX*)bufA, (floatX*)bufB_TR, bufC);
        #elif KERNEL == 9
            myGEMM9<<<blocks, threads>>>(M, N, K, (floatX*)bufA, (floatX*)bufB_TR, bufC);
        #elif KERNEL == 10
            myGEMM10<<<blocks, threads>>>(M_XL, N_XL, K_XL, (floatX*)bufA_XL, (floatX*)bufB_TR_XL, bufC_XL);
        #endif

        // Remove padded zeroes from the larger output
        #if KERNEL == 10
            paddingRemoveZeroes<<<blocksC, threadsC>>>(M_XL, N_XL, bufC_XL, M, N, bufC);
        #endif

        // Wait for calculations to be finished
        cudaDeviceSynchronize();
    }

    // End the timed loop
    timers[timerID].t += (timer() - startTime) / (double)NUM_RUNS;
    timers[timerID].kf += ((long)K * (long)M * (long)N * 2) / 1000;

    // Copy the output matrix C back to the CPU memory
    cudaMemcpy((void*)C, (void*)bufC, M*N*sizeof(*C), cudaMemcpyDeviceToHost);

    // Free the GPU memory objects
    cudaFree(bufA);
    cudaFree(bufB);
    cudaFree(bufB_TR);
    cudaFree(bufC);
    cudaFree(bufA_XL);
    cudaFree(bufB_TR_XL);
    cudaFree(bufC_XL);
}

// =================================================================================================
