
// =================================================================================================
// Project: 
// Exploring the performance of general matrix-multiplication on an NVIDIA Tesla K40m GPU.
//
// File information:
// Institution.... SURFsara <www.surfsara.nl>
// Author......... Cedric Nugteren <cedric.nugteren@surfsara.nl>
// Changed at..... 2014-10-30
// License........ MIT license
// Tab-size....... 4 spaces
// Line length.... 100 characters
//
// =================================================================================================

// Common include
#include "common.h"

// Include CUDA and cuBLAS (API v2)
#include <cublas_v2.h>

// =================================================================================================

// Matrix-multiplication using the cuBLAS library. This function copies the input matrices to the
// GPU, runs SGEMM, and copies the output matrix back to the CPU.
void libcublas(float* A, float* B, float* C,
               int K, int M, int N,
               int timerID) {

    // cuBLAS configuration
    cublasStatus_t status;
    cublasHandle_t handle;
    status = cublasCreate(&handle);

    // Prepare CUDA memory objects
    float* bufA = 0;
    float* bufB = 0;
    float* bufC = 0;
    cudaMalloc((void**)&bufA, M*K*sizeof(*A));
    cudaMalloc((void**)&bufB, K*N*sizeof(*B));
    cudaMalloc((void**)&bufC, M*N*sizeof(*C));

    // Copy matrices to the GPU (also C to erase the results of the previous run)
    cudaMemcpy((void*)bufA, (void*)A, M*K*sizeof(*A), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)bufB, (void*)B, K*N*sizeof(*B), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)bufC, (void*)C, M*N*sizeof(*C), cudaMemcpyHostToDevice);

    // Configure SGEMM
    float alpha = ALPHA;
    float beta = BETA;

    // Start the timed loop
    double startTime = timer();
    for (int r=0; r<NUM_RUNS; r++) {

        // Call cuBLAS
        status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             M, N, K, &alpha,
                             bufA, M,
                             bufB, K, &beta,
                             bufC, M);

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
    cudaFree(bufC);

    // Clean-up cuBLAS
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        exit(1);
    }
}

// =================================================================================================
