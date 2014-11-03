
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

// Include OpenCL and clBlas
#include <clBLAS.h>

// =================================================================================================

// Matrix-multiplication using the clBlas library. This function copies the input matrices to the
// GPU, runs SGEMM, and copies the output matrix back to the CPU.
void libclblas(float* A, float* B, float* C,
               int K, int M, int N,
               int timerID) {
    cl_int err;

    // Define OpenCL variables
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, 0, 0};
    cl_context ctx = 0;
    cl_command_queue queue = 0;
    cl_event event = NULL;

    // Configure the OpenCL environment
    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(ctx, device, 0, &err);

    // Configure clBlas
    err = clblasSetup();

    // Prepare OpenCL memory objects
    cl_mem bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, M*K*sizeof(*A), NULL, &err);
    cl_mem bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, K*N*sizeof(*B), NULL, &err);
    cl_mem bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, M*N*sizeof(*C), NULL, &err);

    // Copy matrices to the GPU (also C to erase the results of the previous run)
    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, M*K*sizeof(*A), A, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, K*N*sizeof(*B), B, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0, M*N*sizeof(*C), C, 0, NULL, NULL);

    // Start the timed loop
    double startTime = timer();
    for (int r=0; r<NUM_RUNS; r++) {

        // Call clBlas
        err = clblasSgemm(clblasColumnMajor, clblasNoTrans, clblasNoTrans,
                          M, N, K, ALPHA,
                          bufA, 0, M,
                          bufB, 0, K, BETA,
                          bufC, 0, M,
                          1, &queue, 0, NULL, &event);

        // Wait for calculations to be finished
        err = clWaitForEvents(1, &event);
    }

    // End the timed loop
    timers[timerID].t += (timer() - startTime) / (double)NUM_RUNS;
    timers[timerID].kf += ((long)K * (long)M * (long)N * 2) / 1000;

    // Copy the output matrix C back to the CPU memory
    err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, M*N*sizeof(*C), C, 0, NULL, NULL);

    // Free the GPU memory objects
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);

    // Clean-up OpenCL and clBlas 
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
}

// =================================================================================================
