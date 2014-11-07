
// =================================================================================================
// Project: 
// Exploring the performance of general matrix-multiplication on an NVIDIA Tesla K40m GPU.
//
// File information:
// Institution.... SURFsara <www.surfsara.nl>
// Author......... Cedric Nugteren <cedric.nugteren@surfsara.nl>
// Changed at..... 2014-11-06
// License........ MIT license
// Tab-size....... 4 spaces
// Line length.... 100 characters
//
// =================================================================================================

// Replace the OpenCL keywords with CUDA equivalent
#define __kernel __placeholder__
#define __global 
#define __placeholder__ __global__
#define __local __shared__
#define restrict __restrict__

// Replace OpenCL synchronisation with CUDA synchronisation
#define barrier(x) __syncthreads()

// Replace the OpenCL get_xxx_ID with CUDA equivalents
__device__ int get_local_id(int x) {
    return (x == 0) ? threadIdx.x : threadIdx.y;
}
__device__ int get_group_id(int x) {
    return (x == 0) ? blockIdx.x : blockIdx.y;
}
__device__ int get_global_id(int x) {
    return (x == 0) ? blockIdx.x*blockDim.x + threadIdx.x : blockIdx.y*blockDim.y + threadIdx.y;
}

// Add the float8 data-type which is not available natively under CUDA
typedef struct { float s0; float s1; float s2; float s3;
                 float s4; float s5; float s6; float s7; } float8;

// =================================================================================================
