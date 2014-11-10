
// =================================================================================================
// Project: 
// Exploring the performance of general matrix-multiplication on an NVIDIA Tesla K40m GPU.
//
// File information:
// Institution.... SURFsara <www.surfsara.nl>
// Author......... Cedric Nugteren <cedric.nugteren@surfsara.nl>
// Changed at..... 2014-11-10
// License........ MIT license
// Tab-size....... 4 spaces
// Line length.... 100 characters
//
// =================================================================================================

// Common C includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

// =================================================================================================

// Repeat all kernels multiple times to get an average timing result
#define NUM_RUNS 4

// Squared matrices are tested within a certain range (e.g. 1024x1024, 2048x2048, 4096x4096)
#define MINSIZE (1024)
#define MAXSIZE (4*1024)

// Set the alpha and beta values for the cuBLAS and clBlas libraries. Note that the myGEMM kernels
// for simplicity only support alpha values of 1 and beta values of 0.
#define ALPHA 1.0f
#define BETA 0.0f

// Define the current GPU's parameters
#define GPU_NAME "Tesla K40m"
#define GPU_CLOCK 0.745 // Core clock in GHz
#define GPU_CORES 2880 // Total number of CUDA cores
#define GPU_MOD 2 // Fused multiply-add

// OpenCL settings
#define MAX_NUM_DEVICES 16
#define MAX_DEVICE_NAME 1024
#define CURRENT_DEVICE 1

// =================================================================================================

// Timer structure
typedef struct {
    double t; // Time
    int long long kf; // KFlops
} profile_t;

// Number of timers
#define NUM_TIMERS 10

// Global variable holding the timing results
extern profile_t timers[NUM_TIMERS];

// =================================================================================================

// Forward declarations of BLAS functions
void libcublas(float* A, float* B, float* C,
               int K, int M, int N,
               int timerID);
void libclblas(float* A, float* B, float* C,
               int K, int M, int N,
               int timerID);
void mycublas(float* A, float* B, float* C,
              int K, int M, int N,
              int timerID);
void myclblas(float* A, float* B, float* C,
              int K, int M, int N,
              int timerID);

// Forward declarations of the timer functions
double timer(void);
double wtime(profile_t timer);
double gflops(profile_t timer);

// Other forward declarations
char* readKernelFile(const char* filename, long* _size);

// =================================================================================================
