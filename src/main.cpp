
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

// Global variable with timing results
profile_t timers[NUM_TIMERS];

// =================================================================================================

// Main function. This takes care of creating matrices of various sizes and iterating over the
// different types of BLAS libraries. It also computes the error rate in terms of the L2-norm with
// respect to cuBLAS (the 'golden' reference).
int main(int argc, char* argv[]) {

    // Start of the function
    printf("\n##\n");
    srand(time(NULL));

    // Set the performance counters to zero
    for (int t=0; t<NUM_TIMERS; t++) {
        timers[t].t = 0.0;
        timers[t].kf = 0;
    }

    // Compute the peak performance of the GPU
    double peak = GPU_CLOCK * GPU_CORES * GPU_MOD;

    // Print information about the different configurations
    printf("## --- Configurations ---\n");
    for (int c=0; c<=3; c++) {
        switch(c) {
            case 0: printf("##    cuBLAS on '%s', peak: %.1lf GFLOPS\n", GPU_NAME, peak); break;
            case 1: printf("##    clBlas on '%s', peak: %.1lf GFLOPS\n", GPU_NAME, peak); break;
            case 2: printf("## myGEMM.cu on '%s', peak: %.1lf GFLOPS\n", GPU_NAME, peak); break;
            case 3: printf("## myGEMM.cl on '%s', peak: %.1lf GFLOPS\n", GPU_NAME, peak); break;
        }
    }

    // Loop over the different input/output matrix sizes
    for (int size=MINSIZE; size<=MAXSIZE; size=size*2) {

        // Set the matrices to be squared (change this to get rectangular matrices)
        const int k = size;
        const int m = size;
        const int n = size;
        printf("##\n");
        printf("## --- %dx%dx%d ---\n", k, m, n);

        // Allocate memory for the matrices and fill the inputs with random numbers
        float* A = (float*)malloc(m*k*sizeof(float*));
        float* B = (float*)malloc(k*n*sizeof(float*));
        float* C = (float*)malloc(m*n*sizeof(float*));
        float* goldC = (float*)malloc(MAXSIZE*MAXSIZE*sizeof(float*));
        for (int i=0; i<m*k; i++) {
            A[i] = (float)rand() / (float)RAND_MAX;
        }
        for (int i=0; i<k*n; i++) {
            B[i] = (float)rand() / (float)RAND_MAX;
        }

        // Run cuBLAS first to get the 'golden' reference output
        libcublas(A, B, goldC, k, m, n, NUM_TIMERS-1);

        // Loop over the configurations
        for (int c=0; c<=3; c++) {

            // Set the output matrix to zero (to erase the results of the previous run)
            for (int i=0; i<m*n; i++) {
                C[i] = 0.0f;
            }

            // Get the name of the configuration
            char name[100];
            switch(c) {
                case 0: sprintf(name, "cuBLAS"); break;
                case 1: sprintf(name, "clBlas"); break;
                case 2: sprintf(name, "myGEMM.cu"); break;
                case 3: sprintf(name, "myGEMM.cl"); break;
            }

            // Perform the matrix-multiplication
            switch(c) {
                case 0: libcublas(A, B, C, k, m, n, c); break;
                case 1: libclblas(A, B, C, k, m, n, c); break;
                case 2: mycublas(A, B, C, k, m, n, c); break;
                case 3: myclblas(A, B, C, k, m, n, c); break;
            }

            // Compare the result to the 'golden' reference output in terms of the L2-norm
            double L2norm = 0.0;
            for (int i=0; i<m*n; i++) {
                double val = C[i] - goldC[i];
                L2norm += val*val;
            }
            L2norm = sqrt(L2norm);

            // Print the results to screen
            double seconds = wtime(timers[c]);
            double performance = gflops(timers[c]);
            double fraction = 100.0 * performance / peak;
            printf("## [%9s] %6.3lf s --> %6.1lf GFLOPS (%2.0lf%%), L2 norm: %.2e\n",
                   name, seconds, performance, fraction, L2norm);
        }

        // Free up the matrices
        free(A);
        free(B);
        free(C);
        free(goldC);
    }

    // End of the program
    printf("##\n");
    printf("\n");
    return 0;
}

// =================================================================================================

// Timer function: Measure the current time
double timer(void) {
    struct timeval Tvalue;
    struct timezone dummy;
    gettimeofday(&Tvalue, &dummy);
    double etime = (double)Tvalue.tv_sec + 1.0e-6*((double)Tvalue.tv_usec);
    return etime;
    //return omp_get_wtime();
}

// Timer function: Get the execution time
double wtime(profile_t timer) {
    return (timer.t);
}

// Timer function: Get the GFLOPS number
double gflops(profile_t timer) {
    return ((double)timer.kf/(1000.0*1000.0)) / (timer.t);
}

// =================================================================================================

// Load an OpenCL kernel from file
char* readKernelFile(const char* filename, long* _size) {

    // Open the file
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("-- Error opening file %s\n", filename);
        exit(1);
    }

    // Get its size
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    rewind(file);

    // Read the kernel code as a string
    char* source = (char *)malloc((size+1)*sizeof(char));
    fread(source, 1, size*sizeof(char), file);
    source[size] = '\0';
    fclose(file);

    // Save the size and return the source string
    *_size = (size+1);
    return source;
}

// =================================================================================================
