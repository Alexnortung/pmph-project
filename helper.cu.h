#ifndef HISTO_HELPER
#define HISTO_HELPER

#include "cub-1.8.0/cub/cub.cuh"
#include <assert.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define GPU_RUNS                   400
#define NUM_BITS                   1
#define ELEM_PER_THREAD_MAKE_HIST  4


//To pass on function name
typedef double (*functiontype)(uint32_t*, uint32_t*, const uint64_t);

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
  unsigned int resolution=1000000;
  long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
  result->tv_sec = diff / resolution;
  result->tv_usec = diff % resolution;
  return (diff<0);
}

#define cudaCheckError() {                                              \
    cudaError_t e=cudaGetLastError();                                   \
    if(e!=cudaSuccess) {                                                \
      printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
      exit(0);                                                          \
    }                                                                   \
}

#define cudaSucceeded(ans) { cudaAssert((ans), __FILE__, __LINE__); }

inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) {
    std::cerr << "cudaAssert failed: "
              << cudaGetErrorString(code)
              << file << ":" << line
              << std::endl;
    if (abort) {
      exit(code);
    }
  }
}

inline uint32_t ceilLog2(uint32_t H) {
    if (H == 0) { printf("Log2(0) is illegal. Exiting!\n"); exit(1); }
    uint32_t log2_val = 0, pow2_val = 1;
    while(pow2_val < H) {
        log2_val ++;
        pow2_val *= 2;
    }
    return log2_val;
}

void writeRuntime(const char *fname, double elapsed) {
  FILE *f = fopen(fname, "w");
  assert(f != NULL);
  fprintf(f, "%f", elapsed);
  fclose(f);
}


// Validates if the output array is correctly sorted
/**
 * num_elems        is the size of the output array
 * output_array     is the output array that has been sorted
 */
bool validate(uint32_t* output_array, uint32_t num_elems) {
    for(uint32_t i = 0; i < num_elems-1; i++){
        if (output_array[i] > output_array[i+1]){
            printf("INVALID RESULT for i:%d, (output[i-1]=%d > output[i]=%d)\n", i, output_array[i-1], output_array[i]);
            return false;
        }
    }
    return true;
}


// This function allocates cuda memory for either the CUB or kernel
// implementation.
/**
 * num_elements     is the size of the input and output arrays
 * input_array      is the array that holds the data that is to be sorted
 * output_array     is the array that holds the sorted data
 * func             is either the funtion of sortRedByKeyCUB() or sortByKernel()
 */
template<class T>
double allocate_initiate(uint32_t num_elements
                        , T* input_array
                        , T* output_array
                        , functiontype func){
    T* d_keys_in;
    T* d_keys_out;
    cudaSucceeded(cudaMalloc((void**) &d_keys_in, num_elements * sizeof(T)));
    cudaSucceeded(cudaMemcpy(d_keys_in, input_array, num_elements * sizeof(T), cudaMemcpyHostToDevice));
    cudaSucceeded(cudaMalloc((void**) &d_keys_out, num_elements * sizeof(T)));

    double elapsed = func( d_keys_in, d_keys_out, num_elements);

    cudaMemcpy(output_array, d_keys_out, num_elements*sizeof(T), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaCheckError();

    // clean up
    cudaFree(d_keys_in); cudaFree(d_keys_out);
    return elapsed;
}

// Error message
void print_usage(char* arg) {
    printf("Usage: %s <num_elements>\n", arg);
}

/********************************************************
 ****************Generate data for test******************
 ********************************************************/

// Generate random (positive) integer data 
/**
 * size             is the size of the data array
 * rand_in_arr      is the input array holding random data
 */
unsigned int* make_rand_int_array(uint32_t size) {
    uint32_t* rand_in_arr = (uint32_t*) malloc(size * sizeof(uint32_t));
    for(uint32_t i = 0; i < size; i++) {
        rand_in_arr[i] = rand() % size;
    }
    return rand_in_arr;
}

// Generate random (positive) float data 
/**
 * size             is the size of the data array
 * rand_in_arr      is the input array holding random data
 */
float* make_rand_fl_array(uint32_t size) {
    float* rand_in_arr = (float*) malloc(size * sizeof(float));
    for(uint32_t i = 0; i < size; i++) {
        rand_in_arr[i] = (float)rand(); //% (float) size;
    }
    return rand_in_arr;
}

// Generate random (positive) tuple data 
/**
 * size             is the size of the data array
 * rand_in_arr      is the input array holding random data
 */
/*float* make_rand_tup_array(uint32_t size) {
    float* rand_in_arr = (float*) malloc(size * sizeof(float));
    for(uint32_t i = 0; i < size; i++) {
        rand_in_arr[i] = (float)rand();
    }
    return rand_in_arr;
}*/

// Generate konstant (0) integer data 
/**
 * size             is the size of the data array
 * input_arr        is the input array holding the data
 */
unsigned int* make_konstant_array(uint32_t size) {
    uint32_t* input_arr = (uint32_t*) malloc(size * sizeof(uint32_t));
    for(uint32_t i = 0; i < size; i++) {
        input_arr[i] = 0;
    }
    return input_arr;
}

// Generate sorted (low -> high) integer data 
/**
 * size             is the size of the data array
 * input_arr        is the input array holding the data
 */
unsigned int* make_sortLtoH_array(uint32_t size) {
    uint32_t* input_arr = (uint32_t*) malloc(size * sizeof(uint32_t));
    for(uint32_t i = 0; i < size; i++) {
        input_arr[i] = i;
    }
    return input_arr;
}

// Generate sorted (high -> low) integer data 
/**
 * size             is the size of the data array
 * input_arr        is the input array holding the data
 */
unsigned int* make_sortHtoL_array(uint32_t size) {
    uint32_t* input_arr = (uint32_t*) malloc(size * sizeof(uint32_t));
    for(uint32_t i = 0; i < size; i++) {
        input_arr[i] = size - i;
    }
    return input_arr;
}

#endif // HISTO_HELPER
