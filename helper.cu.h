#ifndef HISTO_HELPER
#define HISTO_HELPER

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>

#define GPU_RUNS    400
#define NUM_BITS    1
#define ELEM_PER_THREAD  4

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

// Execute CUB-library radix sort on input_array.
// This function is copied from the example code in CUBcode
double sortRedByKeyCUB( uint32_t* input_array
                      , uint32_t* output_array
                      , const uint64_t num_elem
) {
    int beg_bit = 0;
    int end_bit = 32;

    void * tmp_sort_mem = NULL;
    size_t tmp_sort_len = 0;

    { // sort prelude
        cub::DeviceRadixSort::SortKeys( tmp_sort_mem, tmp_sort_len
                                      , input_array, output_array
                                      , num_elem,   beg_bit,  end_bit
                                      );
        cudaMalloc(&tmp_sort_mem, tmp_sort_len);
    }
    cudaCheckError();

    { // one dry run
        cub::DeviceRadixSort::SortKeys( tmp_sort_mem, tmp_sort_len
                                      , input_array, output_array
                                      , num_elem,   beg_bit,  end_bit
                                      );
        cudaDeviceSynchronize();
    }
    cudaCheckError();

    // timing
    double elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    for(int k=0; k<GPU_RUNS; k++) {
        cub::DeviceRadixSort::SortKeys( tmp_sort_mem, tmp_sort_len
                                      , input_array, output_array
                                      , num_elem,   beg_bit,  end_bit
                                      );
    }
    cudaDeviceSynchronize();
    cudaCheckError();

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / ((double)GPU_RUNS);

    cudaFree(tmp_sort_mem);

    return elapsed;
}

bool validate(uint32_t* output_array, uint32_t num_elems) {
    for(uint32_t i = 0; i < num_elems-1; i++){
        if (output_array[i] > output_array[i+1]){
            printf("INVALID RESULT for i:%d, (output[i-1]=%d > output[i]=%d)\n", i, output_array[i-1], output_array[i]);
            return false;
        }
    }
    return true;
}

// Generate data
/**
 * size             is the size of the data array
 * rand_in_arr      is the input array holding random data
 */
unsigned int* make_random_array(uint32_t size) {
    uint32_t* rand_in_arr = (uint32_t*) malloc(size * sizeof(uint32_t));
    for(uint32_t i = 0; i < size; i++) {
        rand_in_arr[i] = rand();
    }
    return rand_in_arr;
}

// This function allocates cuda memory for either the CUB or kernel
// implementation.
/**
 * num_elements     is the size of the input and output arrays
 * input_array      is the array that holds the data that is to be sorted
 * output_array     is the array that holds the sorted data
 * func             is either the funtion of sortRedByKeyCUB() or sortByKernel()
 */
double allocate_initiate(uint32_t num_elements
                        , uint32_t* input_array
                        , uint32_t* output_array
                        , functiontype func){
    uint32_t* d_keys_in;
    uint32_t* d_keys_out;
    cudaSucceeded(cudaMalloc((void**) &d_keys_in, num_elements * sizeof(uint32_t)));
    cudaSucceeded(cudaMemcpy(d_keys_in, input_array, num_elements * sizeof(uint32_t), cudaMemcpyHostToDevice));
    cudaSucceeded(cudaMalloc((void**) &d_keys_out, num_elements * sizeof(uint32_t)));

    double elapsed = func( d_keys_in, d_keys_out, num_elements);

    cudaMemcpy(output_array, d_keys_out, num_elements*sizeof(uint32_t), cudaMemcpyDeviceToHost);
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
#endif // HISTO_HELPER
