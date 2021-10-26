#include <stdio.h>
#include <stdlib.h>
//#include "kernels.cu.h"
#include "cub.cuh"
#include "helper.cu.h"

// Validation - use sequential sorting
// Generate data
// Timing
// kernel calling function - call kernels
// main function
// remember to free data

// Generate data
/**
 * size             is the size of the data array
 * random_array     is the data array holding random data
 */
int* make_random_array(uint32_t size) {
    int* random_array = malloc(sizeof(int32_t) * size); //Skal vi i stedet bruge CUDA malloc?
    for(uint32_t i = 0; i < size; i++) {
        random_array[i] = rand();
    }
    return random_array;
}

// Execute CUB-library radix sort on input_array.
// This function is copied from the example code in CUBcode
double sortRedByKeyCUB( uint32_t* data_keys_in
                      , uint32_t* data_keys_out
                      , const uint64_t N
) {
    int beg_bit = 0;
    int end_bit = 32;

    void * tmp_sort_mem = NULL;
    size_t tmp_sort_len = 0;

    { // sort prelude
        cub::DeviceRadixSort::SortKeys( tmp_sort_mem, tmp_sort_len
                                      , data_keys_in, data_keys_out
                                      , N,   beg_bit,  end_bit
                                      );
        cudaMalloc(&tmp_sort_mem, tmp_sort_len);
    }
    cudaCheckError();

    { // one dry run
        cub::DeviceRadixSort::SortKeys( tmp_sort_mem, tmp_sort_len
                                      , data_keys_in, data_keys_out
                                      , N,   beg_bit,  end_bit
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
                                      , data_keys_in, data_keys_out
                                      , N,   beg_bit,  end_bit
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


// Error message
void print_usage() {
    printf("Usage: %s <num_elements>\n", argv[0]);
}


int main(int argc, char* argv[]) {
    // Validation of arguments
    if (argc != 1 + 1) {
        print_usage();
        exit(1);
    }

    // Check that the input array size is larger than 0
    const uint32_t num_elements = atoi(argv[1]);
    if (num_elements_tmp <= 0) {
        printf("Number of elements should be greater than 0\n");
        print_usage();
        exit(1);
    }

    // skal vi lave en copi af input array'et, så vi ikke ødelægger det når vi
    // først sender det til CUB library og så derefter vores eget?

    //uint32_t num_elements = (uint32_t)num_elements_tmp;

    //Create input_array with random values
    int32_t* input_array = make_random_array(num_elements);
    //Allocate for output_array that will hold the results
    int32_t* output_array  = (int32_t*) malloc(num_elements*sizeof(int32_t));

    //Allocate and Initialize Device data
    int32_t* d_keys_in;
    int32_t* d_keys_out;
    cudaSucceeded(cudaMalloc((void**) &d_keys_in, num_elements * sizeof(int32_t)));
    cudaSucceeded(cudaMemcpy(d_keys_in, input_array, num_elements * sizeof(int32_t), cudaMemcpyHostToDevice));
    cudaSucceeded(cudaMalloc((void**) &d_keys_out, num_elements * sizeof(int32_t)));

    double elapsed = sortRedByKeyCUB( d_keys_in, d_keys_out, num_elements);

    cudaMemcpy(output_array, d_keys_out, num_elements*sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaCheckError();

    //bool success = validateZ(h_keys_res, N);

    printf("CUB Sorting for N=%lu runs in: %.2f us, VALID: %d\n", N, elapsed, success);

    // Cleanup and closing
    cudaFree(d_keys_in); cudaFree(d_keys_out);
    free(input_array); free(output_array);

    return success ? 0 : 1;
}
}
