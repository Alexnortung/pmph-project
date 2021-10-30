#include "../cub-1.8.0/cub/cub.cuh"
#include <stdio.h>
#include <stdlib.h>
//#include "kernels.cu.h"
#include "helper.cu.h"

//To pass on function name
typedef double (*functiontype)(uint32_t*, uint32_t*, const uint64_t);


// Validation - use sequential sorting
// Generate data -Done
// Timing - done
// kernel calling function - call kernels
// main function
// remember to free data

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

bool validate(uint32_t* output_array, uint32_t num_elems) {
    for(uint32_t i = 0; i < num_elems-1; i++){
        if (output_array[i] > output_array[i+1]){
            printf("INVALID RESULT for i:%d, (output[i-1]=%d > output[i]=%d)\n", i, output_array[i-1], output_array[i]);
            return false;
        }
    }
    return true;
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

//Execute the sorting algorithm on kernels
double sortByKernel(uint32_t* input_array
                  , uint32_t* output_array
                  , const uint64_t num_elem){

    uint32_t histogram_size = 1 << NUM_BITS; // 

    unsigned int block_size_make_hist = 256;
    uint32_t num_threads_make_hist = (num_elem + ELEM_PER_THREAD -1)/ELEM_PER_THREAD; // num threads for make_histogram
    unsigned int num_blocks_make_hist = (num_threads_make_hist + block_size - 1) / block_size;
    uint32_t all_histograms_size = num_threads_make_hist * histogram_size;
    
    uint32_t* histograms;
    uint32_t* histograms_trans; // transposed
    uint32_t* relative_offsets;

    cudaSucceeded(cudaMalloc((void**) &histograms, all_histograms_size * sizeof(uint32_t)));
    cudaSucceeded(cudaMalloc((void**) &histograms_trans, all_histograms_size * sizeof(uint32_t)));
    cudaSucceeded(cudaMalloc((void**) &relative_offsets, all_histograms_size * sizeof(uint32_t)));

    for (int i = 0; i < sizeof(uint32_t) * 8; i += NUM_BITS) {
        // TODO: call make_histogram
        int bit_offset = i;
        make_histogram<<< num_blocks_make_hist, block_size_make_hist >>>(input_array, num_elem, bit_offset, histograms, relative_offsets);

        // TODO: call transpose
        transpositions
        // TODO: call segmented scan
        // TODO: call transpose

        // TODO: scatter histogram
    }

    
    double elapsed = num_elem; // TODO: fix this


    //clean up
    cudaFree(count_array);
    return elapsed;
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


int main(int argc, char* argv[]) {
    // Validate the arguments
    if (argc != 1 + 1) {
        print_usage(argv[0]);
        exit(1);
    }

    // Check that the input array size is larger than 0
    const uint32_t num_elements = atoi(argv[1]);
    if (num_elements <= 0) {
        printf("Number of elements should be greater than 0\n");
        print_usage(argv[0]);
        exit(1);
    }

    //Create input_array with random values
    uint32_t* input_array = make_random_array(num_elements);
    //Allocate for output_array that will hold the results for CUB
    uint32_t* out_arr_CUB  = (uint32_t*) malloc(num_elements*sizeof(uint32_t));
    //Allocate for output_array that will hold the results for kernel execution
    uint32_t* out_arr_ker  = (uint32_t*) malloc(num_elements*sizeof(uint32_t));

    //Run the CUB implementation
    functiontype CUB_func = &sortRedByKeyCUB;
    double elapsedCUB = allocate_initiate(num_elements, input_array, out_arr_CUB, CUB_func);

    //Run the kernel implementation
    //functiontype ker_func = &sortByKernel;
    //double elapsedKer = allocate_initiate(num_elements, input_array, out_arr_ker, ker_func);

    
    bool success = validate(out_arr_CUB, num_elements);

    printf("CUB Sorting for N=%lu runs in: %.2f us, VALID: %d\n", num_elements, elapsedCUB, success);
    //printf("CUB Sorting for N=%lu runs in: %.2f us\n", num_elements, elapsedCUB);

    free(input_array); free(out_arr_CUB); free(out_arr_ker);

    return 0;//success ? 0 : 1;
}
