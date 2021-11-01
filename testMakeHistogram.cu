#include "cub-1.8.0/cub/cub.cuh"
#include <stdio.h>
#include <stdlib.h>
#include "helper.cu.h"
#include "common.cu.h"
#include "kernels.cu.h"

template<class T, class U>
void printArray(T* array, U array_size) {
    printf("[ ");
    for (U i = 0; i < array_size; i++) {
        printf("%d, ", array[i]);
    }
    printf("]\n");
}

//Execute the sorting algorithm on kernels
double make_histogram_kernel(uint32_t* input_array
                  , uint32_t* output_array
                  , const uint64_t num_elem){

    uint32_t histogram_size = 1 << NUM_BITS; // 

    unsigned int block_size_make_hist = 256;
    uint32_t num_threads_make_hist = (num_elem + ELEM_PER_THREAD -1)/ELEM_PER_THREAD; // num threads for make_histogram
    unsigned int num_blocks_make_hist = (num_threads_make_hist + block_size_make_hist - 1) / block_size_make_hist;
    uint32_t all_histograms_size = num_threads_make_hist * histogram_size;
    
    uint32_t* histograms;
    uint32_t* relative_offsets;

    cudaSucceeded(cudaMalloc((void**) &histograms, all_histograms_size * sizeof(uint32_t)));
    cudaSucceeded(cudaMalloc((void**) &relative_offsets, all_histograms_size * sizeof(uint32_t)));

    uint64_t bit_offset = 0;
    make_histogram<<< num_blocks_make_hist, block_size_make_hist >>>(input_array, num_elem, bit_offset, histograms, relative_offsets);

    uint32_t* histograms_cpu = (uint32_t*)malloc(all_histograms_size * sizeof(uint32_t));
    cudaMemcpy(histograms_cpu, histograms, all_histograms_size*sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printArray(histograms_cpu, all_histograms_size);

    //clean up
    cudaFree(histograms);
    cudaFree(relative_offsets);
    free(histograms_cpu);
    return 0.0;
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
    //Allocate for output_array that will hold the results for kernel execution
    uint32_t* out_arr_ker  = (uint32_t*) malloc(num_elements*sizeof(uint32_t));

    //Run the kernel implementation
    functiontype ker_func = &make_histogram_kernel;
    double elapsedKer = allocate_initiate(num_elements, input_array, out_arr_ker, ker_func);
    printArray(input_array, num_elements);

    free(input_array);
    free(out_arr_ker);

    return 0;//success ? 0 : 1;
}
