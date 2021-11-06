//#include "../cub-1.8.0/cub/cub.cuh"
#include <stdio.h>
#include <stdlib.h>
#include "kernels.cu.h"
#include "helper.cu.h"
#include "./hostSkel.cu.h"
#include "./generic-sort.cu.h"
#include "./CUBsort-impl.cu.h"

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

    initHwd();

    //Create input_array with random values
    uint32_t* input_array = make_rand_int_array(num_elements);
    //Allocate for output_array that will hold the results for kernel execution
    uint32_t* out_arr_ker  = (uint32_t*) malloc(num_elements*sizeof(uint32_t));
    uint32_t* out_arr_cub  = (uint32_t*) malloc(num_elements*sizeof(uint32_t));

    //Run the kernel implementation
    functiontype ker_func = &sortByKernel;
    double elapsedKer = allocate_initiate(num_elements, input_array, out_arr_ker, ker_func);

    functiontype cub_func = &sortRedByKeyCUB;
    double elapsedCUB = allocate_initiate(num_elements, input_array, out_arr_cub, cub_func);

    //printf("input: [");
    //for(int i = 0; i < num_elements; i++){
    //    printf("%d,", input_array[i]);
    //}
    //printf("]\n");
    //printf("output: [");
    //for(int i = 0; i < num_elements; i++){
    //    printf("%d,", out_arr_ker[i]);
    //}
    //printf("]\n");

    
    bool success = validate(out_arr_ker, num_elements);
    bool success2 = validate_arrays(out_arr_ker, out_arr_cub, num_elements);

    printf("CUB Sorting for N=%lu runs in: %.2f us, VALID: %d\n", num_elements, elapsedKer, success && success2);
    //printf("CUB Sorting for N=%lu runs in: %.2f us\n", num_elements, elapsedCUB);

    free(input_array);
    free(out_arr_ker);
    free(out_arr_cub);

    return 0;//success ? 0 : 1;
}
