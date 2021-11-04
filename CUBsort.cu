#include "helper.cu.h"

// Execute CUB-library radix sort on input_array.
// This function is copied from the example code in CUBcode
template<class T>
double sortRedByKeyCUB( T* input_array
                      , T* output_array
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
    //uint32_t* input_array = make_rand_int_array(num_elements);
    float* input_array = make_rand_fl_array(num_elements);
    //uint32_t* input_array = make_rand_int_array(num_elements);
    
    printf("CUB Sorting for N=%lu\n", num_elements);
    int num_test = 10;
    for(int i = 0; i < num_test; i++){
        //Allocate for output_array that will hold the results for CUB
        //uint32_t* out_arr_CUB  = (uint32_t*) malloc(num_elements*sizeof(uint32_t));
        float* out_arr_CUB  = (float*) malloc(num_elements*sizeof(float));

        //Run the CUB implementation
        typedef double (*functiontypeFloat)(float*, float*, const uint64_t);
        functiontypeFloat CUB_func = &sortRedByKeyCUB;
        double elapsedCUB = allocate_initiate(num_elements, input_array, out_arr_CUB, CUB_func);

        //bool success = validate(input_array, num_elements);
        //bool success = validate(out_f_CUB, num_elements);

        printf("[");
        for(int i = 0; i < num_elements; i++){
            printf("%.2f,", out_arr_CUB[i]);
        }
        printf("]");
        //printf("CUB Sorting for N=%lu runs in: %.2f us, VALID: %d\n", num_elements, elapsedCUB, success);
        //printf("CUB Sorting for N=%lu runs in: %.2f us\n", num_elements, elapsedCUB);
        printf("%.2f\n", elapsedCUB);
        
        free(out_arr_CUB);
    }
    free(input_array); 

    return 0;//success ? 0 : 1;
}
