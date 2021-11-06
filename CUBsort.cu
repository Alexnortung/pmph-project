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
