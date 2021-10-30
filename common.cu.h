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
