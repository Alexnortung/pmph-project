//#include "../cub-1.8.0/cub/cub.cuh"
#include <stdio.h>
#include <stdlib.h>
#include "kernels.cu.h"
#include "helper.cu.h"
#include "./hostSkel.cu.h"


template<class T>
void sgmScanHistogram(const uint32_t B
                    , const size_t N
                    , T* d_histo_trans_in
                    , T* d_histo_trans_out
) {
    const size_t histogram_size = 1 << NUM_BITS;
    uint32_t num_histograms = N / histogram_size;
    uint32_t all_histograms_size = histogram_size * num_histograms;
    T*  d_tmp_vals;
    char* d_tmp_flag;
    char* d_inp_flag;
    cudaMalloc((void**)&d_tmp_vals, MAX_BLOCK*sizeof(T));
    cudaMalloc((void**)&d_tmp_flag, all_histograms_size*sizeof(char));
    cudaMalloc((void**)&d_inp_flag, N * sizeof(char));

    unsigned int make_flags_block_size = B;
    unsigned int make_flags_blocks = (N + make_flags_block_size - 1) / make_flags_block_size;

    make_histogram_flags<<<B, make_flags_blocks>>>(num_histograms, d_inp_flag);

    sgmScanInc< Add<T> >( B, N, d_histo_trans_out, d_inp_flag, d_histo_trans_in, d_tmp_vals, d_tmp_flag );

    cudaFree(d_tmp_vals);
    cudaFree(d_tmp_flag);
    cudaFree(d_inp_flag);
}

//Execute the sorting algorithm on kernels
double sortByKernel(uint32_t* input_array
                  , uint32_t* output_array
                  , const uint64_t num_elem){

    size_t histogram_size = 1 << NUM_BITS; // 

    unsigned int block_size_make_hist = 256;
    uint32_t num_threads_make_hist = (num_elem + ELEM_PER_THREAD -1)/ELEM_PER_THREAD; // num threads for make_histogram
    unsigned int num_blocks_make_hist = (num_threads_make_hist + block_size_make_hist - 1) / block_size_make_hist;
    uint32_t num_histograms = num_blocks_make_hist;
    uint32_t all_histograms_size = num_threads_make_hist * histogram_size;
    uint32_t num_elem_per_histo = num_threads_make_hist * ELEM_PER_THREAD;
    
    uint32_t* histograms;
    uint32_t* histograms_trans; // transposed
    uint32_t* histograms_trans_scanned; // transposed, scanned
    uint32_t* relative_offsets;
    uint32_t* global_offsets;
    uint32_t* d_tmp_scan;

    cudaSucceeded(cudaMalloc((void**) &histograms, all_histograms_size * sizeof(uint32_t)));
    cudaSucceeded(cudaMalloc((void**) &histograms_trans, all_histograms_size * sizeof(uint32_t)));
    cudaSucceeded(cudaMalloc((void**) &histograms_trans_scanned, all_histograms_size * sizeof(uint32_t)));
    cudaSucceeded(cudaMalloc((void**) &relative_offsets, all_histograms_size * sizeof(uint32_t)));
    cudaSucceeded(cudaMalloc((void**) &global_offsets, histogram_size * sizeof(uint32_t)));
    cudaSucceeded(cudaMalloc((void**) &d_tmp_scan, MAX_BLOCK * sizeof(uint32_t)));

    unsigned int tile = 16;
    unsigned int dimx_transpose = (histogram_size + tile - 1) / tile;
    unsigned int dimy_transpose = (num_histograms + tile - 1) / tile;
    dim3 block_transpose(tile, tile, 1);
    dim3 grid_transpose (dimx_transpose, dimy_transpose, 1);
    dim3 block_transpose2(tile, tile, 1);
    dim3 grid_transpose2 (dimy_transpose, dimx_transpose, 1);

    unsigned int block_size_sgm_scan = 256;

    unsigned int block_size_scatter = 256;
    unsigned int scatter_blocks = (num_elem + block_size_scatter - 1) / block_size_scatter;

    cudaMemcpy(input_array, output_array, num_elem*sizeof(uint32_t), cudaMemcpyDeviceToDevice);

    for (int i = 0; i < sizeof(uint32_t) * 8; i += NUM_BITS) {
        // TODO: call make_histogram
        int bit_offset = i;
        make_histogram<<< num_blocks_make_hist, block_size_make_hist >>>(output_array, num_elem, bit_offset, histograms, relative_offsets);

        // TODO: call transpose
        matTransposeKer<<< grid_transpose, block_transpose >>>(histograms, histograms_trans, histogram_size, num_histograms);
        // TODO: call segmented scan
        sgmScanHistogram(block_size_sgm_scan, all_histograms_size, histograms_trans, histograms_trans_scanned);
        // TODO: call transpose
        matTransposeKer<<< grid_transpose2, block_transpose2 >>>(histograms_trans_scanned, histograms, num_histograms, histogram_size);

        uint32_t* last_histogram = &histograms[(num_histograms - 1) * histogram_size];
        scanInc< Add<uint32_t> > ( 64, histogram_size, global_offsets, last_histogram, d_tmp_scan );

        // TODO: scatter histogram
        histogram_scatter<<< scatter_blocks, block_size_scatter >>>(histograms, num_elem, num_elem_per_histo, global_offsets, output_array, bit_offset, relative_offsets, output_array);
    }

    
    double elapsed = num_elem; // TODO: fix this


    //clean up
    cudaFree(histograms);
    cudaFree(histograms_trans);
    cudaFree(histograms_trans_scanned);
    cudaFree(relative_offsets);
    cudaFree(global_offsets);
    cudaFree(d_tmp_scan);
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
    uint32_t* input_array = make_random_array(num_elements);
    //Allocate for output_array that will hold the results for CUB
    uint32_t* out_arr_CUB  = (uint32_t*) malloc(num_elements*sizeof(uint32_t));
    //Allocate for output_array that will hold the results for kernel execution
    uint32_t* out_arr_ker  = (uint32_t*) malloc(num_elements*sizeof(uint32_t));

    //Run the CUB implementation
    functiontype CUB_func = &sortRedByKeyCUB;
    //double elapsedCUB = allocate_initiate(num_elements, input_array, out_arr_CUB, CUB_func);

    //Run the kernel implementation
    functiontype ker_func = &sortByKernel;
    double elapsedKer = allocate_initiate(num_elements, input_array, out_arr_ker, ker_func);

    
    //bool success = validate(out_arr_CUB, num_elements);

    //printf("CUB Sorting for N=%lu runs in: %.2f us, VALID: %d\n", num_elements, elapsedCUB, success);
    //printf("CUB Sorting for N=%lu runs in: %.2f us\n", num_elements, elapsedCUB);

    free(input_array); free(out_arr_CUB); free(out_arr_ker);

    return 0;//success ? 0 : 1;
}
