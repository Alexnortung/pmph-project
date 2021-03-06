#include "./sequentail/helper.h"

template<class T>
void sgmScanHistogramTrans(const uint32_t B
                    , const size_t num_histograms
                    , T* d_histo_trans_in
                    , T* d_histo_trans_out
) {
    const size_t histogram_size = 1 << NUM_BITS;
    uint32_t all_histograms_size = histogram_size * num_histograms;
    T*  d_tmp_vals;
    char* d_tmp_flag;
    char* d_inp_flag;
    cudaMalloc((void**)&d_tmp_vals, MAX_BLOCK*sizeof(T));
    cudaMalloc((void**)&d_tmp_flag, all_histograms_size*sizeof(char));
    cudaMalloc((void**)&d_inp_flag, all_histograms_size*sizeof(char));

    unsigned int make_flags_block_size = B;
    unsigned int make_flags_blocks = (all_histograms_size + make_flags_block_size - 1) / make_flags_block_size;

    make_histogram_trans_flags<<<make_flags_blocks, B>>>(num_histograms, d_inp_flag);

    sgmScanInc< Add<T> >( B, all_histograms_size, d_histo_trans_out, d_inp_flag, d_histo_trans_in, d_tmp_vals, d_tmp_flag );

    cudaFree(d_tmp_vals);
    cudaFree(d_tmp_flag);
    cudaFree(d_inp_flag);
}

template<class T>
void sgmScanHistogram(const uint32_t B
                    , const size_t num_histograms
                    , T* d_histo_in
                    , T* d_histo_out
) {
    const size_t histogram_size = 1 << NUM_BITS;
    uint32_t all_histograms_size = histogram_size * num_histograms;
    T*  d_tmp_vals;
    char* d_tmp_flag;
    char* d_inp_flag;
    cudaMalloc((void**)&d_tmp_vals, MAX_BLOCK*sizeof(T));
    cudaMalloc((void**)&d_tmp_flag, all_histograms_size*sizeof(char));
    cudaMalloc((void**)&d_inp_flag, all_histograms_size*sizeof(char));

    unsigned int make_flags_block_size = B;
    unsigned int make_flags_blocks = (all_histograms_size + make_flags_block_size - 1) / make_flags_block_size;

    make_histogram_flags<<<make_flags_blocks, B>>>(num_histograms, d_inp_flag);

    sgmScanInc< Add<T> >( B, all_histograms_size, d_histo_out, d_inp_flag, d_histo_in, d_tmp_vals, d_tmp_flag );

    cudaFree(d_tmp_vals);
    cudaFree(d_tmp_flag);
    cudaFree(d_inp_flag);
}

//Execute the sorting algorithm on kernels
template<class T>
double sortByKernel(T* input_array
                  , T* output_array
                  , const uint64_t num_elem){

    size_t histogram_size = 1 << NUM_BITS;

    unsigned int block_size_make_hist = 256;
    // get ceil of num_threads_make_hist / block_size_make_hist
    unsigned int num_blocks_make_hist = (num_elem + block_size_make_hist - 1) / block_size_make_hist;
    uint32_t num_histograms = num_blocks_make_hist;
    uint32_t all_histograms_size = num_histograms * histogram_size;
    const uint32_t num_elem_per_histo = block_size_make_hist;//(num_threads_make_hist + num_blocks_make_hist - 1) / num_blocks_make_hist; // elements per histogram
    uint32_t extern_shared_mem_make_hist = block_size_make_hist * sizeof(T) +
                                           block_size_make_hist * sizeof(uint16_t) +
                                           block_size_make_hist * sizeof(uint16_t);
    printf("all_histograms_size: %d\n", all_histograms_size);

    
    uint32_t* histograms; // array of total histograms
    uint32_t* histograms_scanned;
    uint32_t* histograms_trans; // transposed
    uint32_t* histograms_trans_scanned; // transposed, scanned
    uint32_t* global_offsets;
    uint32_t* d_tmp_scan;
    T* output_array_tmp;

    cudaSucceeded(cudaMalloc((void**) &histograms, all_histograms_size * sizeof(uint32_t)));
    cudaSucceeded(cudaMalloc((void**) &histograms_scanned, all_histograms_size * sizeof(uint32_t)));
    cudaSucceeded(cudaMalloc((void**) &histograms_trans, all_histograms_size * sizeof(uint32_t)));
    cudaSucceeded(cudaMalloc((void**) &histograms_trans_scanned, all_histograms_size * sizeof(uint32_t)));
    cudaSucceeded(cudaMalloc((void**) &global_offsets, histogram_size * sizeof(uint32_t)));
    cudaSucceeded(cudaMalloc((void**) &d_tmp_scan, MAX_BLOCK * sizeof(uint32_t)));
    cudaSucceeded(cudaMalloc((void**) &output_array_tmp, num_elem * sizeof(T)));

    unsigned int tile = 16;
    unsigned int dimx_transpose = (histogram_size + tile - 1) / tile;
    unsigned int dimy_transpose = (num_histograms + tile - 1) / tile;
    dim3 block_transpose(tile, tile, 1);
    dim3 grid_transpose (dimx_transpose, dimy_transpose, 1);
    dim3 block_transpose2(tile, tile, 1);
    dim3 grid_transpose2 (dimy_transpose, dimx_transpose, 1); // transposed grid of grid_transpose

    unsigned int block_size_sgm_scan = 256;

    unsigned int block_size_scatter = 256;
    unsigned int scatter_blocks = (num_elem + block_size_scatter - 1) / block_size_scatter;

    printf("starting radix\n");
    // timing
    double elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    printf("memcpy %d %d\n", num_elem, num_elem * sizeof(uint32_t));
    T* host_out_arr = (T*)malloc( num_elem * sizeof(T));
    cudaMemcpy(output_array, input_array, num_elem*sizeof(uint32_t), cudaMemcpyDeviceToDevice);

    for (uint64_t i = 0; i < sizeof(uint32_t) * 8; i += NUM_BITS) {
        // call make_histogram
        uint64_t bit_offset = i;
        make_histogram<<< num_blocks_make_hist, block_size_make_hist >>>(
            output_array,
            num_elem,
            bit_offset,
            histograms
        );
        sort_group<<< num_blocks_make_hist, block_size_make_hist, extern_shared_mem_make_hist >>>(output_array, num_elem, bit_offset);

        sgmScanHistogram(block_size_sgm_scan, num_histograms, histograms, histograms_scanned);

        // call transpose
        matTransposeKer<<< grid_transpose, block_transpose >>>(
            histograms,
            histograms_trans,
            num_histograms,
            histogram_size
        );

        sgmScanHistogramTrans(block_size_sgm_scan, num_histograms, histograms_trans, histograms_trans_scanned);

        // call transpose
        matTransposeKer<<< grid_transpose2, block_transpose2 >>>(
            histograms_trans_scanned,
            histograms,
            histogram_size,
            num_histograms
        );

        // Make global offsets
        uint32_t* last_histogram = &histograms[(num_histograms - 1) * histogram_size];
        scanInc< Add<uint32_t> > ( 64, histogram_size, global_offsets, last_histogram, d_tmp_scan );

        cudaMemcpy(output_array_tmp, output_array, num_elem * sizeof(T), cudaMemcpyDeviceToDevice);
        // scatter histogram
        histogram_scatter<<< scatter_blocks, block_size_scatter >>>(
            histograms, // now mulit_scanned
            histograms_scanned,
            num_elem,
            num_elem_per_histo,
            global_offsets,
            output_array_tmp,
            bit_offset,
            output_array
        );
    }
    cudaDeviceSynchronize();
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / ((double)GPU_RUNS);


    // clean up malloc'ed pointers.
    cudaFree(histograms);
    cudaFree(histograms_scanned);
    cudaFree(histograms_trans);
    cudaFree(histograms_trans_scanned);
    cudaFree(global_offsets);
    cudaFree(d_tmp_scan);
    return elapsed;
}
