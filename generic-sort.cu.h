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
    //char* h_inp_flag = (char*)malloc(all_histograms_size * sizeof(char));
    cudaMalloc((void**)&d_tmp_vals, MAX_BLOCK*sizeof(T));
    cudaMalloc((void**)&d_tmp_flag, all_histograms_size*sizeof(char));
    cudaMalloc((void**)&d_inp_flag, all_histograms_size*sizeof(char));

    unsigned int make_flags_block_size = B;
    unsigned int make_flags_blocks = (all_histograms_size + make_flags_block_size - 1) / make_flags_block_size;

    make_histogram_trans_flags<<<make_flags_blocks, B>>>(num_histograms, d_inp_flag);
    //cudaMemcpy(h_inp_flag, d_inp_flag, all_histograms_size * sizeof(char), cudaMemcpyDeviceToHost);

    sgmScanInc< Add<T> >( B, all_histograms_size, d_histo_trans_out, d_inp_flag, d_histo_trans_in, d_tmp_vals, d_tmp_flag );
    //seq_sgmsum(all_histograms_size, d_histo_trans_in, d_histo_trans_out, h_inp_flag);

    cudaFree(d_tmp_vals);
    cudaFree(d_tmp_flag);
    cudaFree(d_inp_flag);
    //free(h_inp_flag);
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
    //char* h_inp_flag = (char*)malloc(all_histograms_size * sizeof(char));
    cudaMalloc((void**)&d_tmp_vals, MAX_BLOCK*sizeof(T));
    cudaMalloc((void**)&d_tmp_flag, all_histograms_size*sizeof(char));
    cudaMalloc((void**)&d_inp_flag, all_histograms_size*sizeof(char));

    unsigned int make_flags_block_size = B;
    unsigned int make_flags_blocks = (all_histograms_size + make_flags_block_size - 1) / make_flags_block_size;

    make_histogram_flags<<<make_flags_blocks, B>>>(num_histograms, d_inp_flag);
    //cudaMemcpy(h_inp_flag, d_inp_flag, all_histograms_size * sizeof(char), cudaMemcpyDeviceToHost);

    sgmScanInc< Add<T> >( B, all_histograms_size, d_histo_out, d_inp_flag, d_histo_in, d_tmp_vals, d_tmp_flag );
    //seq_sgmsum(all_histograms_size, d_histo_in, d_histo_out, h_inp_flag);

    cudaFree(d_tmp_vals);
    cudaFree(d_tmp_flag);
    cudaFree(d_inp_flag);
    //free(h_inp_flag);
}

//Execute the sorting algorithm on kernels
template<class T>
double sortByKernel(T* input_array
                  , T* output_array
                  , const uint64_t num_elem){

    size_t histogram_size = 1 << NUM_BITS;

    unsigned int block_size_make_hist = 256;
    // get ceil of num_elem / ELEM_PER_THREAD_MAKE_HIST // total threads //
    //uint32_t num_threads_make_hist = num_elem; // num threads for make_histogram
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
    uint32_t* histograms_cpu; // array of total histograms
    uint32_t* histograms_scanned;
    uint32_t* histograms_scanned_cpu;
    uint32_t* histograms_trans; // transposed
    uint32_t* histograms_trans_cpu; // transposed
    uint32_t* histograms_trans_scanned; // transposed, scanned
    uint32_t* histograms_trans_scanned_cpu; // transposed, scanned
    uint32_t* global_offsets_cpu;
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

    histograms_cpu = (uint32_t*)malloc( all_histograms_size * sizeof(uint32_t));
    histograms_scanned_cpu = (uint32_t*)malloc(all_histograms_size * sizeof(uint32_t));
    histograms_trans_cpu = (uint32_t*)malloc(all_histograms_size * sizeof(uint32_t));
    histograms_trans_scanned_cpu = (uint32_t*)malloc(all_histograms_size * sizeof(uint32_t));
    global_offsets_cpu = (uint32_t*)malloc(histogram_size * sizeof(uint32_t));

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

    //cudaMemcpy(output_array, input_array, num_elem*sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    printf("memcpy %d %d\n", num_elem, num_elem * sizeof(uint32_t));
    T* host_out_arr = (T*)malloc( num_elem * sizeof(T));
    //memcpy(output_array, input_array, num_elem * sizeof(uint32_t));
    cudaMemcpy(output_array, input_array, num_elem*sizeof(uint32_t), cudaMemcpyDeviceToDevice);

    for (uint64_t i = 0; i < sizeof(uint32_t) * 8; i += NUM_BITS) {
        // call make_histogram
        uint64_t bit_offset = i;
        printf("making histos at bit_offset: %d\n", bit_offset);
        //make_histogram<<< num_blocks_make_hist, block_size_make_hist, extern_shared_mem_make_hist >>>(
        //    output_array,
        //    num_elem,
        //    bit_offset,
        //    histograms
        //);
        //cudaMemcpy(host_out_arr, output_array, num_elem*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        //cudaMemcpy(histograms_cpu, histograms, all_histograms_size*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(histograms_cpu, histograms, all_histograms_size*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_out_arr, output_array, num_elem*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        seq_make_histogram(
            host_out_arr,
            num_elem,
            bit_offset,
            histograms_cpu,
            num_elem_per_histo,
            num_histograms
        );
        cudaMemcpy(histograms, histograms_cpu, all_histograms_size*sizeof(uint32_t), cudaMemcpyHostToDevice);
        printf("partitioning\n");
        // TODO: for loop for NUM_BITS
        //for (unsigned int i = 0; i < num_elem; i += num_elem_per_histo) {
        //    T* current_pointer = &host_out_arr[i];
        //    unsigned int max_elem = min(i + num_elem_per_histo, (unsigned int)num_elem) - i;
        //    seq_partition2(current_pointer, max_elem, bit_offset);
        //}
        //cudaMemcpy(output_array, host_out_arr, num_elem*sizeof(uint32_t), cudaMemcpyHostToDevice);
        sort_group<<< num_blocks_make_hist, block_size_make_hist, extern_shared_mem_make_hist >>>(output_array, num_elem, bit_offset);
        cudaMemcpy(host_out_arr, output_array, num_elem*sizeof(uint32_t), cudaMemcpyDeviceToHost);

        printf("sgm scanning\n");
        sgmScanHistogram(block_size_sgm_scan, num_histograms, histograms, histograms_scanned);
        cudaMemcpy(histograms_scanned_cpu, histograms_scanned, all_histograms_size*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        

        // call transpose
        matTransposeKer<<< grid_transpose, block_transpose >>>(
            histograms,
            histograms_trans,
            num_histograms,
            histogram_size
        );
        printf("transposing 1\n");
        //seq_matTransposeKer(
        //    histograms_cpu,
        //    histograms_trans,
        //    num_histograms,
        //    histogram_size
        //);

        printf("multi scan\n");
        sgmScanHistogramTrans(block_size_sgm_scan, num_histograms, histograms_trans, histograms_trans_scanned);

        // call transpose
        printf("transposing 2\n");
        matTransposeKer<<< grid_transpose2, block_transpose2 >>>(
            histograms_trans_scanned,
            histograms,
            histogram_size,
            num_histograms
        );
        cudaMemcpy(histograms_cpu, histograms, all_histograms_size*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        //seq_matTransposeKer(
        //    histograms_trans_scanned,
        //    histograms_cpu,
        //    histogram_size,
        //    num_histograms
        //);

        printf("global offsets\n");
        // Make global offsets
        //uint32_t* last_histogram_cpu = &histograms_cpu[(num_histograms - 1) * histogram_size];
        uint32_t* last_histogram = &histograms[(num_histograms - 1) * histogram_size];
        scanInc< Add<uint32_t> > ( 64, histogram_size, global_offsets, last_histogram, d_tmp_scan );
        //seq_scansum(histogram_size, last_histogram_cpu, global_offsets);
        cudaMemcpy(global_offsets_cpu, global_offsets, histogram_size*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(output_array_tmp, output_array, num_elem * sizeof(T), cudaMemcpyDeviceToDevice);

        printf("scattering\n");
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
        //cudaDeviceSynchronize();
        //T* tmp_output_array = (T*)malloc(num_elem * sizeof(T));
        //memcpy(tmp_output_array, host_out_arr, num_elem * sizeof(T));
        //seq_scatter(
        //    histograms_cpu, // now mulit_scanned
        //    histograms_scanned_cpu,
        //    num_elem,
        //    num_elem_per_histo,
        //    global_offsets_cpu,
        //    tmp_output_array,
        //    bit_offset,
        //    host_out_arr
        //);
        //printf("freeing tmp\n");
        //free(tmp_output_array);
        //printf("copying output to device\n");
        //cudaMemcpy(output_array, host_out_arr, num_elem * sizeof(T), cudaMemcpyHostToDevice);
        //printArray(host_out_arr, num_elem);
        printf("Done with iteration %d\n", i);
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
    free(histograms_cpu);
    free(histograms_scanned_cpu);
    free(histograms_trans_cpu);
    free(histograms_trans_scanned_cpu);
    free(global_offsets_cpu);
    return elapsed;
}
