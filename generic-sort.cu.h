template<class T>
void sgmScanHistogramTrans(const uint32_t B
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

    make_histogram_trans_flags<<<make_flags_blocks, B>>>(num_histograms, d_inp_flag);
    //printf("Made histogram flags\n");

    sgmScanInc< Add<T> >( B, N, d_histo_trans_out, d_inp_flag, d_histo_trans_in, d_tmp_vals, d_tmp_flag );

    cudaFree(d_tmp_vals);
    cudaFree(d_tmp_flag);
    cudaFree(d_inp_flag);
}

template<class T>
void sgmScanHistogram(const uint32_t B
                    , const size_t N
                    , T* d_histo_in
                    , T* d_histo_out
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

    make_histogram_flags<<<make_flags_blocks, B>>>(num_histograms, d_inp_flag);
    //printf("Made histogram flags\n");

    sgmScanInc< Add<T> >( B, N, d_histo_out, d_inp_flag, d_histo_in, d_tmp_vals, d_tmp_flag );

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

    unsigned int block_size_make_hist = 32;
    // get ceil of num_elem / ELEM_PER_THREAD_MAKE_HIST // total threads //
    //uint32_t num_threads_make_hist = (num_elem + ELEM_PER_THREAD_MAKE_HIST -1)/ELEM_PER_THREAD_MAKE_HIST; // num threads for make_histogram
    uint32_t num_threads_make_hist = num_elem; // num threads for make_histogram
    // get ceil of num_threads_make_hist / block_size_make_hist
    unsigned int num_blocks_make_hist = (num_threads_make_hist + block_size_make_hist - 1) / block_size_make_hist;
    uint32_t num_histograms = num_blocks_make_hist;
    uint32_t all_histograms_size = num_histograms * histogram_size;
    const uint32_t num_elem_per_histo = (num_threads_make_hist + num_blocks_make_hist - 1) / num_blocks_make_hist; // elements per histogram
    printf("num_blocks_make_hist: %d\n", num_blocks_make_hist);
    printf("num_elem_per_histo: %d\n", num_elem_per_histo);

    
    uint32_t* histograms; // array of total histograms
    uint16_t* indicies;
    uint32_t* histograms_scanned;
    uint32_t* histograms_trans; // transposed
    uint32_t* histograms_trans_scanned; // transposed, scanned
    uint32_t* global_offsets;
    uint32_t* d_tmp_scan;

    cudaSucceeded(cudaMalloc((void**) &histograms, all_histograms_size * sizeof(uint32_t)));
    cudaSucceeded(cudaMalloc((void**) &indicies, num_elem * sizeof(uint64_t)));
    cudaSucceeded(cudaMalloc((void**) &histograms_scanned, all_histograms_size * sizeof(uint32_t)));
    cudaSucceeded(cudaMalloc((void**) &histograms_trans, all_histograms_size * sizeof(uint32_t)));
    cudaSucceeded(cudaMalloc((void**) &histograms_trans_scanned, all_histograms_size * sizeof(uint32_t)));
    cudaSucceeded(cudaMalloc((void**) &global_offsets, histogram_size * sizeof(uint32_t)));
    cudaSucceeded(cudaMalloc((void**) &d_tmp_scan, MAX_BLOCK * sizeof(uint32_t)));

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

    // timing
    double elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    cudaMemcpy(output_array, input_array, num_elem*sizeof(uint32_t), cudaMemcpyDeviceToDevice);

    for (uint64_t i = 0; i < sizeof(uint32_t) * 8; i += NUM_BITS) {
    //for (uint64_t i = 0; i < 1; i += NUM_BITS) {
        // call make_histogram
        uint64_t bit_offset = i;
        //printf("Making histogram\n");
        make_histogram<<< num_blocks_make_hist, block_size_make_hist, block_size_make_hist * sizeof(T) + block_size_make_hist * sizeof(uint16_t) + block_size_make_hist * sizeof(uint16_t) >>>(
            output_array,
            num_elem,
            bit_offset,
            histograms,
            indicies
        );
        //uint32_t* histograms_cpu = (uint32_t*)malloc(all_histograms_size * sizeof(uint32_t));
        //cudaMemcpy(histograms_cpu, histograms, all_histograms_size*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        //printf("hist [");
        //for(int i2 = 0; i2 < all_histograms_size; i2++){
        //    //printf("%d,", histograms_cpu[i2]);
        //}
        //printf("]\n");
        //free(histograms_cpu);

        sgmScanHistogram(block_size_sgm_scan, all_histograms_size, histograms, histograms_scanned);

        // call transpose
        //printf("transposing histogram\n");
        matTransposeKer<<< grid_transpose, block_transpose >>>(
            histograms,
            histograms_trans,
            histogram_size,
            num_histograms
        );
        //uint32_t* histograms_trans_cpu = (uint32_t*)malloc(all_histograms_size * sizeof(uint32_t));
        //cudaMemcpy(histograms_trans_cpu, histograms_trans, all_histograms_size*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        //
        ////printf("all_histograms_size: %d\n", all_histograms_size);
        //printf("hist trans [");
        //for(int i2 = 0; i2 < all_histograms_size; i2++){
        //    printf("%d,", histograms_trans_cpu[i2]);
        //}
        //printf("]\n");
        //free(histograms_trans_cpu);
        // call segmented scan
        //printf("Scanning transposed histogram\n");
        sgmScanHistogramTrans(block_size_sgm_scan, all_histograms_size, histograms_trans, histograms_trans_scanned);
        // call transpose
        //printf("transposing histogram\n");
        matTransposeKer<<< grid_transpose2, block_transpose2 >>>(histograms_trans_scanned, histograms, num_histograms, histogram_size);
        //uint32_t* histograms_cpu = (uint32_t*)malloc(all_histograms_size * sizeof(uint32_t));
        //cudaMemcpy(histograms_cpu, histograms, all_histograms_size*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        //printf("multi hist [");
        //for(int i2 = 0; i2 < all_histograms_size; i2++){
        //    printf("%d,", histograms_cpu[i2]);
        //}
        //printf("]\n");
        //free(histograms_cpu);


        uint32_t* last_histogram = &histograms[(num_histograms - 1) * histogram_size];
        scanInc< Add<uint32_t> > ( 64, histogram_size, global_offsets, last_histogram, d_tmp_scan );


        //uint32_t* global_offsets_cpu = (uint32_t*)malloc(histogram_size * sizeof(uint32_t));
        //cudaMemcpy(global_offsets_cpu, global_offsets, histogram_size*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        //printf("all_histograms_size: %d\n", all_histograms_size);
        //printf("glb_offs [");
        //for(int i2 = 0; i2 < histogram_size; i2++){
        //    printf("%d,", global_offsets_cpu[i2]);
        //}
        //printf("]\n");
        //free(global_offsets_cpu);

        // scatter histogram
        histogram_scatter<<< scatter_blocks, block_size_scatter >>>(
            histograms, // now mulit_scanned
            histograms_scanned,
            num_elem,
            num_elem_per_histo,
            global_offsets,
            output_array,
            bit_offset,
            output_array
        );
        //uint32_t* output_array_cpu = (uint32_t*)malloc(num_elem * sizeof(uint32_t));
        //cudaMemcpy(output_array_cpu, output_array, num_elem*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        //printf("output iter: %d [", i);
        //for(int i2 = 0; i2 < num_elem; i2++){
        //    printf("%d,", output_array_cpu[i2]);
        //}
        //printf("]\n");
        //free(output_array_cpu);

    }
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / ((double)GPU_RUNS);


    //clean up
    cudaFree(histograms);
    cudaFree(histograms_scanned);
    cudaFree(histograms_trans);
    cudaFree(histograms_trans_scanned);
    cudaFree(global_offsets);
    cudaFree(d_tmp_scan);
    return elapsed;
}
