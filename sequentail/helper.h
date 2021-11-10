/*************************
 ******* TRANSPOSE *******
 *************************/
template <class T>
void seq_matTransposeKer(T* A, T* B, int heightA, int widthA) {
    for(int i = 0; i < heightA; i++){
        for (int j = 0; j < widthA; j++){
            //B[j][i] = A[i][j];
            B[j * heightA + i] = A[i * widthA + j];
        }
    }
}


/************************
 **** MAKE_HISTOGRAM ****
 ************************/


template<class T>
void seq_make_histogram(T* input_array
                      , const uint64_t input_arr_size
                      , uint64_t bit_offset
                      , uint32_t* histograms
                      , uint32_t elements_per_histogram
                      , uint32_t num_histograms
) {
    const unsigned int histogram_size = 1 << NUM_BITS;
    uint64_t bitmask = (histogram_size - 1) << bit_offset;

    uint32_t all_histograms_size = histogram_size * num_histograms;
    printf("all_histograms_size is: %d\n", all_histograms_size);
    for (unsigned int i = 0; i < all_histograms_size; i++) {
        printf("setting %d to 0 in histograms\n", i);
        histograms[i] = 0;
    }

    for (unsigned int i = 0; i < input_arr_size; i++) {
        T item = input_array[i];
        uint32_t histogram_index = i / elements_per_histogram;
        uint32_t* histogram = &histograms[histogram_index * histogram_size];
        uint64_t tmp_bin = item & bitmask;
        uint64_t bin = tmp_bin >> bit_offset;
        printf("bin is: %d\n", bin);
        histogram[bin] += 1;
    }
}

template<class T>
void seq_sgmsum(uint32_t num_elements
              , T* input_array
              , T* output_array
              , char* flags
) {
    T accum = 0;
    for (uint32_t i = 0; i < num_elements; i++) {
        if (flags[i] == 1) {
            accum = 0;
        }
        accum += input_array[i];
        output_array[i] = accum;
    }
}

template<class T>
void seq_scansum(uint32_t num_elements
               , T* input_array
               , T* output_array
) {
    T accum = 0;
    for (uint32_t i = 0; i < num_elements; i++) {
        accum += input_array[i];
        output_array[i] = accum;
    }
}

template<class T>
void seq_scatter(uint32_t* histograms_multi_scanned
           , uint32_t* histograms_scanned
           , const uint64_t input_arr_size
           , const uint32_t elements_per_histogram
           , uint32_t* global_offsets
           , T* items
           , uint64_t bit_offset
           , T* output
) {
    unsigned int histogram_size = 1 << NUM_BITS;
    uint64_t bitmask = (histogram_size - 1) << bit_offset;
    for (unsigned int i = 0; i < input_arr_size; i++) {
        T item = items[i];
        unsigned int histogram_index = i / elements_per_histogram;
        unsigned int histogram_offset = histogram_size * histogram_index;
        uint64_t tmp_bin = item & bitmask;
        uint64_t bin = tmp_bin >> bit_offset;
        unsigned int before_offset;
        if (histogram_index <= 0) {
            before_offset = 0;
        } else {
            before_offset = histograms_multi_scanned[histogram_size * (histogram_index - 1) + bin];
        }
        unsigned int histogram_thread_id = i % elements_per_histogram;
        uint32_t global_offset;
        if (bin <= 0) {
            global_offset = 0;
        } else {
            global_offset = global_offsets[bin - 1];
        }
        uint32_t histogram_offset_index;
        if (bin <= 0) {
            histogram_offset_index = 0;
        } else {
            histogram_offset_index = histograms_scanned[histogram_offset + bin - 1];
        }
        uint32_t global_index = global_offset + before_offset + (histogram_thread_id - histogram_offset_index);
        output[global_index] = item;
    }
}

/*************************
 ******* partion 2 *******
 *************************/
template<class T> 
void seq_partition2(T* shmem_input, unsigned int max_elem, char bitoffset){
    uint16_t* tfs = (uint16_t*)malloc(max_elem * sizeof(uint16_t));
    uint16_t* ffs= (uint16_t*)malloc(max_elem * sizeof(uint16_t));
    T* output = (T*)malloc(max_elem * sizeof(T));
    for (unsigned int i = 0; i < max_elem; i++) {
        T array_elem = shmem_input[i];
        uint16_t p = TupAdd<T>::pred(array_elem, bitoffset);
        tfs[i] = p;
        ffs[i] = 1-p;
    }
    seq_scansum(max_elem, tfs, tfs);
    seq_scansum(max_elem, ffs, ffs);

    for (unsigned int i = 0; i < max_elem; i++) {
        T array_elem = shmem_input[i];
        uint16_t p = TupAdd<T>::pred(array_elem, bitoffset);
        uint16_t index;
        if (p) {
            index = tfs[i] - 1;
        } else {
            uint16_t length_bin_0 = tfs[max_elem - 1];
            index = length_bin_0 + ffs[i] - 1;
        }
        output[index] = array_elem;
    }
    for (unsigned int i = 0; i < max_elem; i++) {
        shmem_input[i] = output[i];
    }
    free(tfs);
    free(ffs);
    free(output);
}
