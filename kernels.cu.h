#include <algorithm>
#include <stdio.h>
#include "helper.cu.h"
#include "helperKernel.cu.h"

template<class T>
__global__ void make_histogram(T* input_array
                             , const uint64_t input_arr_size
                             , uint64_t bit_offset
                             , uint32_t* histograms
) {
    const unsigned int histogram_size = 1 << NUM_BITS;
    const uint32_t B = (blockDim.x + ELEM_PER_THREAD_MAKE_HIST - 1) / ELEM_PER_THREAD_MAKE_HIST;
    const uint32_t B_all = blockDim.x;
    uint64_t histogram_index = blockIdx.x;
    __shared__ uint32_t histogram[histogram_size];
    extern __shared__ T es[]; // external shared
    T* elem_input = es; // start of external shared should be used in partition
    uint16_t* tfs = (uint16_t*)&elem_input[B_all]; // the other part of the share array
    uint16_t* ffs = (uint16_t*)&tfs[B_all]; // the other part of the share array

    uint64_t bitmask = (histogram_size - 1) << bit_offset;
    for (int idx = threadIdx.x; idx < histogram_size; idx += B_all) {
        histogram[idx] = 0;
    }

    int i = 0;
    // each thread loops over ELEM_PER_THREAD elements in the block with coalesced access
    uint64_t block_offset = ELEM_PER_THREAD_MAKE_HIST * B * blockIdx.x;
    if (bit_offset == 0 && threadIdx.x == 0) {
        //printf("block_offset: %d\n",block_offset );
    }
    for (int idx = block_offset + threadIdx.x; idx < min(block_offset + ELEM_PER_THREAD_MAKE_HIST * B, input_arr_size) && threadIdx.x < B; idx += B) {
        T item = input_array[idx];
        elem_input[idx - block_offset] = item;
        uint64_t tmp_bin = item & bitmask;
        uint64_t bin = tmp_bin >> bit_offset;
        // increment the value in the histogram and save the relative_offset
        uint32_t relative_offset = atomicAdd(&histogram[bin], 1);
        //printf("relative_off: %d\n", relative_offset);
        i++;
    }
    // Naive:
    //T item = input_array[block_offset + threadIdx.x];
    //elem_input[threadIdx.x] = item;
    //uint64_t tmp_bin = item & bitmask;
    //uint64_t bin = tmp_bin >> bit_offset;
    //uint32_t relative_offset = atomicAdd(&histogram[bin], 1);

    
    __syncthreads();

    for (int idx = threadIdx.x; idx < histogram_size; idx += B_all) {
        histograms[histogram_index * histogram_size + idx] = histogram[idx];
    }

    unsigned partiton_max_elem = B_all;
    if (B_all * (blockIdx.x + 1) > input_arr_size) {
        partiton_max_elem = input_arr_size % B_all;
    }

    for(char j = 0; j < NUM_BITS; j++){
        char new_bit_offset = bit_offset + j;
        partition2(elem_input, tfs, ffs, partiton_max_elem, new_bit_offset);
        __syncthreads();
    }
    
    for (uint32_t idx = threadIdx.x; block_offset + idx < min(input_arr_size, block_offset + B_all); idx += B_all) {
        input_array[block_offset + idx] = elem_input[idx]; 
    }
}

template<class T>
__global__ void make_histogram_flags(T num_histograms, char* flags) {
    uint32_t histogram_size = 1 << NUM_BITS;
    unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if (gid >= num_histograms * histogram_size) {
        return;
    }
    flags[gid] = (gid & (histogram_size - 1)) == 0;
}
// [1, 0 , 0 , 0, 1, 0, 0, 0, ...]

template<class T>
__global__ void make_histogram_trans_flags(T num_histograms, char* flags) {
    uint32_t histogram_size = 1 << NUM_BITS;
    unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if (gid >= num_histograms * histogram_size) {
        return;
    }
    flags[gid] = gid % num_histograms == 0; // TODO: maybe change modulo (%) ?
}

template<class T>
__global__ void histogram_scatter(uint32_t* histograms_multi_scanned
                                , uint32_t* histograms_scanned
                                , const uint64_t input_arr_size
                                , const uint32_t elements_per_histogram
                                , uint32_t* global_offsets
                                , T* items
                                , uint64_t bit_offset
                                , T* output
) {
    unsigned int histogram_size = 1 << NUM_BITS;
    uint32_t gid = blockIdx.x*blockDim.x + threadIdx.x;
    uint32_t global_index;
    T item;
    if (gid < input_arr_size) {
        item = items[gid];
        uint64_t bitmask = (histogram_size - 1) << bit_offset;
        unsigned int histogram_index = gid / elements_per_histogram;
        unsigned int histogram_offset = histogram_size * histogram_index;
        uint64_t tmp_bin = item & bitmask;
        uint64_t bin = tmp_bin >> bit_offset;
        unsigned int before_offset;
        if (histogram_index <= 0) {
            before_offset = 0;
        } else {
            before_offset = histograms_multi_scanned[histogram_size * (histogram_index - 1) + bin];
        }
        unsigned int histogram_thread_id = gid % elements_per_histogram;
        uint32_t global_offset = global_offsets[bin];
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
        //histogram_offset_index = histograms_scanned[histogram_offset + local_offset_index - 1];
        //uint32_t elem_histogram_offset = 
        global_index = global_offset + before_offset + (histogram_thread_id - histogram_offset_index);
        if (bit_offset == 0 && gid < 1000) {
            //printf("(glb_ind: %d, glb_off: %d, beoff: %d, bin: %d, gid: %d, hsize: %d, hind: %d, hoff: %d, hoind: %d, his_tid: %d)\n", global_index, global_offset, (uint32_t)before_offset, (uint32_t)bin, gid, histogram_size, histogram_index, histogram_offset, histogram_offset_index, histogram_thread_id);
        }
    }
    __syncthreads();
    if (gid < input_arr_size) {
        output[global_index] = item;
    }
    __syncthreads();
}
