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
    const uint32_t B = blockDim.x;
    uint64_t block_offset = B * blockIdx.x;
    const uint32_t gid = block_offset + threadIdx.x;
    uint64_t histogram_index = blockIdx.x;
    __shared__ uint32_t histogram[histogram_size];

    uint64_t bitmask = (histogram_size - 1) << bit_offset;

    for (int idx = threadIdx.x; idx < histogram_size; idx += B) {
        histogram[idx] = 0;
    }
    __syncthreads();
    if (gid < input_arr_size) {
        T item = input_array[gid];
        //elem_input[threadIdx.x] = item;
        uint64_t tmp_bin = item & bitmask;
        uint64_t bin = tmp_bin >> bit_offset;
        // increment the value in the histogram and save the relative_offset
        atomicAdd(&histogram[bin], 1);
        //if (bit_offset == 0 && histogram_index == 0) {
        //    printf("bin: %d", bin);
        //}
    }

    __syncthreads();

    for (int idx = threadIdx.x; idx < histogram_size; idx += B) {
        uint32_t insert_index = histogram_index * histogram_size + idx;
        histograms[insert_index] = histogram[idx];
    }
}

template<class T>
__global__ void sort_group(T* input_array
                         , const uint64_t input_arr_size
                         , uint64_t bit_offset
) {
    uint32_t tid = threadIdx.x;
    uint64_t block_offset = blockDim.x * blockIdx.x;
    uint32_t gid = block_offset + tid;
    extern __shared__ T es[]; // external shared
    T* shmem_inp = es; // start of external shared should be used in partition
    uint16_t* tfs = (uint16_t*)&shmem_inp[blockDim.x]; // the other part of the share array
    uint16_t* ffs = (uint16_t*)&tfs[blockDim.x]; // the other part of the share array
    // copy to shared memory
    uint32_t max_elem = blockDim.x;
    if (block_offset + blockDim.x > input_arr_size) {
        // last block
        max_elem = input_arr_size - block_offset;
    }
    if (gid < input_arr_size) {
        shmem_inp[tid] = input_array[gid];
    }
    for(char j = 0; j < NUM_BITS; j++){
        char new_bit_offset = bit_offset + j;
        partition2(shmem_inp, tfs, ffs, max_elem, new_bit_offset);
        __syncthreads();
    }
    if (gid < input_arr_size) {
        input_array[gid] = shmem_inp[tid];
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
    __syncthreads();
}
