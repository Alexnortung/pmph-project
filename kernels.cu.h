#include <algorithm>
#include "helper.cu.h"
#include "helperKernel.cu.h"

template<class T>
__global__ void make_histogram(T* input_array
                             , const uint64_t input_arr_size
                             , uint64_t bit_offset
                             , uint32_t* histograms
                             , uint32_t* relative_offsets
) {
    const unsigned int histogram_size = 1 << NUM_BITS;
    unsigned int B = blockDim.x;
    uint64_t histogram_index = blockIdx.x;
    __shared__ uint32_t histogram[histogram_size];
    __shared__ uint32_t histogram_scanned[histogram_size];

    uint64_t bitmask = (histogram_size - 1) << bit_offset;
    for (int offset = 0; offset + threadIdx.x < histogram_size; offset += B) {
        histogram[offset + threadIdx.x] = 0;
    }

    T priv_arr[ELEM_PER_THREAD_MAKE_HIST];
    int priv_rel_offs[ELEM_PER_THREAD_MAKE_HIST];
    uint64_t priv_bins[ELEM_PER_THREAD_MAKE_HIST];
    int i = 0;
    // each thread loops over ELEM_PER_THREAD elements in the block with coalesced access
    int block_offset = ELEM_PER_THREAD_MAKE_HIST * B * blockIdx.x;
    for (int idx = block_offset + threadIdx.x; idx < std::min(block_offset + ELEM_PER_THREAD_MAKE_HIST * B, input_arr_size); idx += B) {
        T item = input_array[idx];
        priv_arr[i] = item;
        uint64_t tmp_bin = item & bitmask;
        uint64_t bin = tmp_bin >> bit_offset;
        priv_bins[i] = bin;
        // increment the value in the histogram and save the relative_offset
        uint32_t relative_offset = atomicAdd(&histogram[bin], 1);
        priv_rel_offs[i] = relative_offset;
        i++;
    }

    __syncthreads();
    // todo scan histogram;

    i = 0;
    for (int idx = block_offset + threadIdx.x; threadIdx.x < histogram_size && idx < std::min(block_offset + ELEM_PER_THREAD_MAKE_HIST * B, input_arr_size); idx += B) {
        T item = priv_arr[i];
        uint32_t relative_offset = priv_rel_offs[i];
        uint64_t bin = priv_bins[i];
        elm = scanIncBlock<Add<uint32_t>>(shmem_red, threadIdx.x);
        histogram_scanned[threadIdx.x] = elm;
        i++;
    }
    __syncthreads();

    // input_array[id] <-> relative_offsets[id]

    for (int offset = 0; offset + threadIdx.x < histogram_size; offset += B) {
        histograms[histogram_index * histogram_size + offset + threadIdx.x] = histogram[offset + threadIdx.x];
    }
}

template<class T>
__global__ void make_histogram_flags(T num_histograms, char* flags) {
    uint32_t histogram_size = 1 << NUM_BITS;
    unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if (gid >= num_histograms * histogram_size) {
        return;
    }
    flags[gid] = gid % num_histograms == 0; // TODO: maybe change modulo (%) ?
}

template<class T>
__global__ void histogram_scatter(uint32_t* histograms_multi_scanned
                                , const uint64_t input_arr_size
                                , const uint32_t elements_per_histogram
                                , uint32_t* global_offsets
                                , T* items
                                , uint64_t bit_offset
                                , uint32_t* relative_offsets
                                , T* output
) {
    unsigned int histogram_size = 1 << NUM_BITS;
    unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    T item;
    uint64_t global_index;
    if (gid < input_arr_size) {
        item = items[gid];
        uint64_t bitmask = (histogram_size - 1) << bit_offset;
        uint64_t tmp_bin = item & bitmask;
        uint64_t bin = tmp_bin >> bit_offset;
        unsigned int global_offset;
        if (bin <= 0) {
            global_offset = 0;
        } else {
            global_offset = global_offsets[bin - 1];
        }
        unsigned int histogram_index = gid / elements_per_histogram;
        uint32_t histogram_offset;
        if (histogram_index <= 0) {
            histogram_offset = 0;
        } else {
            histogram_offset = histograms_multi_scanned[histogram_index + bin];
        }
        unsigned int relative_offset = relative_offsets[gid];
        global_index = global_offset + histogram_offset + relative_offset;
    }
    __syncthreads();
    if (gid < input_arr_size) {
        output[global_index] = item;
    }
}
