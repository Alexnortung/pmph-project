#include "helper.cu.h"
#include "helperKernel.cu.h"

__global__ void make_histogram(uint32_t* input_array, const uint64_t input_arr_size, uint64_t bit_offset, uint32_t* histograms, uint32_t* relative_offsets) {
    const unsigned int histogram_size = 1 << NUM_BITS;
    int B = blockDim.x;
    uint64_t histogram_index = blockIdx.x;
    __shared__ int histogram[histogram_size];

    uint64_t bitmask = (histogram_size - 1) << bit_offset;
    for (int offset = 0; offset + threadIdx.x < histogram_size; offset += B) {
        histogram[offset + threadIdx.x] = 0;
    }

    int i = 0;
    // each thread loops over ELEM_PER_THREAD elements in the block with coalesced access
    int block_offset = ELEM_PER_THREAD * B * blockIdx.x;
    for (int idx = threadIdx.x; idx < ELEM_PER_THREAD * B; idx += B) {
        int access_index = idx + block_offset;
        if (access_index >= input_arr_size) break;
        uint32_t item = input_array[access_index];
        uint32_t tmp_bin = item & bitmask;
        uint32_t bin = tmp_bin >> bit_offset;
        // increment the value in the histogram and save the relative_offset
        uint32_t relative_offset = atomicAdd(&histogram[bin], 1);
        relative_offsets[access_index] = relative_offset;
        i++;
    }

    // input_array[id] <-> relative_offsets[id]
    __syncthreads();

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
    flags[gid] = gid % num_histograms == 0;
}

__global__ void histogram_scatter(uint32_t* histograms_multi_scanned, const uint64_t input_arr_size, const uint32_t elements_per_histogram, uint32_t* global_offsets, uint32_t* items, uint32_t bit_offset, uint32_t* relative_offsets, uint32_t* output) {
    unsigned int histogram_size = 1 << NUM_BITS;
    unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    uint32_t item;
    uint64_t global_index
    if (gid < input_arr_size) {
        uint64_t bitmask = (histogram_size - 1) << bit_offset;
        uint32_t tmp_bin = item & bitmask;
        uint32_t bin = tmp_bin >> bit_offset;
        item = items[gid];
        uint32_t global_offset = global_offsets[bin];
        uint32_t histogram_index = gid / elements_per_histogram;
        uint32_t histogram_offset = histograms_multi_scanned[histogram_index + bin];
        uint32_t relative_offset = relative_offsets[gid];
        global_index = global_offset + histogram_offset + relative_offset;
    }
    __syncthreads();
    if (gid < input_arr_size) {
        output[global_index] = item;
    }
}
