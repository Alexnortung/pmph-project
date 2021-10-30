#include "helper.cu.h"
#include "helperKernel.cu.h"

__global__ void make_histogram(uint32_t* input_array, const uint64_t input_arr_size, uint64_t bit_offset, uint32_t* histograms, uint32_t* relative_offsets) {
    const unsigned int histogram_size = 1 << NUM_BITS;
    int B = blockDim.x;
    uint64_t histogram_index = blockIdx.x;
    __shared__ int histogram[histogram_size];

    uint64_t bitmask = (histogram_size - 1) << bit_offset;
    for (int offset = 0; offset + B < histogram_size; offset += B) {
        if (offset + threadIdx.x < histogram_size) {
            histogram[offset + threadIdx.x] = 0;
        }
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

__global__ void histogram_scatter(int* histograms_multi_scanned, int* global_offsets, uint32_t* items, int bit_offset, uint32_t* relative_offsets, uint32_t* output) {
    //unsigned int histogram_size = 1 << NUM_BITS;
    //int gid = blockIdx.x*blockDim.x + threadIdx.x;
    ////if (gid >= ) // TODO:
    //uint32_t relative_offset = relative_offsets[gid];
    //int histogram_index = 0; // TODO:
    //char bin = bins[gid]; // TODO
    //uint32_t item = items[gid];
    //int global_offset = global_offsets[bin];
    //int histogram_offset = histograms_multi_scanned[histogram_index + bin];
    //int global_index = global_offset + histogram_offset + relative_offset;
    //output[global_index] = item;
}
