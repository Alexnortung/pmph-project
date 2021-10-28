#include "helper.cu.h"

__global__ void make_histogram(uint32_t* input_array, int bit_offset, int* histograms, int* output) {
    unsigned int histogram_size = 1 << NUM_BITS;
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    int B = blockDim.x;
    uint32_t items[ELEM_PER_THREAD];
    char bins[ELEM_PER_THREAD]; // If num_bits is greater than 8, this cannot be char.
    __shared__ int histogram[histogram_size];
    
    unsigned int bitmask = (histogram_size - 1) << bit_offset;

    int i = 0;
    // each thread loops over ELEM_PER_THREAD elements in the block with coalesced access
    for (int idx = threadIdx.x;idx < ELEM_PER_THREAD * B; idx += B) {
        uint32_t item = input_array[idx];
        //items[i] = item;
        uint32_t tmp_bin = item & bitmask;
        int bin = tmp_bin >> bit_offset;
        //bins[i] = (char)bin;
        // increment the value in the histogram and save the relative_offset
        int relative_offset = atomicAdd(&(histogram[bin]), 1);
        output[idx] = (item, bin, relative_offset);
        i++;
    }

    int histogram_index = blockIdx.x;
    for (int idx = threadIdx.x; idx < histogram_size; idx += B)
        histograms[histogram_index * histogram_size + idx] = histogram[idx];
    }
}

__global__ void histogram_scan() {

}

__global__ void histogram_scatter(int* histograms_multi_scanned, int* global_offsets, uint32_t* items, char* bins, uint32_t* relative_offsets, uint32_t* output) {
    unsigned int histogram_size = 1 << NUM_BITS;
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    //if (gid >= ) // TODO:
    uint32_t relative_offset = relative_offsets[gid];
    int histogram_index = 0; // TODO:
    char bin = bins[gid];
    uint32_t item = items[gid];
    int global_offset = global_offsets[bin];
    int histogram_offset = histograms_multi_scanned[histogram_index + bin];
    int global_index = global_offset + histogram_offset + relative_offset;
    output[global_index] = item;
}

__global__ void count_and_sort(uint32_t* input_array, uint32_t* output_array, unsigned int size, int* count_array, int bit_offset) {
    unsigned int count_arr_size = 1 << NUM_BITS;
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int indi_count_arr_index = gid * count_arr_size;
    unsigned int indi_arr_index = gid * ELEM_PER_THREAD;
    unsigned int bitmask = (count_arr_size - 1) << bit_offset;

    for(int i = 0; i < ELEM_PER_THREAD; i++){
        uint32_t elem = input_array[i + indi_arr_index];
        uint32_t part = elem & bitmask;
        uint32_t local_index = part >> bit_offset;
        count_array[indi_count_arr_index + local_index] +=1;
    }

    //Exclusive scan
    uint32_t* local_count_array[count_arr_size];
    uint32_t accum = 0;
    for(int i = 0; i < count_arr_size; i++){
        local_count_array[i] = accum;
        accum += count_array[indi_count_arr_index + i];
    }

    //Each thread sorts its elements of input array
    for(int j = 0; j < ELEM_PER_THREAD; i++){
        uint32_t elem = input_array[i + indi_arr_index];
        uint32_t part = elem & bitmask;
        uint32_t local_index = part >> bit_offset;
        uint32_t local_index_to_out = local_count_array[local_index];
        local_count_array[local_index] += 1; 
        output_array[local_index_to_out + indi_arr_index] = elem; 
    }

//d = index - localOffset + (his + GO)

}

__global__ void make_histogram(int* count_array, unsigned int size, unsigned int* histogram) {

}

__global__ void scatter(unsigned int* histogram, int* array, unsigned int size) {
    
}
