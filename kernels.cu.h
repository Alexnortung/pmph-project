#include "helper.cu.h"

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
