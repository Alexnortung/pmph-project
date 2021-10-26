#include <stdio.h>
#include <stdlib.h>
#include "kernels.cu.h"

// Validation - use sequential sorting
// Generate data
// Timing
// kernel calling function - call kernels
// main function

// Generate data
int* make_random_array(unsigned int size) {
    int* random_array = malloc(sizeof(int) * size);
    for(unsigned int i = 0; i < size; i++) {
        random_array[i] = rand();
    }
    return random_array;
}

void print_usage() {
    printf("Usage: %s <num_elements>\n", argv[0]);
}

int main(int argc, char* argv[]) {
    // Validation
    if (argc != 1 + 1) {
        print_usage();
        exit(1);
    }

    int num_elements_tmp = atoi(argv[1]);
    if (num_elements_tmp <= 0) {
        printf("Number of elements should be greater than 0\n");
        print_usage();
        exit(1);
    }

    unsigned int num_elements = (unsigned int)num_elements_tmp;

    int* random_array = make_random_array(num_elements);
    
}
