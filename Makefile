CUB=cub-1.8.0

all: cub-sort

ker-sort: main.cu helper.cu.h
	nvcc -I$(CUB)/cub -o test-ker main.cu
	./test-ker 100000000

cub-sort: CUBsort.cu helper.cu.h
	nvcc -I$(CUB)/cub -o test-CUB CUBsort.cu
	./test-CUB 10


clean:
	rm -f test-ker

test-make-histogram:
	nvcc -I$(CUB)/cub -o test-make-histogram testMakeHistogram.cu
	./test-make-histogram 5
