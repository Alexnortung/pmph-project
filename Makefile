CUB=cub-1.8.0

all: cub-sort

cub-sort: main.cu helper.cu.h
	nvcc -I$(CUB)/cub -o test-cub main.cu
	./test-cub 100000000

clean:
	rm -f test-cub

test-make-histogram:
	nvcc -I$(CUB)/cub -o test-make-histogram testMakeHistogram.cu
	./test-make-histogram 5
