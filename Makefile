NVCC := nvcc
CXX := g++
CXXFLAGS := -O2 -std=c++14
NVCCFLAGS := -O2 -std=c++14 -ccbin $(CXX) -Xcompiler "-fPIC"
INCLUDES := -Iinclude

SRC_CPP := src/othello.cpp src/serial/serial.cpp src/heuristic.cpp src/main.cpp
# benchmark and required source files
BENCH_SRC := src/benchmark/benchmark.cpp
CUDA_SRCS := src/naive_parallel/naive.cu src/parallel/parallel.cu src/parallel_optimized_1/parallel.cu

all: benchmark

benchmark:
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(BENCH_SRC) src/othello.cpp src/serial/serial.cpp src/heuristic.cpp $(CUDA_SRCS) -o bin/benchmark

clean:
	rm -f bin/benchmark

.PHONY: all benchmark clean
