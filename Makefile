CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Werror -Isrc

# gather_bench 只在测量主机编译。云端机器没有 nvcc，不要在这里执行 make gpu。
# 默认同时生成 sm_89（RTX 4090）和 sm_90（H100）的代码。
# 1.2 ms/token 是方案里 H100 的原文口径，不是 4090 的停线。S2 阈值留空。
NVCC ?= nvcc
NVCC_ARCH ?= -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90

.PHONY: all test gpu clean

all: gather_ref pq_layer

gather_ref: src/gather_ref.cc src/gather_layout.h
	$(CXX) $(CXXFLAGS) src/gather_ref.cc -o gather_ref

pq_layer: src/pq_layer.cc
	$(CXX) $(CXXFLAGS) src/pq_layer.cc -o pq_layer

gather_bench: src/gather_bench.cu src/gather_layout.h
	$(NVCC) -O3 -std=c++17 $(NVCC_ARCH) src/gather_bench.cu -o gather_bench

gpu: gather_bench

test: all
	./gather_ref
	./pq_layer

clean:
	rm -f gather_ref pq_layer gather_bench
