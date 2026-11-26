CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Werror -Isrc

.PHONY: all test gpu clean

all: gather_ref pq_layer

gather_ref: src/gather_ref.cc src/gather_layout.h
	$(CXX) $(CXXFLAGS) src/gather_ref.cc -o gather_ref

pq_layer: src/pq_layer.cc
	$(CXX) $(CXXFLAGS) src/pq_layer.cc -o pq_layer

gather_bench: src/gather_bench.cu src/gather_layout.h
	nvcc -O3 -std=c++17 -arch=sm_90 src/gather_bench.cu -o gather_bench

gpu: gather_bench

test: all
	./gather_ref
	./pq_layer

clean:
	rm -f gather_ref pq_layer gather_bench
