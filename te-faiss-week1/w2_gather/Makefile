# W2 gather microbench. Compile on the target host only.
# CUDA 12.8, sm_89 (RTX 4090). Do not retarget to Tesla P4 / sm_61 or H100-only archs.
#
#   export CUDA_HOME=/usr/local/cuda-12.8
#   make -C /workspace/te-faiss-week1/w2_gather

CUDA_HOME ?= /usr/local/cuda-12.8
NVCC     ?= $(CUDA_HOME)/bin/nvcc

SRC_DIR  := src
OUT_DIR  := out
TARGET   := $(OUT_DIR)/gather_microbench
SRC      := $(SRC_DIR)/gather_microbench.cu

NVCCFLAGS := -O3 -std=c++17 -arch=sm_89 -lineinfo \
             -I$(SRC_DIR) \
             -Xcompiler -Wall,-Wextra,-fno-strict-aliasing

.PHONY: all clean check-nvcc

all: check-nvcc $(TARGET)

check-nvcc:
	@if [ ! -x "$(NVCC)" ]; then \
	  echo "error: nvcc not found at $(NVCC)"; \
	  echo "  On the Chenjian Cloud 4090 host, set CUDA_HOME=/usr/local/cuda-12.8"; \
	  echo "  This Makefile will not fall back to another CUDA toolkit."; \
	  exit 2; \
	fi
	@echo "Using NVCC=$(NVCC)"
	@$(NVCC) --version

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

$(TARGET): $(SRC) $(SRC_DIR)/llama3_8b_shapes.h | $(OUT_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRC)

clean:
	rm -f $(TARGET)
