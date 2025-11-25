NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_80 -std=c++17 -lnvToolsExt
INCLUDES = -I../../third_party/cutlass/include -I../../third_party/cutlass/tools/util/include

TARGET = gemm
SRC = gemm_generalized.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

clean:
	rm -f $(TARGET)
