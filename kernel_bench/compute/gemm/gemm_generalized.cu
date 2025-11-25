#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>

// =========================================================================
// Error Handling Macros
// =========================================================================
// (No special macros needed for NVTX)

// =========================================================================
// TensorProfiler Class (NVTX Wrapper)
// =========================================================================
class TensorProfiler {
public:
  TensorProfiler(bool enable) : enabled_(enable) {
    if (enabled_) {
      range_id_ = nvtxRangeStart("TensorCoreRange");
    }
  }

  ~TensorProfiler() {
    if (enabled_) {
      nvtxRangeEnd(range_id_);
    }
  }

private:
  bool enabled_;
  nvtxRangeId_t range_id_
};
// =========================================================================
// 0. Basic Definitions & Utils
// =========================================================================
#define CUDA_CHECK(status)                                                     \
  {                                                                            \
    if (status != cudaSuccess) {                                               \
      std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl;  \
      exit(1);                                                                 \
    }                                                                          \
  }

using ArchTag = cutlass::arch::Sm80;
using OpClass = cutlass::arch::OpClassTensorOp;

struct DeviceSpecs {
  std::string name;
  double peak_tflops; // TFLOPS
  double peak_bw;     // GB/s
  double clock_rate_khz;
  int num_sms;

  static DeviceSpecs &get() {
    static DeviceSpecs instance;
    static bool init = false;
    if (!init) {
      int dev;
      cudaGetDevice(&dev);
      cudaDeviceProp p;
      cudaGetDeviceProperties(&p, dev);
      instance.name = p.name;
      instance.clock_rate_khz = (double)p.clockRate;
      instance.num_sms = p.multiProcessorCount;

      std::string name_str = p.name;
      // Rough estimates for peak FP16 Tensor Core TFLOPS
      if (name_str.find("H100") != std::string::npos) {
        instance.peak_tflops = 989.0;
        instance.peak_bw = 3350.0;
      } else if (name_str.find("A100") != std::string::npos) {
        instance.peak_tflops = 312.0;
        instance.peak_bw = 1555.0;
      } else if (name_str.find("V100") != std::string::npos) {
        instance.peak_tflops = 125.0;
        instance.peak_bw = 900.0;
      } else if (name_str.find("4090") != std::string::npos) {
        instance.peak_tflops = 330.0;
        instance.peak_bw = 1008.0;
      } else {
        instance.peak_tflops = 100.0;
        instance.peak_bw = 900.0;
      }
      init = true;
    }
    return instance;
  }
};

enum class RunStatus { Success, SMemExceeded, Unsupported, RuntimeError };

struct RunResult {
  std::string name;
  RunStatus status;
  double time_ms;
  double tflops;
  long long cycles;
  double mfu;
  double measured_mfu;
  double hbm_eff;

  int grid_size;
  int active_blocks;
  double grid_util;
  int split_k;
};

struct CaseResult {
  std::string dtype;
  int m, n, k;
  RunResult best_run;
  std::vector<RunResult> all_runs;
};

// =========================================================================
// 1. Argument Parser
// =========================================================================
struct ArgParser {
  static std::vector<int> parse(const std::string &input) {
    std::vector<int> values;
    if (input.front() == '[' && input.back() == ']') {
      std::string content = input.substr(1, input.size() - 2);
      size_t colon = content.find(':');
      if (colon == std::string::npos) {
        std::cerr << "Invalid range format. Use [start:end]" << std::endl;
        exit(1);
      }
      int start = std::stoi(content.substr(0, colon));
      int end = std::stoi(content.substr(colon + 1));

      for (int i = start; i <= end; i *= 2) {
        values.push_back(i);
      }
    } else {
      values.push_back(std::stoi(input));
    }
    return values;
  }
};

// =========================================================================
// 2. CUTLASS Runner Interface
// =========================================================================
struct GemmRunner {
  virtual ~GemmRunner() {}
  virtual std::string name() const = 0;
  virtual RunResult run(int M, int N, int K, void *d_A, void *d_B, void *d_C,
                        bool profile) = 0;
};

struct DeviceProps {
  int max_smem;
  static const DeviceProps &get() {
    static DeviceProps instance;
    static bool init = false;
    if (!init) {
      int dev;
      cudaGetDevice(&dev);
      cudaDeviceProp p;
      cudaGetDeviceProperties(&p, dev);
      instance.max_smem = p.sharedMemPerBlockOptin;
      init = true;
    }
    return instance;
  }
};

// =========================================================================
// 3. Gemm Implementations
// =========================================================================

// Standard GEMM
template <typename ElementA, typename LayoutA, typename ElementB,
          typename LayoutB, typename ElementC, typename LayoutC,
          typename ElementAccumulator, typename ThreadblockShape,
          typename WarpShape, typename InstructionShape, int kStages,
          int kAlignmentA, int kAlignmentB>
struct GemmImpl : public GemmRunner {

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementC, 1, ElementAccumulator, ElementAccumulator>;

  using GemmHandle = cutlass::gemm::device::Gemm<
      ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
      ElementAccumulator, OpClass, ArchTag, ThreadblockShape, WarpShape,
      InstructionShape, EpilogueOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, kStages,
      kAlignmentA, kAlignmentB>;

  using DeviceKernel = typename GemmHandle::GemmKernel;

  std::string name() const override {
    std::stringstream ss;
    ss << ThreadblockShape::kM << "x" << ThreadblockShape::kN << "x"
       << ThreadblockShape::kK << "_S" << kStages;
    return ss.str();
  }

  RunResult run(int M, int N, int K, void *d_A, void *d_B, void *d_C,
                bool profile) override {
    RunResult res;
    res.name = name();
    res.split_k = 1;
    res.time_ms = 0;
    res.tflops = 0;
    res.cycles = 0;
    res.mfu = 0;
    res.hbm_eff = 0;
    res.grid_size = 0;
    res.active_blocks = 0;
    res.grid_util = 0;

    int smem = sizeof(typename DeviceKernel::SharedStorage);
    if (smem > DeviceProps::get().max_smem) {
      res.status = RunStatus::SMemExceeded;
      return res;
    }

    int lda = (std::is_same<LayoutA, cutlass::layout::RowMajor>::value) ? K : M;
    int ldb = (std::is_same<LayoutB, cutlass::layout::RowMajor>::value) ? N : K;
    int ldc = (std::is_same<LayoutC, cutlass::layout::RowMajor>::value) ? N : M;

    typename GemmHandle::Arguments args(
        {M, N, K}, {static_cast<ElementA *>(d_A), lda},
        {static_cast<ElementB *>(d_B), ldb},
        {static_cast<ElementC *>(d_C), ldc},
        {static_cast<ElementC *>(d_C), ldc},
        {ElementAccumulator(1.0), ElementAccumulator(0.0)});
    GemmHandle gemm;

    if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
      res.status = RunStatus::Unsupported;
      return res;
    }

    if (smem >= (48 << 10)) {
      if (cudaFuncSetAttribute(cutlass::Kernel<DeviceKernel>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               smem) != cudaSuccess) {
        res.status = RunStatus::SMemExceeded;
        return res;
      }
    }

    int max_active_blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks, cutlass::Kernel<DeviceKernel>,
        DeviceKernel::kThreadCount, smem);
    res.active_blocks = max_active_blocks;

    int grid_m = (M + ThreadblockShape::kM - 1) / ThreadblockShape::kM;
    int grid_n = (N + ThreadblockShape::kN - 1) / ThreadblockShape::kN;
    res.grid_size = grid_m * grid_n;

    const auto &specs = DeviceSpecs::get();
    res.grid_util = (double)res.grid_size / (double)specs.num_sms;

    size_t ws_size = GemmHandle::get_workspace_size(args);
    void *ws = nullptr;
    if (ws_size)
      cudaMalloc(&ws, ws_size);

    if (gemm.initialize(args, ws) != cutlass::Status::kSuccess) {
      if (ws)
        cudaFree(ws);
      res.status = RunStatus::RuntimeError;
      return res;
    }

    res.measured_mfu = -1.0;
    gemm(); // Warmup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Timing Run
    cudaEventRecord(start);
    for (int i = 0; i < 5; i++)
      gemm();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Profiling Run
    if (profile) {
      for (int i = 0; i < 5; i++) {
        TensorProfiler profiler(true);
        gemm();
      }
      cudaDeviceSynchronize();
    }

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    res.time_ms = ms / 5.0f;

    double gflops = (2.0 * M * N * K) * 1e-9;
    res.tflops = gflops / res.time_ms;
    res.cycles = (long long)(res.time_ms * specs.clock_rate_khz);
    res.mfu = (res.tflops / specs.peak_tflops) * 100.0;

    double bytes =
        (double)(M * K * sizeof(ElementA) + K * N * sizeof(ElementB) +
                 M * N * sizeof(ElementC));
    double gb = bytes * 1e-9;
    double bw = gb / (res.time_ms * 1e-3);
    res.hbm_eff = (bw / specs.peak_bw) * 100.0;

    res.status = RunStatus::Success;

    if (ws)
      cudaFree(ws);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return res;
  }
};

// Split-K Parallel GEMM
template <typename ElementA, typename LayoutA, typename ElementB,
          typename LayoutB, typename ElementC, typename LayoutC,
          typename ElementAccumulator, typename ThreadblockShape,
          typename WarpShape, typename InstructionShape, int kStages,
          int kSplitK, int kAlignmentA, int kAlignmentB>
struct GemmSplitKImpl : public GemmRunner {
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementC, 1, ElementAccumulator, ElementAccumulator>;

  using GemmHandle = cutlass::gemm::device::GemmSplitKParallel<
      ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
      ElementAccumulator, OpClass, ArchTag, ThreadblockShape, WarpShape,
      InstructionShape, EpilogueOp,
      cutlass::epilogue::thread::Convert<ElementAccumulator, 1,
                                         ElementAccumulator>,
      cutlass::reduction::thread::ReduceAdd<ElementAccumulator,
                                            ElementAccumulator, 1>,
      cutlass::gemm::threadblock::GemmSplitKHorizontalThreadblockSwizzle,
      kStages, kAlignmentA, kAlignmentB>;

  using DeviceKernel = typename GemmHandle::GemmKernel;

  std::string name() const override {
    std::stringstream ss;
    ss << ThreadblockShape::kM << "x" << ThreadblockShape::kN << "x"
       << ThreadblockShape::kK << "_S" << kStages << "_SK" << kSplitK;
    return ss.str();
  }

  RunResult run(int M, int N, int K, void *d_A, void *d_B, void *d_C,
                bool profile) override {
    RunResult res;
    res.name = name();
    res.split_k = kSplitK;
    res.time_ms = 0;
    res.tflops = 0;
    res.cycles = 0;
    res.mfu = 0;
    res.hbm_eff = 0;
    res.grid_size = 0;
    res.active_blocks = 0;
    res.grid_util = 0;

    int smem = sizeof(typename DeviceKernel::SharedStorage);
    if (smem > DeviceProps::get().max_smem) {
      res.status = RunStatus::SMemExceeded;
      return res;
    }

    int lda = (std::is_same<LayoutA, cutlass::layout::RowMajor>::value) ? K : M;
    int ldb = (std::is_same<LayoutB, cutlass::layout::RowMajor>::value) ? N : K;
    int ldc = (std::is_same<LayoutC, cutlass::layout::RowMajor>::value) ? N : M;

    typename GemmHandle::Arguments args(
        {M, N, K}, {static_cast<ElementA *>(d_A), lda},
        {static_cast<ElementB *>(d_B), ldb},
        {static_cast<ElementC *>(d_C), ldc},
        {static_cast<ElementC *>(d_C), ldc},
        {ElementAccumulator(1.0), ElementAccumulator(0.0)}, kSplitK);
    GemmHandle gemm;

    if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
      res.status = RunStatus::Unsupported;
      return res;
    }

    if (smem >= (48 << 10)) {
      if (cudaFuncSetAttribute(cutlass::Kernel<DeviceKernel>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               smem) != cudaSuccess) {
        res.status = RunStatus::SMemExceeded;
        return res;
      }
    }

    int max_active_blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks, cutlass::Kernel<DeviceKernel>,
        DeviceKernel::kThreadCount, smem);
    res.active_blocks = max_active_blocks;

    int grid_m = (M + ThreadblockShape::kM - 1) / ThreadblockShape::kM;
    int grid_n = (N + ThreadblockShape::kN - 1) / ThreadblockShape::kN;
    res.grid_size = grid_m * grid_n * kSplitK;

    const auto &specs = DeviceSpecs::get();
    res.grid_util = (double)res.grid_size / (double)specs.num_sms;

    size_t ws_size = GemmHandle::get_workspace_size(args);
    void *ws = nullptr;
    if (ws_size)
      cudaMalloc(&ws, ws_size);

    if (gemm.initialize(args, ws) != cutlass::Status::kSuccess) {
      if (ws)
        cudaFree(ws);
      res.status = RunStatus::RuntimeError;
      return res;
    }

    res.measured_mfu = -1.0;
    gemm(); // Warmup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Timing Run
    cudaEventRecord(start);
    for (int i = 0; i < 5; i++)
      gemm();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Profiling Run
    if (profile) {
      TensorProfiler profiler(true);
      gemm();
      cudaDeviceSynchronize();
    }

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    res.time_ms = ms / 5.0f;

    double gflops = (2.0 * M * N * K) * 1e-9;
    res.tflops = gflops / res.time_ms;
    res.cycles = (long long)(res.time_ms * specs.clock_rate_khz);
    res.mfu = (res.tflops / specs.peak_tflops) * 100.0;

    double bytes =
        (double)(M * K * sizeof(ElementA) + K * N * sizeof(ElementB) +
                 M * N * sizeof(ElementC));
    double gb = bytes * 1e-9;
    double bw = gb / (res.time_ms * 1e-3);
    res.hbm_eff = (bw / specs.peak_bw) * 100.0;

    res.status = RunStatus::Success;

    if (ws)
      cudaFree(ws);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return res;
  }
};

// =========================================================================
// 4. Sweeper Class
// =========================================================================
class GemmSweeper {
  std::vector<std::shared_ptr<GemmRunner>> configs;

public:
  GemmSweeper(std::string dtype) {
    if (dtype == "fp32") {
      setup_fp32();
    } else if (dtype == "fp16") {
      setup_fp16();
    } else if (dtype == "bf16") {
      setup_bf16();
    } else if (dtype == "int8") {
      setup_int8();
    } else {
      std::cerr << "Unknown dtype: " << dtype << std::endl;
      exit(1);
    }
  }

  void setup_fp32() {
    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    using ElementAccum = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using Inst = cutlass::gemm::GemmShape<16, 8, 8>;

#define REG_FP32(Tm, Tn, Tk, Wm, Wn, Wk, Stg)                                  \
  configs.push_back(                                                           \
      std::make_shared<                                                        \
          GemmImpl<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,    \
                   ElementAccum, cutlass::gemm::GemmShape<Tm, Tn, Tk>,         \
                   cutlass::gemm::GemmShape<Wm, Wn, Wk>, Inst, Stg, 4, 4>>())

#define REG_SK_FP32(Tm, Tn, Tk, Wm, Wn, Wk, Stg, SK)                           \
  configs.push_back(                                                           \
      std::make_shared<GemmSplitKImpl<                                         \
          ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,             \
          ElementAccum, cutlass::gemm::GemmShape<Tm, Tn, Tk>,                  \
          cutlass::gemm::GemmShape<Wm, Wn, Wk>, Inst, Stg, SK, 4, 4>>())

    REG_FP32(128, 128, 32, 64, 64, 32, 3);
    REG_FP32(128, 256, 32, 64, 64, 32, 3);
    REG_FP32(256, 256, 32, 64, 64, 32, 3);
    REG_FP32(128, 64, 32, 64, 32, 32, 3);
    REG_FP32(64, 64, 32, 32, 32, 32, 3);
    REG_SK_FP32(128, 128, 32, 64, 64, 32, 3, 2);
    REG_SK_FP32(64, 64, 32, 32, 32, 32, 3, 4);
  }

  void setup_fp16() {
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementAccum = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using Inst = cutlass::gemm::GemmShape<16, 8, 16>;

#define REG_FP16(Tm, Tn, Tk, Wm, Wn, Wk, Stg)                                  \
  configs.push_back(                                                           \
      std::make_shared<                                                        \
          GemmImpl<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,    \
                   ElementAccum, cutlass::gemm::GemmShape<Tm, Tn, Tk>,         \
                   cutlass::gemm::GemmShape<Wm, Wn, Wk>, Inst, Stg, 8, 8>>())

#define REG_SK_FP16(Tm, Tn, Tk, Wm, Wn, Wk, Stg, SK)                           \
  configs.push_back(                                                           \
      std::make_shared<GemmSplitKImpl<                                         \
          ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,             \
          ElementAccum, cutlass::gemm::GemmShape<Tm, Tn, Tk>,                  \
          cutlass::gemm::GemmShape<Wm, Wn, Wk>, Inst, Stg, SK, 8, 8>>())

    REG_FP16(256, 128, 32, 64, 64, 32, 3);
    REG_FP16(128, 256, 64, 64, 64, 64, 3);
    REG_FP16(256, 256, 64, 64, 64, 64, 3);
    REG_FP16(128, 256, 32, 64, 64, 32, 3);
    REG_FP16(128, 128, 32, 64, 64, 32, 3);
    REG_FP16(64, 64, 32, 32, 32, 32, 3);
    REG_SK_FP16(128, 128, 32, 64, 64, 32, 3, 2);
    REG_SK_FP16(64, 64, 32, 32, 32, 32, 3, 4);
  }

  void setup_bf16() {
    using ElementA = cutlass::bfloat16_t;
    using ElementB = cutlass::bfloat16_t;
    using ElementC = cutlass::bfloat16_t;
    using ElementAccum = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using Inst = cutlass::gemm::GemmShape<16, 8, 16>;

#define REG_BF16(Tm, Tn, Tk, Wm, Wn, Wk, Stg)                                  \
  configs.push_back(                                                           \
      std::make_shared<                                                        \
          GemmImpl<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,    \
                   ElementAccum, cutlass::gemm::GemmShape<Tm, Tn, Tk>,         \
                   cutlass::gemm::GemmShape<Wm, Wn, Wk>, Inst, Stg, 8, 8>>())

#define REG_SK_BF16(Tm, Tn, Tk, Wm, Wn, Wk, Stg, SK)                           \
  configs.push_back(                                                           \
      std::make_shared<GemmSplitKImpl<                                         \
          ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,             \
          ElementAccum, cutlass::gemm::GemmShape<Tm, Tn, Tk>,                  \
          cutlass::gemm::GemmShape<Wm, Wn, Wk>, Inst, Stg, SK, 8, 8>>())

    REG_BF16(256, 128, 32, 64, 64, 32, 3);
    REG_BF16(128, 256, 64, 64, 64, 64, 3);
    REG_BF16(256, 256, 64, 64, 64, 64, 3);
    REG_BF16(128, 128, 32, 64, 64, 32, 3);
    REG_SK_BF16(128, 128, 32, 64, 64, 32, 3, 2);
  }

  void setup_int8() {
    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = int32_t;
    using ElementAccum = int32_t;
    // INT8 Tensor Core typically prefers ColumnMajor B
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using Inst = cutlass::gemm::GemmShape<16, 8, 32>;

#define REG_INT8(Tm, Tn, Tk, Wm, Wn, Wk, Stg)                                  \
  configs.push_back(                                                           \
      std::make_shared<GemmImpl<                                               \
          ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,             \
          ElementAccum, cutlass::gemm::GemmShape<Tm, Tn, Tk>,                  \
          cutlass::gemm::GemmShape<Wm, Wn, Wk>, Inst, Stg, 16, 16>>())

#define REG_SK_INT8(Tm, Tn, Tk, Wm, Wn, Wk, Stg, SK)                           \
  configs.push_back(                                                           \
      std::make_shared<GemmSplitKImpl<                                         \
          ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,             \
          ElementAccum, cutlass::gemm::GemmShape<Tm, Tn, Tk>,                  \
          cutlass::gemm::GemmShape<Wm, Wn, Wk>, Inst, Stg, SK, 16, 16>>())

    REG_INT8(256, 128, 64, 64, 64, 64, 3);
    REG_INT8(128, 256, 64, 64, 64, 64, 3);
    REG_INT8(256, 256, 64, 64, 64, 64, 3);
    REG_INT8(128, 128, 64, 64, 64, 64, 3);
    REG_SK_INT8(128, 128, 64, 64, 64, 64, 3, 2);
  }

  CaseResult find_best(int M, int N, int K, void *d_A, void *d_B, void *d_C,
                       bool profile) {
    CaseResult result;
    result.m = M;
    result.n = N;
    result.k = K;
    result.best_run.tflops = -1.0;

    for (auto &cfg : configs) {
      RunResult res = cfg->run(M, N, K, d_A, d_B, d_C, profile);
      result.all_runs.push_back(res);

      if (res.status == RunStatus::Success &&
          res.tflops > result.best_run.tflops) {
        result.best_run = res;
      }
    }
    return result;
  }
};

// =========================================================================
// 5. Main
// =========================================================================
int main(int argc, char **argv) {
  std::string m_str, n_str, k_str;
  bool profile = false;
  std::string csv_path = "gemm_results.csv";
  std::string dtype_arg = "fp32";

  double override_peak_tflops = 0.0;
  double override_peak_bw = 0.0;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--M") == 0 && i + 1 < argc)
      m_str = argv[++i];
    else if (strcmp(argv[i], "--N") == 0 && i + 1 < argc)
      n_str = argv[++i];
    else if (strcmp(argv[i], "--K") == 0 && i + 1 < argc)
      k_str = argv[++i];
    else if (strcmp(argv[i], "--profile") == 0)
      profile = true;
    else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc)
      csv_path = argv[++i];
    else if (strcmp(argv[i], "--dtype") == 0 && i + 1 < argc)
      dtype_arg = argv[++i];
    else if (strcmp(argv[i], "--peak_tflops") == 0 && i + 1 < argc)
      override_peak_tflops = std::stod(argv[++i]);
    else if (strcmp(argv[i], "--peak_bw") == 0 && i + 1 < argc)
      override_peak_bw = std::stod(argv[++i]);
  }

  if (m_str.empty() || n_str.empty() || k_str.empty()) {
    std::cerr << "Usage: " << argv[0]
              << " --M [start:end] --N [start:end] --K val [--dtype "
                 "fp32,fp16,bf16,int8] [--profile] [--csv path]"
              << std::endl;
    return 1;
  }

  if (override_peak_tflops > 0)
    DeviceSpecs::get().peak_tflops = override_peak_tflops;
  if (override_peak_bw > 0)
    DeviceSpecs::get().peak_bw = override_peak_bw;

  const auto &specs = DeviceSpecs::get();
  std::cout << ">>> Device: " << specs.name << " (" << specs.num_sms << " SMs)"
            << std::endl;
  std::cout << ">>> Peak TFLOPS: " << specs.peak_tflops
            << " | Peak BW: " << specs.peak_bw << " GB/s" << std::endl;

  // Parse dtypes
  std::vector<std::string> dtypes;
  std::stringstream ss(dtype_arg);
  std::string segment;
  while (std::getline(ss, segment, ',')) {
    dtypes.push_back(segment);
  }

  auto ms = ArgParser::parse(m_str);
  auto ns = ArgParser::parse(n_str);
  auto ks = ArgParser::parse(k_str);

  int max_m = *std::max_element(ms.begin(), ms.end());
  int max_n = *std::max_element(ns.begin(), ns.end());
  int max_k = *std::max_element(ks.begin(), ks.end());

  // Allocate for largest possible element size (4 bytes for FP32/INT32)
  size_t element_size = 4;
  size_t bytes_A = (size_t)max_m * max_k * element_size;
  size_t bytes_B = (size_t)max_k * max_n * element_size;
  size_t bytes_C = (size_t)max_m * max_n * element_size;

  void *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
  CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
  CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
  cudaMemset(d_A, 0, bytes_A);
  cudaMemset(d_B, 0, bytes_B);
  cudaMemset(d_C, 0, bytes_C);

  std::vector<CaseResult> summary;

  std::cout << ">>> Starting Scan... Output CSV: " << csv_path << std::endl;
  std::cout << std::left << std::setw(8) << "Type" << std::setw(8) << "M"
            << std::setw(8) << "N" << std::setw(8) << "K"
            << "| " << std::setw(20) << "Best Tile"
            << " | " << std::setw(8) << "TFLOPS"
            << " | " << std::setw(8) << "MFU(%)"
            << " | " << std::setw(8) << "Waves" << std::endl;
  std::cout << "---------------------------------------------------------------"
               "-----------------"
            << std::endl;

  for (const auto &dtype : dtypes) {
    std::cout << ">>> Data Type: " << dtype << std::endl;
    GemmSweeper sweeper(dtype);

    for (int m : ms) {
      for (int n : ns) {
        for (int k : ks) {
          CaseResult res = sweeper.find_best(m, n, k, d_A, d_B, d_C, profile);
          res.dtype = dtype; // Set dtype in result
          summary.push_back(res);

          std::cout << std::left << std::setw(8) << dtype << std::setw(8) << m
                    << std::setw(8) << n << std::setw(8) << k << "| "
                    << std::setw(20) << res.best_run.name << " | "
                    << std::setw(8) << std::fixed << std::setprecision(2)
                    << res.best_run.tflops << " | " << std::setw(8)
                    << res.best_run.mfu << " | " << std::setw(8)
                    << std::setprecision(2) << res.best_run.grid_util
                    << std::endl;
        }
      }
    }
  }

  // Write CSV
  std::ofstream csv(csv_path);
  csv << "dtype,M,N,K,Tile,Status,Time_ms,Cycles,TFLOPS,MFU,measured_mfu,HBM_"
         "Eff,GridSize,"
         "ActiveBlocks,"
         "Waves,SplitK\n";
  for (const auto &case_res : summary) {
    for (const auto &run : case_res.all_runs) {
      csv << case_res.dtype << "," << case_res.m << "," << case_res.n << ","
          << case_res.k << "," << run.name << "," << (int)run.status << ","
          << run.time_ms << "," << run.cycles << "," << run.tflops << ","
          << run.mfu << "," << run.measured_mfu << "," << run.hbm_eff << ","
          << run.grid_size << "," << run.active_blocks << "," << run.grid_util
          << "," << run.split_k << "\n";
    }
  }
  csv.close();

  // Final Report
  std::cout << "\n============================================================="
               "====================================================="
            << std::endl;
  std::cout << "                                        FINAL SUMMARY REPORT   "
               "                                                   "
            << std::endl;
  std::cout << "==============================================================="
               "==================================================="
            << std::endl;
  std::cout << "| " << std::left << std::setw(6) << "M"
            << "| " << std::setw(6) << "N"
            << "| " << std::setw(6) << "K"
            << "| " << std::setw(18) << "Winner Config"
            << "| " << std::setw(8) << "Time(ms)"
            << "| " << std::setw(10) << "Cycles"
            << "| " << std::setw(8) << "TFLOPS"
            << "| " << std::setw(8) << "MFU(%)"
            << "| " << std::setw(8) << "HBM(%)"
            << "| " << std::setw(8) << "Waves"
            << " |" << std::endl;
  std::cout << "|-------+-------+-------+-------------------+----------+-------"
               "---+--------+--------+--------+--------|"
            << std::endl;

  for (const auto &r : summary) {
    std::cout << "| " << std::left << std::setw(6) << r.m << "| "
              << std::setw(6) << r.n << "| " << std::setw(6) << r.k
              << "| \033[32m" << std::setw(18) << r.best_run.name << "\033[0m"
              << "| " << std::setw(8) << std::fixed << std::setprecision(3)
              << r.best_run.time_ms << "| " << std::setw(10)
              << r.best_run.cycles << "| " << std::setw(8)
              << std::setprecision(2) << r.best_run.tflops << "| "
              << std::setw(8) << r.best_run.mfu << "| " << std::setw(8)
              << r.best_run.hbm_eff << "| " << std::setw(8)
              << r.best_run.grid_util << " |" << std::endl;
  }
  std::cout << "==============================================================="
               "==================================================="
            << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return 0;
}