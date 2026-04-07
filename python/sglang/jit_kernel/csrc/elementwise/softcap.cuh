#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr int kBlockSize = 256;
// Use arch-optimal vector width: 32 bytes on Blackwell+, 16 on older.
constexpr int kVecBytes = static_cast<int>(device::kMaxVecBytes);

// ---------------------------------------------------------------------------
// Parameter structs — passed via __grid_constant__ so every warp reads
// from the constant-cache broadcast path instead of per-lane register spills.
// ---------------------------------------------------------------------------
struct SoftcapInplaceParams {
  void* ptr;
  int64_t n_total;
  int64_t n_vec_elems;
  float c;
  float inv_c;
};

struct SoftcapOutFP32Params {
  const void* input;
  void* output;
  int64_t n_total;
  int64_t n_vec_elems;
  float c;
  float inv_c;
};

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------
SGL_DEVICE float softcap_compute(float x, float inv_c, float c) {
  return tanhf(x * inv_c) * c;
}

// ---------------------------------------------------------------------------
// Unified in-place kernel: vectorised body + scalar tail in one launch.
// ---------------------------------------------------------------------------
template <typename T, int kVecSize, bool kUsePDL>
__global__ __launch_bounds__(kBlockSize)
void softcap_inplace_kernel(const SoftcapInplaceParams __grid_constant__ p) {
  device::PDLWaitPrimary<kUsePDL>();

  using vec_t = device::AlignedVector<T, kVecSize>;
  T* ptr = static_cast<T*>(p.ptr);

  const int64_t n_vecs = p.n_vec_elems / kVecSize;
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t vid = tid; vid < n_vecs; vid += total_threads) {
    vec_t v;
    v.load(ptr, vid);
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      v[j] = device::cast<T>(softcap_compute(device::cast<fp32_t>(v[j]), p.inv_c, p.c));
    }
    v.store(ptr, vid);
  }

  for (int64_t idx = p.n_vec_elems + tid; idx < p.n_total; idx += total_threads) {
    ptr[idx] = device::cast<T>(softcap_compute(device::cast<fp32_t>(ptr[idx]), p.inv_c, p.c));
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

// ---------------------------------------------------------------------------
// Unified out-of-place kernel (input T -> output fp32)
// ---------------------------------------------------------------------------
template <typename T, int kVecSize, bool kUsePDL>
__global__ __launch_bounds__(kBlockSize)
void softcap_out_fp32_kernel(const SoftcapOutFP32Params __grid_constant__ p) {
  device::PDLWaitPrimary<kUsePDL>();

  using in_vec_t  = device::AlignedVector<T, kVecSize>;
  constexpr int kFP32VecSize = kVecBytes / sizeof(fp32_t);
  using out_vec_t = device::AlignedVector<fp32_t, kFP32VecSize>;

  const T*    input  = static_cast<const T*>(p.input);
  fp32_t*     output = static_cast<fp32_t*>(p.output);

  const int64_t n_vecs = p.n_vec_elems / kVecSize;
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

  constexpr int kOutChunks = kVecSize / kFP32VecSize;
  static_assert(kVecSize % kFP32VecSize == 0);

  for (int64_t vid = tid; vid < n_vecs; vid += total_threads) {
    in_vec_t iv;
    iv.load(input, vid);

    const int64_t out_base = vid * kVecSize / kFP32VecSize;
#pragma unroll
    for (int ch = 0; ch < kOutChunks; ++ch) {
      out_vec_t ov;
#pragma unroll
      for (int j = 0; j < kFP32VecSize; ++j) {
        ov[j] = softcap_compute(
            device::cast<fp32_t>(iv[ch * kFP32VecSize + j]), p.inv_c, p.c);
      }
      ov.store(output, out_base + ch);
    }
  }

  for (int64_t idx = p.n_vec_elems + tid; idx < p.n_total; idx += total_threads) {
    output[idx] = softcap_compute(device::cast<fp32_t>(input[idx]), p.inv_c, p.c);
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------
template <typename T, bool kUsePDL>
void softcap_inplace(tvm::ffi::TensorView tensor, float softcap_const) {
  using namespace host;
  SymbolicSize N = {"num_elements"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({N})
      .with_dtype<T>()
      .with_device(device_)
      .verify(tensor);
  const int64_t n = static_cast<int64_t>(N.unwrap());
  if (n == 0) return;

  constexpr int kVecSize = kVecBytes / sizeof(T);
  constexpr auto kernel = softcap_inplace_kernel<T, kVecSize, kUsePDL>;

  const int64_t n_vec_elems = (n / kVecSize) * kVecSize;
  const DLDevice dev = device_.unwrap();
  const auto params = SoftcapInplaceParams{
      .ptr = tensor.data_ptr(),
      .n_total = n,
      .n_vec_elems = n_vec_elems,
      .c = softcap_const,
      .inv_c = 1.0f / softcap_const,
  };

  static const uint32_t max_occ = runtime::get_blocks_per_sm(kernel, kBlockSize);
  static const uint32_t num_sm  = runtime::get_sm_count(dev.device_id);
  const size_t needed = std::max<size_t>(
      1, div_ceil(static_cast<size_t>(n / kVecSize), static_cast<size_t>(kBlockSize)));
  const size_t grid = std::min<size_t>(needed, static_cast<size_t>(max_occ * num_sm));

  LaunchKernel(static_cast<unsigned>(grid), kBlockSize, dev)
      .enable_pdl(kUsePDL)(kernel, params);
}

template <typename T, bool kUsePDL>
void softcap_out_fp32(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView output,
    float softcap_const) {
  using namespace host;
  SymbolicSize N = {"num_elements"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({N})
      .with_dtype<T>()
      .with_device(device_)
      .verify(input);
  TensorMatcher({N})
      .with_dtype<fp32_t>()
      .with_device(device_)
      .verify(output);
  const int64_t n = static_cast<int64_t>(N.unwrap());
  if (n == 0) return;

  constexpr int kVecSize = kVecBytes / sizeof(T);
  constexpr auto kernel = softcap_out_fp32_kernel<T, kVecSize, kUsePDL>;

  const int64_t n_vec_elems = (n / kVecSize) * kVecSize;
  const DLDevice dev = device_.unwrap();
  const auto params = SoftcapOutFP32Params{
      .input = input.data_ptr(),
      .output = output.data_ptr(),
      .n_total = n,
      .n_vec_elems = n_vec_elems,
      .c = softcap_const,
      .inv_c = 1.0f / softcap_const,
  };

  static const uint32_t max_occ = runtime::get_blocks_per_sm(kernel, kBlockSize);
  static const uint32_t num_sm  = runtime::get_sm_count(dev.device_id);
  const size_t needed = std::max<size_t>(
      1, div_ceil(static_cast<size_t>(n / kVecSize), static_cast<size_t>(kBlockSize)));
  const size_t grid = std::min<size_t>(needed, static_cast<size_t>(max_occ * num_sm));

  LaunchKernel(static_cast<unsigned>(grid), kBlockSize, dev)
      .enable_pdl(kUsePDL)(kernel, params);
}

}  // namespace
