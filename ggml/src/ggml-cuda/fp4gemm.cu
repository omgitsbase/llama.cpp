// cuBLASLt's block-scaled FP4 GEMM for NVFP4 and MXFP4 weights over several sequences. The activation quantization
// follows vLLM's csrc/quantization/fp4 (nvfp4_utils.cuh, Apache-2.0): a per-16 E4M3 scale under a per-tensor scale for
// NVFP4, a per-32 power-of-two E8M0 scale for MXFP4. The per-tensor scale is the checkpoint's input scale when it has one
// (ggml_cuda_fp4_set_input_scale), as vLLM takes it, else the activations' own maximum. FP8 E4M3 weights are at the end.

#include "fp4gemm.cuh"
#include "convert.cuh"
#include "unary.cuh"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <mutex>
#include <vector>
#include <tuple>
#include <unordered_map>

// the swizzled scale layout of block-scaled tensor-core GEMMs: [row/128][col/4] tiles of 32x4x4
static __device__ __forceinline__ int64_t fp4_sf_offset(const int64_t row, const int64_t col, const int64_t n_col_tiles) {
    return (((row >> 7)*n_col_tiles + (col >> 2)) << 9) | ((row & 31) << 4) | (((row >> 5) & 3) << 2) | (col & 3);
}

// ---------------------------------------------------------------------------------------------------------------------
// weights: the packed FP4 values in K order, two to a byte with the first in the low nibble, and the swizzled scales

template <int qk, int qk_sub, typename block_t>
static __global__ void fp4_weight_values(const block_t * __restrict__ x, uint8_t * __restrict__ out, const int64_t n, const int64_t k) {
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n*(k/2)) {
        return;
    }
    const int64_t row = idx / (k/2);
    const int64_t k0  = (idx % (k/2))*2;
    const block_t & b = x[row*(k/qk) + k0/qk];
    const int sub = (int) (k0%qk) / qk_sub;
    const int j   = (int) (k0%qk_sub);
    const uint8_t * qs = b.qs + sub*(qk_sub/2);
    // in a ggml block byte j holds element j (low nibble) and element j + qk_sub/2 (high nibble)
    uint8_t lo, hi;
    if (j < qk_sub/2) {
        lo = qs[j] & 0x0F;
        hi = qs[j + 1] & 0x0F;
    } else {
        lo = qs[j - qk_sub/2] >> 4;
        hi = qs[j - qk_sub/2 + 1] >> 4;
    }
    out[idx] = lo | (hi << 4);
}

static __global__ void fp4_weight_scales_nvfp4(const block_nvfp4 * __restrict__ x, uint8_t * __restrict__ sf,
        const int64_t n, const int64_t k, const int64_t n_col_tiles) {
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t n_g = k/QK_NVFP4_SUB;
    if (idx >= n*n_g) {
        return;
    }
    const int64_t row = idx / n_g;
    const int64_t g   = idx % n_g;
    uint8_t d = x[row*(k/QK_NVFP4) + g/(QK_NVFP4/QK_NVFP4_SUB)].d[g%(QK_NVFP4/QK_NVFP4_SUB)];
    // ggml reads 0x7F as a zero scale; as E4M3 it is a NaN
    sf[fp4_sf_offset(row, g, n_col_tiles)] = d == 0x7F ? 0 : d;
}

static __global__ void fp4_weight_scales_mxfp4(const block_mxfp4 * __restrict__ x, uint8_t * __restrict__ sf,
        const int64_t n, const int64_t k, const int64_t n_col_tiles) {
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t n_g = k/QK_MXFP4;
    if (idx >= n*n_g) {
        return;
    }
    const int64_t row = idx / n_g;
    const int64_t g   = idx % n_g;
    sf[fp4_sf_offset(row, g, n_col_tiles)] = x[row*n_g + g].e;
}

// ---------------------------------------------------------------------------------------------------------------------
// activations

// the largest |x|, as one partial per block: the quantizer reduces them, so nothing has to be zeroed first
static constexpr int fp4_absmax_blocks = 64;

static __global__ void fp4_absmax(const float * __restrict__ x, const int64_t ne, float * __restrict__ partial) {
    float m = 0.0f;
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < ne; i += (int64_t) gridDim.x*blockDim.x) {
        m = fmaxf(m, fabsf(x[i]));
    }
    m = warp_reduce_max(m);
    __shared__ float s[32];
    if (threadIdx.x % WARP_SIZE == 0) {
        s[threadIdx.x / WARP_SIZE] = m;
    }
    __syncthreads();
    if (threadIdx.x < WARP_SIZE) {
        m = threadIdx.x < blockDim.x / WARP_SIZE ? s[threadIdx.x] : 0.0f;
        m = warp_reduce_max(m);
        if (threadIdx.x == 0) {
            partial[blockIdx.x] = m;
        }
    }
}

#if defined(BLACKWELL_MMA_AVAILABLE)
static __device__ __forceinline__ uint32_t fp4_pack8(const float * v) {
    uint32_t r;
    asm volatile(
        "{\n"
        ".reg .b8 b0;\n"
        ".reg .b8 b1;\n"
        ".reg .b8 b2;\n"
        ".reg .b8 b3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
        "mov.b32 %0, {b0, b1, b2, b3};\n"
        "}"
        : "=r"(r)
        : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
    return r;
}
#endif // defined(BLACKWELL_MMA_AVAILABLE)

// NVFP4's tensor scale, a block's E4M3 scale and the factor its values take, and the GEMM's alpha, in correctly rounded
// operations: under fast math the same divisions came out an ulp apart in two kernels, so every kernel that quantizes
// an activation (fp4_quantize_act, fp4_geglu_quant_out) goes through these
static __device__ __forceinline__ float fp4_nv_tensor_scale(const float * in_s, const float amax) {
    return in_s ? __frcp_rn(in_s[0]) : (amax > 0.0f ? __fdiv_rn(448.0f*6.0f, amax) : 1.0f);
}

static __device__ __forceinline__ float fp4_nv_alpha(const float * w_s, const float tensor_scale) {
    return __fdiv_rn(w_s ? w_s[0] : 1.0f, tensor_scale);
}

static __device__ __forceinline__ uint8_t fp4_nv_block_scale(const float tensor_scale, const float vmax, float & out_scale) {
    const __nv_fp8_e4m3 s(__fmul_rn(tensor_scale, __fdiv_rn(vmax, 6.0f)));
    const float sv = float(s);
    out_scale = sv != 0.0f ? __fdiv_rn(tensor_scale, sv) : 0.0f;
    return s.__x;
}

// one thread per block of 16 (NVFP4) or 32 (MXFP4) values of a row; NVFP4 also writes alpha, the per-tensor scale
// The activations of one GEMM quantized to FP4 with one e4m3 scale per block of vec values. As vLLM's cvt_fp16_to_fp4
// maps it: 8 values a thread, vec/8 adjacent lanes to a block, the block's |max| exchanged between them; the same
// scale and the same packing as one lane per block gave, with vec/8 times the threads in flight.
// in_s: the checkpoint's static input scale (1/global scale) in place of the activation's |max|, as vLLM quantizes
// with input_global_scale; w_s: the weight's per-tensor scale, folded into alpha (vLLM's alpha = input * weight scale)
template <bool nvfp4>
static __global__ void fp4_quantize_act(const float * __restrict__ x, uint8_t * __restrict__ q, uint8_t * __restrict__ sf,
        const float * __restrict__ amax, float * __restrict__ alpha, const int64_t m, const int64_t k, const int64_t n_col_tiles,
        const int n_amax, const float * __restrict__ in_s, const float * __restrict__ w_s) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    constexpr int vec = nvfp4 ? 16 : 32;
    constexpr int tpg = vec/8; // lanes per block of vec values
    const int64_t n_g = k/vec;
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;

    __shared__ float tensor_amax;
    if constexpr (nvfp4) {
        if (in_s == nullptr && threadIdx.x < WARP_SIZE) {
            float a = 0.0f;
            for (int i = threadIdx.x; i < n_amax; i += WARP_SIZE) {
                a = fmaxf(a, amax[i]);
            }
            a = warp_reduce_max(a);
            if (threadIdx.x == 0) {
                tensor_amax = a;
            }
        }
        __syncthreads();
    }

    // every lane takes part in the exchange below, the ones past the end with nothing
    const bool    active = idx < m*n_g*tpg;
    const int64_t gi     = active ? idx/tpg : 0;
    const int     part   = (int) (idx % tpg);
    const int64_t row    = gi / n_g;
    const int64_t g      = gi % n_g;
    const float * xb = x + row*k + g*vec + part*8;

    float v[8];
    float vmax = 0.0f;
    if (active) {
        const float4 a = *(const float4 *) xb;
        const float4 b = *(const float4 *) (xb + 4);
        v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
        v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            vmax = fmaxf(vmax, fabsf(v[i]));
        }
    }
#pragma unroll
    for (int o = 1; o < tpg; o *= 2) {
        vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
    }
    if (!active) {
        return;
    }

    float out_scale;
    uint8_t sf_byte;
    if constexpr (nvfp4) {
        // vLLM: scale = tensor_scale * max/6 as E4M3, values times tensor_scale / scale
        const float tensor_scale = fp4_nv_tensor_scale(in_s, in_s != nullptr ? 0.0f : tensor_amax);
        if (idx == 0) {
            *alpha = fp4_nv_alpha(w_s, tensor_scale);
        }
        sf_byte = fp4_nv_block_scale(tensor_scale, vmax, out_scale);
    } else {
        if (idx == 0) {
            *alpha = w_s ? w_s[0] : 1.0f;
        }
        // vLLM's OCP MX rounding of the E8M0 exponent: round the max up from 1.75 * 2^e, then take two off for 6 = 2^2.58
        const uint32_t bits    = (__float_as_uint(vmax) + (1u << 21)) & 0xFF800000u;
        const uint32_t biased  = (bits >> 23) & 0xFFu;
        uint32_t e = biased > 2u ? biased - 2u : 0u;
        e = min(e, 254u);
        sf_byte = (uint8_t) e;
        const float sv = __uint_as_float(e << 23);
        out_scale = sv != 0.0f ? 1.0f / sv : 0.0f;
    }
    if (part == 0) {
        sf[fp4_sf_offset(row, g, n_col_tiles)] = sf_byte;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] *= out_scale;
    }
    uint32_t * qo = (uint32_t *) (q + row*(k/2) + g*(vec/2));
    qo[part] = fp4_pack8(v);
#else
    GGML_UNUSED_VARS(x, q, sf, amax, alpha, m, k, n_col_tiles, n_amax, in_s, w_s);
    NO_DEVICE_CODE;
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

// ---------------------------------------------------------------------------------------------------------------------

struct fp4_weight {
    uint8_t * q  = nullptr;
    uint8_t * sf = nullptr;
    size_t    q_bytes  = 0;
    size_t    sf_bytes = 0;
};

struct fp4_plan {
    bool ok = false;
    cudaDataType_t         d_type = CUDA_R_32F; // a 16-bit D is converted to dst
    cublasLtMatmulDesc_t   desc = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t   algo;
    // the heuristic's candidates; the first call of the shape outside a graph capture times them on its own operands
    // and keeps the fastest (fp4_tune), as vLLM's CUTLASS NVFP4 GEMM fixes its kernel per M bucket
    std::vector<cublasLtMatmulAlgo_t> cands;
    bool tuned = false;
};

static __global__ void fp4_to_f32(const void * __restrict__ x, float * __restrict__ y, const int64_t ne, const bool bf16) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < ne) {
        y[i] = bf16 ? __bfloat162float(((const nv_bfloat16 *) x)[i]) : __half2float(((const half *) x)[i]);
    }
}

// per backend context, since contexts run on their own streams (a drafter's next to its target's)
void ggml_cuda_graphs_invalidate(ggml_backend_cuda_context & ctx);

struct fp4_ctx {
    // the quantized activations (FP4 values, swizzled scales, the absmax partials and alpha) and the cuBLASLt workspace
    uint8_t * buf      = nullptr;
    size_t    buf_size = 0;

    // the activations in buf: products of the same tensor in a row (Q, K and V of one input) quantize it once
    uint64_t            gen      = 0;
    uint64_t            act_gen  = ~(uint64_t) 0;
    const ggml_tensor * act_t    = nullptr;
    const void        * act_data = nullptr;
    // the static input scale and the weight scale the quantization and alpha in buf were made with
    const float       * act_ins  = nullptr;
    const float       * act_ws   = nullptr;
};

// each NVFP4 weight's static input scale (its data -> the scale's, both on the device), see ggml_cuda_fp4_set_input_scale
static std::mutex fp4_in_s_mtx;
static std::unordered_map<const void *, const float *> fp4_in_s;

void ggml_cuda_fp4_set_input_scale(const void * w_data, const float * s_data) {
    std::lock_guard<std::mutex> lock(fp4_in_s_mtx);
    if (s_data) {
        fp4_in_s[w_data] = s_data;
    } else {
        fp4_in_s.erase(w_data);
    }
}

// GGML_CUDA_FP4_STATIC=0 quantizes every activation by its own |max| instead
static const float * fp4_input_scale_of(const void * w_data) {
    static const bool off = getenv("GGML_CUDA_FP4_STATIC") && atoi(getenv("GGML_CUDA_FP4_STATIC")) == 0;
    if (off) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(fp4_in_s_mtx);
    auto it = fp4_in_s.find(w_data);
    return it == fp4_in_s.end() ? nullptr : it->second;
}

static std::unordered_map<const void *, fp4_ctx> fp4_ctxs;

struct fp4_device {
    std::unordered_map<const void *, fp4_weight> weights;
    // q, k and v (gate and up) of one input repacked as one matrix, rows in that order
    std::map<std::vector<const void *>, fp4_weight> group_weights;
    // a part's data -> the group copy its GEMM reads
    std::unordered_map<const void *, const fp4_weight *> group_of;

    std::map<std::tuple<int64_t, int64_t, int64_t, int>, fp4_plan> plans;
    cublasLtHandle_t lt = nullptr;
    float * beta = nullptr; // a device zero for the device pointer mode
};

static std::mutex fp4_mtx;
static fp4_device fp4_devices[GGML_CUDA_MAX_DEVICES];

static constexpr size_t fp4_workspace_size = 32ull << 20;
// A buffer that grows synchronizes the stream and drops every captured graph (they read the old one): the first one is
// big enough for a 2048-row batch over a K of 10240 in either format, so a larger prompt batch later does not
static constexpr size_t gemm_buf_floor = 56ull << 20;

static int fp4_mode() {
    static const int mode = [] {
        const char * e = getenv("GGML_CUDA_FP4_GEMM");
        return e == nullptr ? 1 : atoi(e);
    }();
    return mode;
}

bool ggml_cuda_fp4_gemm_forced() {
    return fp4_mode() == 2;
}

bool ggml_cuda_fp4_gemm_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
    if (fp4_mode() == 0 || !blackwell_mma_available(cc)) {
        return false;
    }
    // NVFP4 only: cuBLASLt 13.6 has no algorithm for E2M1 with VEC32_UE8M0 scales (MXFP4) on sm_120
    if (src0->type != GGML_TYPE_NVFP4) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    // the tensor cores' FP4 operands come in rows of 32 values
    return src0->ne[0] % 64 == 0 && src0->ne[1] % 32 == 0;
}

static const fp4_weight * fp4_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * src0) {
    fp4_device & fd = fp4_devices[ctx.device];

    auto it = fd.weights.find(src0->data);
    if (it != fd.weights.end()) {
        return &it->second;
    }

    // a repack inside a captured graph would run again on every replay; the first evaluation of a shape is eager
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &capture));
    if (capture != cudaStreamCaptureStatusNone) {
        return nullptr;
    }

    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    const bool nvfp4 = src0->type == GGML_TYPE_NVFP4;
    const int64_t vec = nvfp4 ? QK_NVFP4_SUB : QK_MXFP4;
    const int64_t n_col_tiles = (k/vec + 3)/4;
    const size_t  sf_bytes = (size_t) ((n + 127)/128*128) * n_col_tiles*4;

    fp4_weight w;
    w.q_bytes  = (size_t) n*k/2;
    w.sf_bytes = sf_bytes;
    CUDA_CHECK(cudaMalloc(&w.q, (size_t) n*k/2));
    CUDA_CHECK(cudaMalloc(&w.sf, sf_bytes));

    cudaStream_t stream = ctx.stream();
    CUDA_CHECK(cudaMemsetAsync(w.sf, 0, sf_bytes, stream));
    const int threads = 256;
    const int64_t nv = n*(k/2);
    const int64_t ns = n*(k/vec);
    if (nvfp4) {
        fp4_weight_values<QK_NVFP4, QK_NVFP4_SUB, block_nvfp4><<<(nv + threads - 1)/threads, threads, 0, stream>>>((const block_nvfp4 *) src0->data, w.q, n, k);
        fp4_weight_scales_nvfp4<<<(ns + threads - 1)/threads, threads, 0, stream>>>((const block_nvfp4 *) src0->data, w.sf, n, k, n_col_tiles);
    } else {
        fp4_weight_values<QK_MXFP4, QK_MXFP4, block_mxfp4><<<(nv + threads - 1)/threads, threads, 0, stream>>>((const block_mxfp4 *) src0->data, w.q, n, k);
        fp4_weight_scales_mxfp4<<<(ns + threads - 1)/threads, threads, 0, stream>>>((const block_mxfp4 *) src0->data, w.sf, n, k, n_col_tiles);
    }
    CUDA_CHECK(cudaGetLastError());

    return &(fd.weights[src0->data] = w);
}

// the matrices of a group as one, each part's rows after the previous one's; every part but the last has a multiple
// of 128 rows, so the swizzled scale tiles of the parts follow each other too
static const fp4_weight * fp4_prepare_group(ggml_backend_cuda_context & ctx, const ggml_tensor * const * src0s, const int n_mats) {
    fp4_device & fd = fp4_devices[ctx.device];

    std::vector<const void *> key;
    for (int i = 0; i < n_mats; ++i) {
        key.push_back(src0s[i]->data);
    }
    auto it = fd.group_weights.find(key);
    if (it != fd.group_weights.end()) {
        return &it->second;
    }

    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &capture));
    if (capture != cudaStreamCaptureStatusNone) {
        return nullptr;
    }

    const int64_t k = src0s[0]->ne[0];
    int64_t n_total = 0;
    for (int i = 0; i < n_mats; ++i) {
        GGML_ASSERT(src0s[i]->type == GGML_TYPE_NVFP4 && src0s[i]->ne[0] == k);
        GGML_ASSERT(i == n_mats - 1 || src0s[i]->ne[1] % 128 == 0);
        n_total += src0s[i]->ne[1];
    }
    const int64_t vec = QK_NVFP4_SUB;
    const int64_t n_col_tiles = (k/vec + 3)/4;
    const size_t  sf_bytes = (size_t) ((n_total + 127)/128*128) * n_col_tiles*4;

    fp4_weight w;
    w.q_bytes  = (size_t) n_total*k/2;
    w.sf_bytes = sf_bytes;
    CUDA_CHECK(cudaMalloc(&w.q, (size_t) n_total*k/2));
    CUDA_CHECK(cudaMalloc(&w.sf, sf_bytes));

    cudaStream_t stream = ctx.stream();
    CUDA_CHECK(cudaMemsetAsync(w.sf, 0, sf_bytes, stream));
    const int threads = 256;
    int64_t n0 = 0;
    for (int i = 0; i < n_mats; ++i) {
        const int64_t n  = src0s[i]->ne[1];
        const int64_t nv = n*(k/2);
        const int64_t ns = n*(k/vec);
        uint8_t * q  = w.q  + n0*(k/2);
        uint8_t * sf = w.sf + (n0/128)*128*n_col_tiles*4;
        fp4_weight_values<QK_NVFP4, QK_NVFP4_SUB, block_nvfp4><<<(nv + threads - 1)/threads, threads, 0, stream>>>((const block_nvfp4 *) src0s[i]->data, q, n, k);
        fp4_weight_scales_nvfp4<<<(ns + threads - 1)/threads, threads, 0, stream>>>((const block_nvfp4 *) src0s[i]->data, sf, n, k, n_col_tiles);
        n0 += n;
    }
    CUDA_CHECK(cudaGetLastError());

    const fp4_weight * g = &(fd.group_weights[key] = w);
    for (const void * part : key) {
        fd.group_of[part] = g;
    }
    return g;
}

// the rows of a group's product to each part's output (row-major [m][n_i]), from a 32-bit or 16-bit D
// each part times its weight's per-tensor scale when it has one (s.p[i])
struct fp4_part_scales {
    const float * p[4];
};

template <typename T>
static __global__ void fp4_split_out(const T * __restrict__ d, const int64_t n_total, const int64_t m,
        float * __restrict__ o0, float * __restrict__ o1, float * __restrict__ o2, float * __restrict__ o3,
        const int64_t n0, const int64_t n1, const int64_t n2, const int64_t n3, const bool bf16, const fp4_part_scales s) {
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= m*n_total) {
        return;
    }
    const int64_t row = idx / n_total;
    int64_t       j   = idx % n_total;
    float * o;
    int64_t n;
    const float * sp;
    if (j < n0) {
        o = o0; n = n0; sp = s.p[0];
    } else if (j < n0 + n1) {
        j -= n0; o = o1; n = n1; sp = s.p[1];
    } else if (j < n0 + n1 + n2) {
        j -= n0 + n1; o = o2; n = n2; sp = s.p[2];
    } else {
        j -= n0 + n1 + n2; o = o3; n = n3; sp = s.p[3];
    }
    float v;
    if constexpr (sizeof(T) == 4) {
        v = ((const float *) d)[idx];
    } else {
        v = bf16 ? __bfloat162float(((const nv_bfloat16 *) d)[idx]) : __half2float(((const half *) d)[idx]);
    }
    o[row*n + j] = sp ? v*sp[0] : v;
}

// gelu(gate)*up from a gate and up group's product, one block per row: the same arithmetic as the GLU kernel over the
// parts fp4_split_out writes (unary.cu, op_gelu), and the row's |max| for the quantizer of the down projection
template <typename T, int block_size>
static __global__ void fp4_geglu_out(const T * __restrict__ d, const int64_t n_total, const int64_t gate_off,
        const int64_t up_off, const int64_t nc, float * __restrict__ dst, float * __restrict__ row_amax, const bool bf16,
        const float * __restrict__ gate_s, const float * __restrict__ up_s) {
    const int64_t row = blockIdx.x;
    const T * dr = d + row*n_total;
    float * o = dst + row*nc;
    auto ld = [&](const int64_t idx) -> float {
        if constexpr (sizeof(T) == 4) {
            return ((const float *) dr)[idx];
        } else {
            return bf16 ? __bfloat162float(((const nv_bfloat16 *) dr)[idx]) : __half2float(((const half *) dr)[idx]);
        }
    };
    const float gs = gate_s ? gate_s[0] : 1.0f;
    const float us = up_s   ? up_s[0]   : 1.0f;
    float amax = 0.0f;
    for (int64_t j = threadIdx.x; j < nc; j += block_size) {
        const float x = gate_s ? ld(gate_off + j)*gs : ld(gate_off + j);
        const float g = up_s   ? ld(up_off + j)*us   : ld(up_off + j);
        const float v = ggml_cuda_op_gelu_single(x) * g;
        o[j] = v;
        amax = fmaxf(amax, fabsf(v));
    }
    if (row_amax != nullptr) {
        amax = warp_reduce_max(amax);
        __shared__ float sm[32];
        if (threadIdx.x % WARP_SIZE == 0) {
            sm[threadIdx.x / WARP_SIZE] = amax;
        }
        __syncthreads();
        if (threadIdx.x < WARP_SIZE) {
            amax = threadIdx.x < block_size / WARP_SIZE ? sm[threadIdx.x] : 0.0f;
            amax = warp_reduce_max(amax);
            if (threadIdx.x == 0) {
                row_amax[row] = amax;
            }
        }
    }
}

// fp4_geglu_out with the down projection's activation quantized in it, as vLLM fuses the activation and its NVFP4
// quant: 8 adjacent columns a lane, two lanes to a block of 16, the down weight's static input scale; the values, scales
// and packing fp4_quantize_act gives the F32 row (which is still written for any other reader)
template <typename T, int block_size>
static __global__ void fp4_geglu_quant_out(const T * __restrict__ d, const int64_t n_total, const int64_t gate_off,
        const int64_t up_off, const int64_t nc, float * __restrict__ dst, const bool bf16,
        const float * __restrict__ gate_s, const float * __restrict__ up_s,
        uint8_t * __restrict__ q, uint8_t * __restrict__ sf, float * __restrict__ alpha, const int64_t n_col_tiles,
        const float * __restrict__ in_s, const float * __restrict__ w_s) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int64_t row = blockIdx.x;
    const T * dr = d + row*n_total;
    float * o = dst + row*nc;
    const float gs = gate_s ? gate_s[0] : 1.0f;
    const float us = up_s   ? up_s[0]   : 1.0f;
    const float tensor_scale = fp4_nv_tensor_scale(in_s, 0.0f);
    if (row == 0 && threadIdx.x == 0) {
        *alpha = fp4_nv_alpha(w_s, tensor_scale);
    }
    const int64_t n_lanes = nc/8;
    // every lane runs every round, the exchange below needs both of a block's lanes
    for (int64_t l0 = 0; l0 < n_lanes; l0 += block_size) {
        const int64_t l      = l0 + threadIdx.x;
        const bool    active = l < n_lanes;
        // set on every path: an array the asm below reads, written on one path only, reached it as zeros
        float v[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        float vmax = 0.0f;
        if (active) {
            // the lane's 8 gate and 8 up values in 16-byte loads (the rows and the parts' offsets are 16-byte aligned)
            float xg[8];
            float xu[8];
            if constexpr (sizeof(T) == 4) {
                const float4 * pg = (const float4 *) ((const float *) dr + gate_off + l*8);
                const float4 * pu = (const float4 *) ((const float *) dr + up_off   + l*8);
                const float4 g0 = pg[0], g1 = pg[1], u0 = pu[0], u1 = pu[1];
                xg[0] = g0.x; xg[1] = g0.y; xg[2] = g0.z; xg[3] = g0.w; xg[4] = g1.x; xg[5] = g1.y; xg[6] = g1.z; xg[7] = g1.w;
                xu[0] = u0.x; xu[1] = u0.y; xu[2] = u0.z; xu[3] = u0.w; xu[4] = u1.x; xu[5] = u1.y; xu[6] = u1.z; xu[7] = u1.w;
            } else {
                const uint4 rg = *(const uint4 *) ((const uint16_t *) dr + gate_off + l*8);
                const uint4 ru = *(const uint4 *) ((const uint16_t *) dr + up_off   + l*8);
                const uint16_t * hg = (const uint16_t *) &rg;
                const uint16_t * hu = (const uint16_t *) &ru;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    xg[i] = bf16 ? __bfloat162float(((const nv_bfloat16 *) hg)[i]) : __half2float(((const half *) hg)[i]);
                    xu[i] = bf16 ? __bfloat162float(((const nv_bfloat16 *) hu)[i]) : __half2float(((const half *) hu)[i]);
                }
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float x = gate_s ? xg[i]*gs : xg[i];
                const float g = up_s   ? xu[i]*us : xu[i];
                v[i] = ggml_cuda_op_gelu_single(x) * g;
                vmax = fmaxf(vmax, fabsf(v[i]));
            }
            float4 * po = (float4 *) (o + l*8);
            po[0] = make_float4(v[0], v[1], v[2], v[3]);
            po[1] = make_float4(v[4], v[5], v[6], v[7]);
        }
        vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, 1));
        if (active) {
            const int64_t g    = l/2;
            const int     part = (int) (l % 2);
            float out_scale;
            const uint8_t sf_byte = fp4_nv_block_scale(tensor_scale, vmax, out_scale);
            if (part == 0) {
                sf[fp4_sf_offset(row, g, n_col_tiles)] = sf_byte;
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                v[i] *= out_scale;
            }
            ((uint32_t *) (q + row*(nc/2) + g*8))[part] = fp4_pack8(v);
        }
    }
#else
    GGML_UNUSED_VARS(d, n_total, gate_off, up_off, nc, dst, bf16, gate_s, up_s, q, sf, alpha, n_col_tiles, in_s, w_s);
    NO_DEVICE_CODE;
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

// a cuBLASLt algorithm's split-K factor (1: none, no reduce kernel after it)
static int lt_algo_split_k(const cublasLtMatmulAlgo_t & algo) {
    int split = 1;
    size_t sz = 0;
    if (cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split, sizeof(split), &sz) != CUBLAS_STATUS_SUCCESS) {
        return 1;
    }
    return split;
}

// D [n x m] (column-major, ld n) = A^T B: A the weights, K x n column-major; B the activations, K x m column-major.
// d16: a BF16 D first, for a product the next kernel reads once (the GEGLU after gate and up: vLLM's BF16 activations,
// half the bytes through memory), else an F32 one
static fp4_plan * fp4_get_plan(fp4_device & fd, const int64_t m, const int64_t n, const int64_t k, const bool nvfp4, const bool d16 = false) {
    auto key = std::make_tuple(m, n, k, (int) nvfp4 + 2*(int) d16);
    auto it = fd.plans.find(key);
    if (it != fd.plans.end()) {
        return &it->second;
    }
    fp4_plan & p = fd.plans[key];

    CUBLAS_CHECK(cublasLtMatmulDescCreate(&p.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    const cublasLtMatmulMatrixScale_t mode = nvfp4 ? CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 : CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode)));
    const cublasLtPointerMode_t pm = CUBLASLT_POINTER_MODE_DEVICE;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pm, sizeof(pm)));

    // the heuristic refuses a block-scaled description without scale pointers; the real ones are set per call
    const void * placeholder = fd.beta;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &placeholder, sizeof(placeholder)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &placeholder, sizeof(placeholder)));

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_4F_E2M1, k, n, k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_4F_E2M1, k, m, k));

    cublasLtMatmulPreference_t pref = nullptr;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    const size_t ws = fp4_workspace_size;
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
    cublasStatus_t st = CUBLAS_STATUS_NOT_SUPPORTED;
    const std::initializer_list<cudaDataType_t> d_types_f32 = {CUDA_R_32F, CUDA_R_16BF, CUDA_R_16F};
    const std::initializer_list<cudaDataType_t> d_types_16  = {CUDA_R_16BF, CUDA_R_32F, CUDA_R_16F};
    for (cudaDataType_t d_type : d16 ? d_types_16 : d_types_f32) {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.c, d_type, n, m, n));
        cublasLtMatmulHeuristicResult_t res[64] = {};
        int n_res = 0;
        st = cublasLtMatmulAlgoGetHeuristic(fd.lt, p.desc, p.a, p.b, p.c, p.c, pref, 64, res, &n_res);
        if (st == CUBLAS_STATUS_SUCCESS && n_res > 0) {
            p.ok     = true;
            p.d_type = d_type;
            p.algo   = res[0].algo;
            for (int i = 0; i < n_res; ++i) {
                if (res[i].state == CUBLAS_STATUS_SUCCESS) {
                    p.cands.push_back(res[i].algo);
                }
            }
            break;
        }
        cublasLtMatrixLayoutDestroy(p.c);
        p.c = nullptr;
    }
    cublasLtMatmulPreferenceDestroy(pref);
    if (!p.ok) {
        GGML_LOG_WARN("%s: no cuBLASLt FP4 algorithm for m=%lld n=%lld k=%lld (status %d)\n", __func__,
                (long long) m, (long long) n, (long long) k, (int) st);
    }
    return &p;
}

// The heuristic's first cuBLASLt algorithm is often not the fastest for a few hundred rows or less (Gemma 4 E4B at 128
// rows, weights cold: the FP8 o projection 6.5 -> 5.3 us, the FP4 down 18.2 -> 15.6 us). The first call of a shape outside
// a capture times the heuristic's candidates and their split-K variants on the call's own operands, each run after a
// read of more than L2 holds, so the weights come from memory as in a forward, and keeps the fastest (vLLM's CUTLASS
// kernels are fixed per M bucket the same way, offline). GGML_CUDA_GEMM_TUNE_CACHE names a file the choices are kept in:
// a later run on the same GPU and cuBLASLt takes the same kernels, so its results are the same, without timing.
// GGML_CUDA_GEMM_TUNE=0 keeps the heuristic's first.
static bool lt_tune_on() {
    static const bool on = !(getenv("GGML_CUDA_GEMM_TUNE") && atoi(getenv("GGML_CUDA_GEMM_TUNE")) == 0);
    return on;
}

// the rows a choice is made for: up to a power of two (vLLM's M buckets), so a batch of a new size takes its bucket's
// kernel instead of a timing of its own
static int64_t lt_tune_bucket(const int64_t m) {
    int64_t b = 16;
    while (b < m && b < 4096) {
        b *= 2;
    }
    return m <= 8 ? m : b;
}

struct lt_tune_cache {
    bool loaded = false;
    std::string header;
    std::map<std::string, cublasLtMatmulAlgo_t> algos;
};
static lt_tune_cache lt_cache;

static std::string lt_tune_header(const int device) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    char buf[512];
    snprintf(buf, sizeof(buf), "gemm-tune 1 | %s | cublasLt %zu", prop.name, cublasLtGetVersion());
    return buf;
}

static void lt_tune_load(const int device) {
    if (lt_cache.loaded) {
        return;
    }
    lt_cache.loaded = true;
    lt_cache.header = lt_tune_header(device);
    const char * path = getenv("GGML_CUDA_GEMM_TUNE_CACHE");
    if (path == nullptr) {
        return;
    }
    FILE * f = fopen(path, "r");
    if (f == nullptr) {
        return;
    }
    char line[1024];
    bool ok = fgets(line, sizeof(line), f) != nullptr && lt_cache.header == std::string(line).substr(0, std::string(line).find_last_not_of("\r\n") + 1);
    while (ok && fgets(line, sizeof(line), f)) {
        char key[256];
        unsigned long long w[8];
        if (sscanf(line, "%255[^=]= %llx %llx %llx %llx %llx %llx %llx %llx", key, &w[0], &w[1], &w[2], &w[3], &w[4], &w[5], &w[6], &w[7]) == 9) {
            cublasLtMatmulAlgo_t a;
            static_assert(sizeof(a) == sizeof(w), "cublasLtMatmulAlgo_t layout");
            memcpy(&a, w, sizeof(a));
            std::string k = key;
            k.erase(k.find_last_not_of(' ') + 1);
            lt_cache.algos[k] = a;
        }
    }
    fclose(f);
}

static void lt_tune_store(const std::string & key, const cublasLtMatmulAlgo_t & a) {
    lt_cache.algos[key] = a;
    const char * path = getenv("GGML_CUDA_GEMM_TUNE_CACHE");
    if (path == nullptr) {
        return;
    }
    FILE * f = fopen(path, "w");
    if (f == nullptr) {
        return;
    }
    fprintf(f, "%s\n", lt_cache.header.c_str());
    for (const auto & kv : lt_cache.algos) {
        unsigned long long w[8];
        memcpy(w, &kv.second, sizeof(w));
        fprintf(f, "%s = %llx %llx %llx %llx %llx %llx %llx %llx\n", kv.first.c_str(), w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
    }
    fclose(f);
}

// a read of the whole buffer, its sum kept so the loads stay: what was in L2 before is gone after
static __global__ void k_l2_flush(const float4 * __restrict__ x, const size_t n, float * __restrict__ sink) {
    float acc = 0.0f;
    for (size_t i = blockIdx.x*(size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x*blockDim.x) {
        const float4 v = x[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 1234.5f) {
        *sink = acc;
    }
}

// The fastest of the candidates and their split-K variants for this call (key: the shape and type), from the cache or
// timed now; false when nothing better than cands[0] can be chosen now (a capture is running: the caller keeps its algo
// and asks again on an eager call)
template <typename F>
static bool lt_tune_select(cublasLtHandle_t lt, cublasLtMatmulDesc_t desc, cublasLtMatrixLayout_t la, cublasLtMatrixLayout_t lb,
        cublasLtMatrixLayout_t lc, const std::vector<cublasLtMatmulAlgo_t> & cands, const std::string & key, const int device,
        cudaStream_t stream, F && run, cublasLtMatmulAlgo_t & out, const size_t ws_size = fp4_workspace_size) {
    lt_tune_load(device);
    auto ic = lt_cache.algos.find(key);
    if (ic != lt_cache.algos.end()) {
        cublasLtMatmulHeuristicResult_t chk = {};
        if (cublasLtMatmulAlgoCheck(lt, desc, la, lb, lc, lc, &ic->second, &chk) == CUBLAS_STATUS_SUCCESS && chk.workspaceSize <= ws_size) {
            out = ic->second;
        }
        // else the caller's own choice (out as passed) stays
        return true;
    }
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) {
        return false;
    }

    std::vector<cublasLtMatmulAlgo_t> all;
    auto add = [&](const cublasLtMatmulAlgo_t & a) {
        cublasLtMatmulHeuristicResult_t chk = {};
        if (cublasLtMatmulAlgoCheck(lt, desc, la, lb, lc, lc, &a, &chk) != CUBLAS_STATUS_SUCCESS || chk.workspaceSize > ws_size) {
            return;
        }
        for (const auto & o : all) {
            if (memcmp(&o, &a, sizeof(a)) == 0) {
                return;
            }
        }
        all.push_back(a);
    };
    for (const auto & c : cands) {
        add(c);
    }
    // split-K variants of the heuristic's first few; the in-place reduction keeps its order with counters (no
    // separate reduce kernel)
    for (size_t ic = 0; ic < cands.size() && ic < 16; ++ic) {
        const auto & c = cands[ic];
        for (int split : { 2, 3, 4, 6, 8 }) {
            for (int red : { (int) CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE, (int) CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE,
                             (int) CUBLASLT_REDUCTION_SCHEME_INPLACE }) {
                cublasLtMatmulAlgo_t a = c;
                if (cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split, sizeof(split)) == CUBLAS_STATUS_SUCCESS &&
                        cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof(red)) == CUBLAS_STATUS_SUCCESS) {
                    add(a);
                }
            }
        }
    }
    if (all.empty()) {
        return false;
    }

    const size_t flush_bytes = 160ull << 20;
    void * flush = nullptr;
    float * sink = nullptr;
    CUDA_CHECK(cudaMalloc(&flush, flush_bytes + 256));
    CUDA_CHECK(cudaMemsetAsync(flush, 0, flush_bytes + 256, stream));
    sink = (float *) ((char *) flush + flush_bytes);
    constexpr int n_rep = 3;
    std::vector<cudaEvent_t> ev(2*n_rep);
    for (auto & e : ev) {
        CUDA_CHECK(cudaEventCreate(&e));
    }
    float best_us = 1e30f;
    int   best    = 0;
    for (size_t i = 0; i < all.size(); ++i) {
        bool ok = true;
        for (int r = 0; r < n_rep && ok; ++r) {
            k_l2_flush<<<1024, 256, 0, stream>>>((const float4 *) flush, flush_bytes/sizeof(float4), sink);
            CUDA_CHECK(cudaEventRecord(ev[2*r], stream));
            ok = run(all[i]) == CUBLAS_STATUS_SUCCESS;
            CUDA_CHECK(cudaEventRecord(ev[2*r + 1], stream));
        }
        CUDA_CHECK(cudaEventSynchronize(ev[2*n_rep - 1]));
        if (!ok) {
            cudaGetLastError();
            continue;
        }
        float t[n_rep];
        for (int r = 0; r < n_rep; ++r) {
            CUDA_CHECK(cudaEventElapsedTime(&t[r], ev[2*r], ev[2*r + 1]));
        }
        std::sort(t, t + n_rep);
        const float us = t[n_rep/2] * 1000.0f;
        if (us < best_us) {
            best_us = us;
            best    = (int) i;
        }
    }
    for (auto & e : ev) {
        CUDA_CHECK(cudaEventDestroy(e));
    }
    CUDA_CHECK(cudaFree(flush));
    out = all[best];
    lt_tune_store(key, out);
    static const bool say = getenv("LLAMA_ROUND_TIMING") != nullptr;
    if (say) {
        fprintf(stderr, "[gemm-tune] %s: %zu candidates, best #%d %.2f us (split %d)\n", key.c_str(), all.size(), best, best_us, lt_algo_split_k(out));
    }
    return true;
}

static void fp4_tune(fp4_plan * plan, fp4_device & fd, const void * alpha, const void * a, const void * b, void * d,
        void * ws, cudaStream_t stream, const int64_t m, const int64_t n, const int64_t k, const int device) {
    if (plan->tuned || !lt_tune_on() || plan->cands.empty()) {
        plan->tuned = true;
        return;
    }
    char key[128];
    snprintf(key, sizeof(key), "fp4 %lldx%lldx%lld d%d", (long long) lt_tune_bucket(m), (long long) n, (long long) k, (int) plan->d_type);
    auto run = [&](const cublasLtMatmulAlgo_t & al) {
        return cublasLtMatmul(fd.lt, plan->desc, alpha, a, plan->a, b, plan->b, fd.beta, d, plan->c, d, plan->c, &al, ws, fp4_workspace_size, stream);
    };
    cublasLtMatmulAlgo_t al = plan->algo;
    if (lt_tune_select(fd.lt, plan->desc, plan->a, plan->b, plan->c, plan->cands, key, device, stream, run, al)) {
        plan->algo  = al;
        plan->tuned = true;
    }
}

bool ggml_cuda_fp4_gemm(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const float * w_s) {
    const float * in_s = src0->type == GGML_TYPE_NVFP4 ? fp4_input_scale_of(src0->data) : nullptr;
    std::lock_guard<std::mutex> lock(fp4_mtx);

    fp4_device & fd = fp4_devices[ctx.device];
    if (fd.lt == nullptr) {
        CUBLAS_CHECK(cublasLtCreate(&fd.lt));
        CUDA_CHECK(cudaMalloc(&fd.beta, sizeof(float)));
        CUDA_CHECK(cudaMemset(fd.beta, 0, sizeof(float)));
    }

    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    const int64_t m = src1->ne[1];
    const bool nvfp4 = src0->type == GGML_TYPE_NVFP4;

    fp4_plan * plan = fp4_get_plan(fd, m, n, k, nvfp4);
    if (!plan->ok) {
        return false;
    }
    const fp4_weight * w = fp4_prepare(ctx, src0);
    if (w == nullptr) {
        return false;
    }

    cudaStream_t stream = ctx.stream();
    const int64_t vec = nvfp4 ? QK_NVFP4_SUB : QK_MXFP4;
    const int64_t n_col_tiles = (k/vec + 3)/4;
    const int64_t sf_bytes = (m + 127)/128*128 * n_col_tiles*4;

    // every operand cuBLASLt reads on a 256-byte boundary
    auto align = [](const size_t x) { return (x + 255) & ~(size_t) 255; };
    const size_t off_sf = align((size_t) m*k/2);
    const size_t off_s  = off_sf + align((size_t) sf_bytes);
    const size_t off_ws = off_s + align(fp4_absmax_blocks*sizeof(float)) + 256;
    const size_t need   = off_ws + fp4_workspace_size;
    fp4_ctx & fc = fp4_ctxs[&ctx];
    if (fc.buf_size < need) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
        if (capture != cudaStreamCaptureStatusNone) {
            return false;
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (fc.buf) {
            CUDA_CHECK(cudaFree(fc.buf));
        }
        fc.buf_size = std::max(need + need/4, gemm_buf_floor);
        CUDA_CHECK(cudaMalloc(&fc.buf, fc.buf_size));
        fc.act_gen = ~(uint64_t) 0;
        // graphs captured before read the freed buffer
        ggml_cuda_graphs_invalidate(ctx);
    }
    uint8_t * q_act  = fc.buf;
    uint8_t * sf_act = fc.buf + off_sf;
    float   * amax   = (float *) (fc.buf + off_s);
    float   * alpha  = (float *) (fc.buf + off_ws - 256);
    uint8_t * ws     = fc.buf + off_ws;

    const int threads = 256;
    const bool fresh = !(fc.act_gen == fc.gen && fc.act_t == src1 && fc.act_data == src1->data && fc.act_ins == in_s && fc.act_ws == w_s);
    if (fresh) {
        const int64_t n_g = m*(k/vec);
        if (nvfp4) {
            // a static input scale, else the producer's row maxes when it left them, else a pass over the activation
            const float * row_amax = in_s ? nullptr : ggml_cuda_row_amax_find(ctx, src1, src1->data, m);
            if (!in_s && !row_amax) {
                fp4_absmax<<<fp4_absmax_blocks, threads, 0, stream>>>((const float *) src1->data, m*k, amax);
            }
            fp4_quantize_act<true><<<(n_g*(vec/8) + threads - 1)/threads, threads, 0, stream>>>((const float *) src1->data, q_act, sf_act,
                    row_amax ? row_amax : amax, alpha, m, k, n_col_tiles, row_amax ? (int) m : fp4_absmax_blocks, in_s, w_s);
        } else {
            fp4_quantize_act<false><<<(n_g*(vec/8) + threads - 1)/threads, threads, 0, stream>>>((const float *) src1->data, q_act, sf_act, amax, alpha, m, k, n_col_tiles, 0, nullptr, w_s);
        }
        CUDA_CHECK(cudaGetLastError());
        fc.act_gen  = fc.gen;
        fc.act_t    = src1;
        fc.act_data = src1->data;
        fc.act_ins  = in_s;
        fc.act_ws   = w_s;
    }

    const void * a_sf = w->sf;
    const void * b_sf = sf_act;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan->desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_sf, sizeof(a_sf)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan->desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_sf, sizeof(b_sf)));

    ggml_cuda_pool_alloc<uint16_t> d16(ctx.pool());
    void * d = dst->data;
    if (plan->d_type != CUDA_R_32F) {
        d = d16.alloc(m*n);
    }

    fp4_tune(plan, fd, alpha, w->q, q_act, d, ws, stream, m, n, k, ctx.device);
    CUBLAS_CHECK(cublasLtMatmul(fd.lt, plan->desc, alpha, w->q, plan->a, q_act, plan->b, fd.beta,
            d, plan->c, d, plan->c, &plan->algo, ws, fp4_workspace_size, stream));
    if (plan->d_type != CUDA_R_32F) {
        fp4_to_f32<<<(m*n + threads - 1)/threads, threads, 0, stream>>>(d16.get(), (float *) dst->data, m*n, plan->d_type == CUDA_R_16BF);
    }
    return true;
}

static bool fp4_gemm_group_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * const * src0s, const int n_mats, const ggml_tensor * src1,
        ggml_tensor * const * dsts, const int gate_part, ggml_tensor * glu, const float * const * w_s,
        const ggml_tensor * down_w = nullptr, const float * down_ws = nullptr) {
    // the GEGLU's reader, an NVFP4 down projection with a static input scale, has its activation quantized with it
    const float * dn_in_s = glu && down_w && down_w->type == GGML_TYPE_NVFP4 ? fp4_input_scale_of(down_w->data) : nullptr;
    // one static input scale for the whole group (one quantization of the activation), else each activation's |max|
    const float * in_s = fp4_input_scale_of(src0s[0]->data);
    for (int i = 1; i < n_mats && in_s; ++i) {
        if (fp4_input_scale_of(src0s[i]->data) != in_s) {
            in_s = nullptr;
        }
    }
    fp4_part_scales ps = { { nullptr, nullptr, nullptr, nullptr } };
    for (int i = 0; i < n_mats && w_s; ++i) {
        ps.p[i] = w_s[i];
    }
    std::lock_guard<std::mutex> lock(fp4_mtx);

    fp4_device & fd = fp4_devices[ctx.device];
    if (fd.lt == nullptr) {
        CUBLAS_CHECK(cublasLtCreate(&fd.lt));
        CUDA_CHECK(cudaMalloc(&fd.beta, sizeof(float)));
        CUDA_CHECK(cudaMemset(fd.beta, 0, sizeof(float)));
    }

    const int64_t k = src0s[0]->ne[0];
    const int64_t m = src1->ne[1];
    int64_t n = 0;
    int64_t np[4] = {0, 0, 0, 0};
    for (int i = 0; i < n_mats; ++i) {
        np[i] = src0s[i]->ne[1];
        n += np[i];
    }
    const bool nvfp4 = true;

    // the GEGLU reads the gate and up products once: a BF16 product. GGML_CUDA_FP4_GEGLU_F32=1 keeps F32
    static const bool geglu_f32 = getenv("GGML_CUDA_FP4_GEGLU_F32") != nullptr;
    fp4_plan * plan = fp4_get_plan(fd, m, n, k, nvfp4, glu != nullptr && !geglu_f32);
    if (!plan->ok) {
        return false;
    }
    const fp4_weight * w = fp4_prepare_group(ctx, src0s, n_mats);
    if (w == nullptr) {
        return false;
    }

    cudaStream_t stream = ctx.stream();
    const int64_t vec = QK_NVFP4_SUB;
    const int64_t n_col_tiles = (k/vec + 3)/4;
    const int64_t sf_bytes = (m + 127)/128*128 * n_col_tiles*4;

    auto align = [](const size_t x) { return (x + 255) & ~(size_t) 255; };
    const size_t off_sf = align((size_t) m*k/2);
    const size_t off_s  = off_sf + align((size_t) sf_bytes);
    const size_t off_ws = off_s + align(fp4_absmax_blocks*sizeof(float)) + 256;
    const size_t need   = off_ws + fp4_workspace_size;
    fp4_ctx & fc = fp4_ctxs[&ctx];
    if (fc.buf_size < need) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
        if (capture != cudaStreamCaptureStatusNone) {
            return false;
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (fc.buf) {
            CUDA_CHECK(cudaFree(fc.buf));
        }
        fc.buf_size = std::max(need + need/4, gemm_buf_floor);
        CUDA_CHECK(cudaMalloc(&fc.buf, fc.buf_size));
        fc.act_gen = ~(uint64_t) 0;
        // graphs captured before read the freed buffer
        ggml_cuda_graphs_invalidate(ctx);
    }
    uint8_t * q_act  = fc.buf;
    uint8_t * sf_act = fc.buf + off_sf;
    float   * amax   = (float *) (fc.buf + off_s);
    float   * alpha  = (float *) (fc.buf + off_ws - 256);
    uint8_t * ws     = fc.buf + off_ws;

    const int threads = 256;
    const bool fresh = !(fc.act_gen == fc.gen && fc.act_t == src1 && fc.act_data == src1->data && fc.act_ins == in_s && fc.act_ws == nullptr);
    if (fresh) {
        const int64_t n_g = m*(k/vec);
        const float * row_amax = in_s ? nullptr : ggml_cuda_row_amax_find(ctx, src1, src1->data, m);
        if (!in_s && !row_amax) {
            fp4_absmax<<<fp4_absmax_blocks, threads, 0, stream>>>((const float *) src1->data, m*k, amax);
        }
        fp4_quantize_act<true><<<(n_g*(vec/8) + threads - 1)/threads, threads, 0, stream>>>((const float *) src1->data, q_act, sf_act,
                row_amax ? row_amax : amax, alpha, m, k, n_col_tiles, row_amax ? (int) m : fp4_absmax_blocks, in_s, nullptr);
        CUDA_CHECK(cudaGetLastError());
        fc.act_gen  = fc.gen;
        fc.act_t    = src1;
        fc.act_data = src1->data;
        fc.act_ins  = in_s;
        fc.act_ws   = nullptr;
    }

    const void * a_sf = w->sf;
    const void * b_sf = sf_act;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan->desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_sf, sizeof(a_sf)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan->desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_sf, sizeof(b_sf)));

    // the product of all parts, then each part's rows to its output
    const size_t esz = plan->d_type == CUDA_R_32F ? 4 : 2;
    ggml_cuda_pool_alloc<uint8_t> dtmp(ctx.pool(), (size_t) m*n*esz);
    fp4_tune(plan, fd, alpha, w->q, q_act, dtmp.get(), ws, stream, m, n, k, ctx.device);
    CUBLAS_CHECK(cublasLtMatmul(fd.lt, plan->desc, alpha, w->q, plan->a, q_act, plan->b, fd.beta,
            dtmp.get(), plan->c, dtmp.get(), plan->c, &plan->algo, ws, fp4_workspace_size, stream));
    if (glu != nullptr) {
        // gelu(gate)*up per row; the row maxes are registered for the down projection's quantizer
        const int64_t nc      = glu->ne[0];
        const int64_t gate_off = gate_part == 0 ? 0 : np[0];
        const int64_t up_off   = gate_part == 0 ? np[0] : 0;
        constexpr int bs = 1024;
        const float * gate_s = ps.p[gate_part];
        const float * up_s   = ps.p[1 - gate_part];
        // the down projection's quantization, at the offsets its GEMM computes for its own m and k (ggml_cuda_fp4_gemm):
        // that GEMM finds it in buf and skips its own
        const int64_t dn_tiles = (nc/vec + 3)/4;
        const size_t  dn_sf    = align((size_t) m*nc/2);
        const size_t  dn_ws    = dn_sf + align((size_t) ((m + 127)/128*128 * dn_tiles*4)) + align(fp4_absmax_blocks*sizeof(float)) + 256;
        // its 16-byte loads and stores: the product's rows and parts, and the GEGLU's rows, on 16-byte boundaries
        const bool vec_ok = n % 8 == 0 && gate_off % 8 == 0 && up_off % 8 == 0 && (uintptr_t) glu->data % 16 == 0;
        if (dn_in_s && vec_ok && nc % 16 == 0 && glu->ne[2] == 1 && glu->ne[3] == 1 && fc.buf_size >= dn_ws + fp4_workspace_size) {
            uint8_t * dn_q     = fc.buf;
            uint8_t * dn_sfp   = fc.buf + dn_sf;
            float   * dn_alpha = (float *) (fc.buf + dn_ws - 256);
            if (plan->d_type == CUDA_R_32F) {
                fp4_geglu_quant_out<float, bs><<<m, bs, 0, stream>>>((const float *) dtmp.get(), n, gate_off, up_off, nc, (float *) glu->data,
                        false, gate_s, up_s, dn_q, dn_sfp, dn_alpha, dn_tiles, dn_in_s, down_ws);
            } else {
                fp4_geglu_quant_out<uint16_t, bs><<<m, bs, 0, stream>>>((const uint16_t *) dtmp.get(), n, gate_off, up_off, nc, (float *) glu->data,
                        plan->d_type == CUDA_R_16BF, gate_s, up_s, dn_q, dn_sfp, dn_alpha, dn_tiles, dn_in_s, down_ws);
            }
            CUDA_CHECK(cudaGetLastError());
            fc.act_gen  = fc.gen;
            fc.act_t    = glu;
            fc.act_data = glu->data;
            fc.act_ins  = dn_in_s;
            fc.act_ws   = down_ws;
            return true;
        }
        float * row_amax = glu->ne[2] == 1 && glu->ne[3] == 1 ? ggml_cuda_row_amax_begin(ctx, glu, glu->data, m) : nullptr;
        if (plan->d_type == CUDA_R_32F) {
            fp4_geglu_out<float, bs><<<m, bs, 0, stream>>>((const float *) dtmp.get(), n, gate_off, up_off, nc, (float *) glu->data, row_amax, false, gate_s, up_s);
        } else {
            fp4_geglu_out<uint16_t, bs><<<m, bs, 0, stream>>>((const uint16_t *) dtmp.get(), n, gate_off, up_off, nc, (float *) glu->data, row_amax, plan->d_type == CUDA_R_16BF, gate_s, up_s);
        }
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    float * o[4] = { nullptr, nullptr, nullptr, nullptr };
    for (int i = 0; i < n_mats; ++i) {
        o[i] = (float *) dsts[i]->data;
    }
    const int64_t ne = m*n;
    if (plan->d_type == CUDA_R_32F) {
        fp4_split_out<float><<<(ne + threads - 1)/threads, threads, 0, stream>>>((const float *) dtmp.get(), n, m, o[0], o[1], o[2], o[3], np[0], np[1], np[2], np[3], false, ps);
    } else {
        fp4_split_out<uint16_t><<<(ne + threads - 1)/threads, threads, 0, stream>>>((const uint16_t *) dtmp.get(), n, m, o[0], o[1], o[2], o[3], np[0], np[1], np[2], np[3], plan->d_type == CUDA_R_16BF, ps);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_fp4_gemm_group(ggml_backend_cuda_context & ctx, const ggml_tensor * const * src0s, const int n_mats, const ggml_tensor * src1,
        ggml_tensor * const * dsts, const float * const * w_s) {
    return fp4_gemm_group_impl(ctx, src0s, n_mats, src1, dsts, -1, nullptr, w_s);
}

bool ggml_cuda_fp4_gemm_group_geglu(ggml_backend_cuda_context & ctx, const ggml_tensor * const * src0s, const ggml_tensor * src1,
        const int gate_part, ggml_tensor * glu, const float * const * w_s, const ggml_tensor * down_w, const float * down_ws) {
    return fp4_gemm_group_impl(ctx, src0s, 2, src1, nullptr, gate_part, glu, w_s, down_w, down_ws);
}

// ---------------------------------------------------------------------------------------------------------------------
// small weights through a TF32 GEMM over their F32 copies, see fp4gemm.cuh

struct small_gemm_plan {
    bool ok = false;
    cublasLtMatmulDesc_t   desc = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t   algo;
    std::vector<cublasLtMatmulAlgo_t> cands; // the heuristic's, for lt_tune_select
    bool tuned = false;
};

struct small_gemm_device {
    std::unordered_map<const void *, float *> weights; // the quantized data -> its F32 copy
    std::map<std::tuple<int64_t, int64_t, int64_t, int64_t>, small_gemm_plan> plans;
    void * ws = nullptr;
};

static small_gemm_device small_gemm_devices[GGML_CUDA_MAX_DEVICES];
static constexpr size_t  small_gemm_ws_size   = 4ull << 20;
// under 4 MB of F32: past that the copy costs more memory than the quantized kernels' fixed overheads are worth (an MoE
// shared expert of 2048x512 ran slower through it and held 480 MB of copies)
static constexpr int64_t small_gemm_max_elems = (1ll << 20) - 1;

bool ggml_cuda_small_gemm_applies(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    static const bool off = getenv("GGML_CUDA_SMALL_GEMM") && atoi(getenv("GGML_CUDA_SMALL_GEMM")) == 0;
    if (off || dst->op != GGML_OP_MUL_MAT) {
        return false;
    }
    return ggml_is_quantized(src0->type) && ggml_get_to_fp32_cuda(src0->type) != nullptr && ggml_is_contiguous(src0) &&
        src0->ne[2] == 1 && src0->ne[3] == 1 && src0->ne[0]*src0->ne[1] <= small_gemm_max_elems &&
        src0->buffer != nullptr && ggml_backend_buffer_get_usage(src0->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
        src1->type == GGML_TYPE_F32 && src1->ne[2] == 1 && src1->ne[3] == 1 && src1->nb[0] == sizeof(float) &&
        src1->ne[1] >= 8 && src1->nb[1] % sizeof(float) == 0 &&
        dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst) && dst->ne[2] == 1 && dst->ne[3] == 1 &&
        ggml_mul_mat_get_n_seqs(dst) > 1;
}

bool ggml_cuda_small_gemm(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    std::lock_guard<std::mutex> lock(fp4_mtx);

    fp4_device & fd = fp4_devices[ctx.device];
    if (fd.lt == nullptr) {
        CUBLAS_CHECK(cublasLtCreate(&fd.lt));
        CUDA_CHECK(cudaMalloc(&fd.beta, sizeof(float)));
        CUDA_CHECK(cudaMemset(fd.beta, 0, sizeof(float)));
    }
    small_gemm_device & sd = small_gemm_devices[ctx.device];
    cudaStream_t stream = ctx.stream();

    const int64_t k   = src0->ne[0];
    const int64_t n   = src0->ne[1];
    const int64_t m   = src1->ne[1];
    const int64_t ldb = src1->nb[1] / sizeof(float);

    auto it = sd.weights.find(src0->data);
    if (it == sd.weights.end() || sd.ws == nullptr) {
        // the copy is made outside a graph capture: the shape's first call is never captured
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
        if (capture != cudaStreamCaptureStatusNone) {
            return false;
        }
        if (sd.ws == nullptr) {
            CUDA_CHECK(cudaMalloc(&sd.ws, small_gemm_ws_size));
        }
        if (it == sd.weights.end()) {
            float * w = nullptr;
            CUDA_CHECK(cudaMalloc(&w, (size_t) k*n*sizeof(float)));
            ggml_get_to_fp32_cuda(src0->type)(src0->data, w, k*n, stream);
            CUDA_CHECK(cudaGetLastError());
            it = sd.weights.emplace(src0->data, w).first;
        }
    }

    small_gemm_plan & p = sd.plans[std::make_tuple(m, n, k, ldb)];
    if (p.desc == nullptr) {
        CUBLAS_CHECK(cublasLtMatmulDescCreate(&p.desc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
        const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_32F, k, n, k));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_32F, k, m, ldb));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.c, CUDA_R_32F, n, m, n));
        cublasLtMatmulPreference_t pref = nullptr;
        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
        const size_t ws = small_gemm_ws_size;
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
        cublasLtMatmulHeuristicResult_t res[64] = {};
        int n_res = 0;
        const cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(fd.lt, p.desc, p.a, p.b, p.c, p.c, pref, 64, res, &n_res);
        cublasLtMatmulPreferenceDestroy(pref);
        p.ok = st == CUBLAS_STATUS_SUCCESS && n_res > 0;
        if (p.ok) {
            p.algo = res[0].algo;
            for (int i = 0; i < n_res; ++i) {
                if (res[i].state == CUBLAS_STATUS_SUCCESS) {
                    p.cands.push_back(res[i].algo);
                }
            }
        }
    }
    if (!p.ok) {
        return false;
    }

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    if (!p.tuned) {
        if (!lt_tune_on() || p.cands.empty()) {
            p.tuned = true;
        } else {
            char key[128];
            snprintf(key, sizeof(key), "tf32 %lldx%lldx%lld l%lld", (long long) lt_tune_bucket(m), (long long) n, (long long) k, (long long) ldb);
            auto run = [&](const cublasLtMatmulAlgo_t & al) {
                return cublasLtMatmul(fd.lt, p.desc, &alpha, it->second, p.a, src1->data, p.b, &beta, dst->data, p.c, dst->data, p.c, &al,
                        sd.ws, small_gemm_ws_size, stream);
            };
            cublasLtMatmulAlgo_t al = p.algo;
            if (lt_tune_select(fd.lt, p.desc, p.a, p.b, p.c, p.cands, key, ctx.device, stream, run, al, small_gemm_ws_size)) {
                p.algo  = al;
                p.tuned = true;
            }
        }
    }
    CUBLAS_CHECK(cublasLtMatmul(fd.lt, p.desc, &alpha, it->second, p.a, src1->data, p.b, &beta,
            dst->data, p.c, dst->data, p.c, &p.algo, sd.ws, small_gemm_ws_size, stream));
    return true;
}

void ggml_cuda_fp4_gemm_forget(int device, const void * base, size_t size) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return;
    }
    {
        // the input scales of weights in the buffer, or held in it
        std::lock_guard<std::mutex> lock(fp4_in_s_mtx);
        auto in_range = [&](const void * p) { return (const char *) p >= (const char *) base && (const char *) p < (const char *) base + size; };
        for (auto it = fp4_in_s.begin(); it != fp4_in_s.end();) {
            it = in_range(it->first) || in_range(it->second) ? fp4_in_s.erase(it) : std::next(it);
        }
    }
    std::lock_guard<std::mutex> lock(fp4_mtx);
    {
        auto & small = small_gemm_devices[device].weights;
        for (auto it = small.begin(); it != small.end();) {
            const char * p = (const char *) it->first;
            if (p >= (const char *) base && p < (const char *) base + size) {
                CUDA_CHECK(cudaFree(it->second));
                it = small.erase(it);
            } else {
                ++it;
            }
        }
    }
    auto & weights = fp4_devices[device].weights;
    for (auto it = weights.begin(); it != weights.end();) {
        const char * p = (const char *) it->first;
        if (p >= (const char *) base && p < (const char *) base + size) {
            CUDA_CHECK(cudaFree(it->second.q));
            CUDA_CHECK(cudaFree(it->second.sf));
            it = weights.erase(it);
        } else {
            ++it;
        }
    }
    auto & fd = fp4_devices[device];
    for (auto it = fd.group_weights.begin(); it != fd.group_weights.end();) {
        bool hit = false;
        for (const void * part : it->first) {
            hit = hit || ((const char *) part >= (const char *) base && (const char *) part < (const char *) base + size);
        }
        if (!hit) {
            ++it;
            continue;
        }
        for (const void * part : it->first) {
            fd.group_of.erase(part);
        }
        CUDA_CHECK(cudaFree(it->second.q));
        CUDA_CHECK(cudaFree(it->second.sf));
        it = fd.group_weights.erase(it);
    }
}

void ggml_cuda_fp4_gemm_graph_begin(ggml_backend_cuda_context & ctx) {
    std::lock_guard<std::mutex> lock(fp4_mtx);
    auto it = fp4_ctxs.find(&ctx);
    if (it != fp4_ctxs.end()) {
        it->second.gen++;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// row |max| registry, see fp4gemm.cuh. One device array per context, a slot of max_rows floats per producing tensor.
// A producer's maxes are read by the GEMM right after it in the same graph, so the slots go round in a ring from the
// start of each graph: a graph takes the same slots every time it runs (a captured one keeps them), and graphs never
// run at once.

struct fp4_row_amax {
    static constexpr int64_t max_rows = 2048;
    static constexpr int     n_slots  = 256;
    float *  buf = nullptr;
    uint64_t gen = 0;
    int      cursor = 0;
    std::unordered_map<const ggml_tensor *, int> slot_of;
    std::vector<const ggml_tensor *> owner = std::vector<const ggml_tensor *>(n_slots, nullptr);
    struct entry { const void * data; int64_t n_rows; uint64_t gen; };
    std::unordered_map<const ggml_tensor *, entry> written;
};
static std::unordered_map<const ggml_backend_cuda_context *, fp4_row_amax> fp4_row_amaxes;
// its own lock: the GEMM asks for row maxes while it holds fp4_mtx
static std::mutex fp4_ra_mtx;

void ggml_cuda_row_amax_graph_begin(ggml_backend_cuda_context & ctx) {
    std::lock_guard<std::mutex> lock(fp4_ra_mtx);
    auto & ra = fp4_row_amaxes[&ctx];
    ra.gen++;
    ra.cursor = 0;
    ra.slot_of.clear();
    ra.written.clear();
    std::fill(ra.owner.begin(), ra.owner.end(), nullptr);
}

float * ggml_cuda_row_amax_begin(ggml_backend_cuda_context & ctx, const ggml_tensor * t, const void * data, int64_t n_rows) {
    static const bool off = getenv("GGML_CUDA_FP4_NO_ROWAMAX") != nullptr;
    if (off || n_rows <= 0 || n_rows > fp4_row_amax::max_rows) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(fp4_ra_mtx);
    auto & ra = fp4_row_amaxes[&ctx];
    if (!ra.buf) {
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaMalloc(&ra.buf, (size_t) fp4_row_amax::n_slots*fp4_row_amax::max_rows*sizeof(float)));
    }
    auto it = ra.slot_of.find(t);
    if (it == ra.slot_of.end()) {
        // the next slot of the ring; its previous producer's maxes are no longer offered
        const int slot = ra.cursor++ % fp4_row_amax::n_slots;
        if (const ggml_tensor * prev = ra.owner[slot]) {
            ra.slot_of.erase(prev);
            ra.written.erase(prev);
        }
        ra.owner[slot] = t;
        it = ra.slot_of.emplace(t, slot).first;
    }
    ra.written[t] = { data, n_rows, ra.gen };
    return ra.buf + (size_t) it->second*fp4_row_amax::max_rows;
}

const float * ggml_cuda_row_amax_find(ggml_backend_cuda_context & ctx, const ggml_tensor * t, const void * data, int64_t n_rows) {
    std::lock_guard<std::mutex> lock(fp4_ra_mtx);
    auto rit = fp4_row_amaxes.find(&ctx);
    if (rit == fp4_row_amaxes.end()) {
        return nullptr;
    }
    auto & ra = rit->second;
    auto it = ra.written.find(t);
    static const bool say = getenv("LLAMA_ROUND_TIMING") != nullptr;
    static int64_t n_hit = 0, n_miss = 0;
    const bool hit = !(it == ra.written.end() || it->second.gen != ra.gen || it->second.data != data || it->second.n_rows != n_rows);
    if (say) {
        (hit ? n_hit : n_miss)++;
        if ((n_hit + n_miss) % 20000 == 0) {
            fprintf(stderr, "[round] fp4 row amax: %lld from the producer, %lld own passes\n", (long long) n_hit, (long long) n_miss);
        }
    }
    if (!hit) {
        return nullptr;
    }
    auto st = ra.slot_of.find(t);
    return st == ra.slot_of.end() ? nullptr : ra.buf + (size_t) st->second*fp4_row_amax::max_rows;
}

// ---------------------------------------------------------------------------------------------------------------------
// FP8 E4M3 weights with a scale per row (GGML_TYPE_F8_E4M3 and its .scale tensor), as vLLM runs an FP8 checkpoint
// (W8A8: per-channel weight scales, dynamic per-token activation scales). One sequence's few rows go through a matvec
// of their F32 values; several sequences' rows are quantized per token and go through cuBLASLt's E4M3 GEMM. cuBLASLt
// has no per-row or per-token scale vectors for E4M3 on sm_120 (OUTER_VEC_32F, alpha vectors), so the product takes
// both scales in one pass after it.

bool ggml_cuda_fp8_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    return src0->type == GGML_TYPE_F8_E4M3 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(src0) && src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        src1->nb[0] == sizeof(float) && src1->nb[1] % (4*sizeof(float)) == 0 && ggml_is_contiguous(dst) &&
        src0->ne[0] % 16 == 0;
}

// 16 E4M3 values of a 128-bit load, as floats (exact: E4M3 fits FP16)
static __device__ __forceinline__ void fp8_unpack16(const uint4 q, float * f) {
    const uint32_t w[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const __half2_raw lo = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) (w[j] & 0xFFFF), __NV_E4M3);
        const __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) (w[j] >> 16),    __NV_E4M3);
        const float2 a = __half22float2(*(const half2 *) &lo);
        const float2 b = __half22float2(*(const half2 *) &hi);
        f[4*j + 0] = a.x;
        f[4*j + 1] = a.y;
        f[4*j + 2] = b.x;
        f[4*j + 3] = b.y;
    }
}

// a warp a row, 16 weights a lane at a time; ncols activation rows of F32, the row's and the tensor's scale on the sums
template <int ncols>
static __global__ void fp8_mul_mat_vec(const uint8_t * __restrict__ w, const float * __restrict__ y, float * __restrict__ dst,
        const float * __restrict__ s_row, const float * __restrict__ s_t, const int64_t k, const int64_t nrows,
        const int64_t ldy, const int64_t ldd) {
    const int64_t row = (int64_t) blockIdx.x*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    if (row >= nrows) {
        return;
    }
    const int lane = threadIdx.x % WARP_SIZE;
    const uint8_t * wr = w + row*k;

    float sum[ncols];
#pragma unroll
    for (int c = 0; c < ncols; ++c) {
        sum[c] = 0.0f;
    }
    for (int64_t i = 16*lane; i < k; i += 16*WARP_SIZE) {
        float wf[16];
        fp8_unpack16(*(const uint4 *) (wr + i), wf);
#pragma unroll
        for (int c = 0; c < ncols; ++c) {
            const float4 * yc = (const float4 *) (y + c*ldy + i);
            float acc = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float4 v = yc[j];
                acc += wf[4*j + 0]*v.x + wf[4*j + 1]*v.y + wf[4*j + 2]*v.z + wf[4*j + 3]*v.w;
            }
            sum[c] += acc;
        }
    }
    const float sc = (s_row ? s_row[row] : 1.0f) * (s_t ? s_t[0] : 1.0f);
#pragma unroll
    for (int c = 0; c < ncols; ++c) {
        const float v = warp_reduce_sum(sum[c]);
        if (lane == 0) {
            dst[c*ldd + row] = v*sc;
        }
    }
}

template <int ncols>
static void fp8_mul_mat_vec_cuda(const uint8_t * w, const float * y, float * dst, const float * s_row, const float * s_t,
        const int64_t k, const int64_t nrows, const int64_t ldy, const int64_t ldd, cudaStream_t stream) {
    constexpr int rows_per_block = 8;
    const dim3 grid((unsigned) ((nrows + rows_per_block - 1) / rows_per_block));
    fp8_mul_mat_vec<ncols><<<grid, rows_per_block*WARP_SIZE, 0, stream>>>(w, y, dst, s_row, s_t, k, nrows, ldy, ldd);
}

// up to 8 columns at a time
static void fp8_mul_mat_vec_any(const uint8_t * w, const float * y, float * dst, const float * s_row, const float * s_t,
        const int64_t k, const int64_t nrows, const int64_t ncols, const int64_t ldy, const int64_t ldd, cudaStream_t stream) {
    for (int64_t c0 = 0; c0 < ncols; c0 += 8) {
        const float * yc = y + c0*ldy;
        float * dc = dst + c0*ldd;
        switch (std::min<int64_t>(8, ncols - c0)) {
            case 1: fp8_mul_mat_vec_cuda<1>(w, yc, dc, s_row, s_t, k, nrows, ldy, ldd, stream); break;
            case 2: fp8_mul_mat_vec_cuda<2>(w, yc, dc, s_row, s_t, k, nrows, ldy, ldd, stream); break;
            case 3: fp8_mul_mat_vec_cuda<3>(w, yc, dc, s_row, s_t, k, nrows, ldy, ldd, stream); break;
            case 4: fp8_mul_mat_vec_cuda<4>(w, yc, dc, s_row, s_t, k, nrows, ldy, ldd, stream); break;
            case 5: fp8_mul_mat_vec_cuda<5>(w, yc, dc, s_row, s_t, k, nrows, ldy, ldd, stream); break;
            case 6: fp8_mul_mat_vec_cuda<6>(w, yc, dc, s_row, s_t, k, nrows, ldy, ldd, stream); break;
            case 7: fp8_mul_mat_vec_cuda<7>(w, yc, dc, s_row, s_t, k, nrows, ldy, ldd, stream); break;
            default: fp8_mul_mat_vec_cuda<8>(w, yc, dc, s_row, s_t, k, nrows, ldy, ldd, stream); break;
        }
    }
}

// vLLM's dynamic per-token FP8 quantization: one block a token, its |max| over the row, the scale |max|/448 (at least
// 1/(448*512)), the row over the scale as E4M3
static __global__ void fp8_quantize_act(const float * __restrict__ x, uint8_t * __restrict__ q, float * __restrict__ s,
        const int64_t k, const int64_t ldx) {
    const int64_t row = blockIdx.x;
    const float * xr = x + row*ldx;
    float amax = 0.0f;
    for (int64_t i = 4*threadIdx.x; i < k; i += 4*blockDim.x) {
        const float4 v = *(const float4 *) (xr + i);
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
    __shared__ float sm[32];
    amax = warp_reduce_max(amax);
    if (threadIdx.x % WARP_SIZE == 0) {
        sm[threadIdx.x / WARP_SIZE] = amax;
    }
    __syncthreads();
    amax = threadIdx.x % WARP_SIZE < blockDim.x / WARP_SIZE ? sm[threadIdx.x % WARP_SIZE] : 0.0f;
    amax = warp_reduce_max(amax);
    const float sc  = ggml_cuda_fp8_act_scale(amax);
    const float inv = 1.0f / sc;
    if (threadIdx.x == 0) {
        s[row] = sc;
    }
    uint8_t * qr = q + row*k;
    for (int64_t i = 4*threadIdx.x; i < k; i += 4*blockDim.x) {
        const float4 v = *(const float4 *) (xr + i);
        const __nv_fp8x4_e4m3 p(make_float4(v.x*inv, v.y*inv, v.z*inv, v.w*inv));
        *(uint32_t *) (qr + i) = p.__x;
    }
}

// out = d * s_tok[token] * s_t * s_row[row]: d and out [n_tok][n_rows] (out may be d), a token a block row, 4 rows a
// thread (n_rows % 4 == 0)
static __global__ void fp8_scale_out(const float * __restrict__ d, float * __restrict__ out, const float * __restrict__ s_tok,
        const float * __restrict__ s_row, const float * __restrict__ s_t, const int64_t n_rows) {
    const int64_t tok = blockIdx.y;
    const int64_t r   = 4*((int64_t) blockIdx.x*blockDim.x + threadIdx.x);
    if (r >= n_rows) {
        return;
    }
    const float st = s_tok[tok] * (s_t ? s_t[0] : 1.0f);
    float4 v = *(const float4 *) (d + tok*n_rows + r);
    if (s_row) {
        const float4 s = *(const float4 *) (s_row + r);
        v.x *= s.x*st; v.y *= s.y*st; v.z *= s.z*st; v.w *= s.w*st;
    } else {
        v.x *= st; v.y *= st; v.z *= st; v.w *= st;
    }
    *(float4 *) (out + tok*n_rows + r) = v;
}

// gelu(gate)*up from the raw products of a gate and an up projection over the same quantized activations, their rows'
// and the token's scales applied first (fp4_geglu_out's arithmetic): vLLM's merged gate_up and gelu_and_mul
static __global__ void fp8_geglu_out(const float * __restrict__ dg, const float * __restrict__ du, float * __restrict__ out,
        const float * __restrict__ s_tok, const float * __restrict__ sg_row, const float * __restrict__ su_row,
        const float * __restrict__ sg_t, const float * __restrict__ su_t, const int64_t n_rows) {
    const int64_t tok = blockIdx.y;
    const int64_t r   = 4*((int64_t) blockIdx.x*blockDim.x + threadIdx.x);
    if (r >= n_rows) {
        return;
    }
    const float st = s_tok[tok];
    const float gt = st * (sg_t ? sg_t[0] : 1.0f);
    const float ut = st * (su_t ? su_t[0] : 1.0f);
    const float4 g = *(const float4 *) (dg + tok*n_rows + r);
    const float4 u = *(const float4 *) (du + tok*n_rows + r);
    const float4 sg = sg_row ? *(const float4 *) (sg_row + r) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    const float4 su = su_row ? *(const float4 *) (su_row + r) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    float4 o;
    o.x = ggml_cuda_op_gelu_single(g.x*sg.x*gt) * (u.x*su.x*ut);
    o.y = ggml_cuda_op_gelu_single(g.y*sg.y*gt) * (u.y*su.y*ut);
    o.z = ggml_cuda_op_gelu_single(g.z*sg.z*gt) * (u.z*su.z*ut);
    o.w = ggml_cuda_op_gelu_single(g.w*sg.w*gt) * (u.w*su.w*ut);
    *(float4 *) (out + tok*n_rows + r) = o;
}

// fp8_geglu_out's arithmetic from F32 or BF16 products, a block a token, and for an FP8 down projection the row's per-token
// E4M3 quantization (fp8_quantize_act's) in the same launch: the row's |max| is the block's, so the down GEMM skips its
// quantize pass (ggml_cuda_fp8_act_claim). out is written too, for any other reader of the GEGLU
template <typename T>
static __global__ void fp8_geglu_quant_out(const T * __restrict__ dg, const T * __restrict__ du, float * __restrict__ out,
        const float * __restrict__ s_tok, const float * __restrict__ sg_row, const float * __restrict__ su_row,
        const float * __restrict__ sg_t, const float * __restrict__ su_t, const int64_t n_rows, uint8_t * __restrict__ q8,
        float * __restrict__ q8_s) {
    const int64_t tok = blockIdx.x;
    const float st = s_tok[tok];
    const float gt = st * (sg_t ? sg_t[0] : 1.0f);
    const float ut = st * (su_t ? su_t[0] : 1.0f);
    auto ld = [](const T * p) -> float4 {
        if constexpr (std::is_same_v<T, float>) {
            return *(const float4 *) p;
        } else {
            const nv_bfloat162 a = ((const nv_bfloat162 *) p)[0];
            const nv_bfloat162 b = ((const nv_bfloat162 *) p)[1];
            return make_float4(__low2float(a), __high2float(a), __low2float(b), __high2float(b));
        }
    };
    float amax = 0.0f;
    for (int64_t r = 4*threadIdx.x; r < n_rows; r += 4*blockDim.x) {
        const float4 g = ld(dg + tok*n_rows + r);
        const float4 u = ld(du + tok*n_rows + r);
        const float4 sg = sg_row ? *(const float4 *) (sg_row + r) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        const float4 su = su_row ? *(const float4 *) (su_row + r) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        float4 o;
        o.x = ggml_cuda_op_gelu_single(g.x*sg.x*gt) * (u.x*su.x*ut);
        o.y = ggml_cuda_op_gelu_single(g.y*sg.y*gt) * (u.y*su.y*ut);
        o.z = ggml_cuda_op_gelu_single(g.z*sg.z*gt) * (u.z*su.z*ut);
        o.w = ggml_cuda_op_gelu_single(g.w*sg.w*gt) * (u.w*su.w*ut);
        *(float4 *) (out + tok*n_rows + r) = o;
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(o.x), fabsf(o.y)), fmaxf(fabsf(o.z), fabsf(o.w))));
    }
    if (q8 == nullptr) {
        return;
    }
    __shared__ float sm[32];
    amax = warp_reduce_max(amax);
    if (threadIdx.x % WARP_SIZE == 0) {
        sm[threadIdx.x / WARP_SIZE] = amax;
    }
    __syncthreads();
    amax = threadIdx.x % WARP_SIZE < blockDim.x / WARP_SIZE ? sm[threadIdx.x % WARP_SIZE] : 0.0f;
    amax = warp_reduce_max(amax);
    const float sc  = ggml_cuda_fp8_act_scale(amax);
    const float inv = 1.0f / sc;
    if (threadIdx.x == 0) {
        q8_s[tok] = sc;
    }
    uint8_t * qr = q8 + tok*n_rows;
    for (int64_t r = 4*threadIdx.x; r < n_rows; r += 4*blockDim.x) {
        const float4 v = *(const float4 *) (out + tok*n_rows + r);
        const __nv_fp8x4_e4m3 p(make_float4(v.x*inv, v.y*inv, v.z*inv, v.w*inv));
        *(uint32_t *) (qr + r) = p.__x;
    }
}

struct fp8_plan {
    bool ok = false;
    cublasLtMatmulDesc_t   desc = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t   algo;
    std::vector<cublasLtMatmulAlgo_t> cands; // the heuristic's, for lt_tune_select
    bool tuned = false;
};

// (device, tokens, rows, k, BF16 D)
static std::map<std::tuple<int, int64_t, int64_t, int64_t, int>, fp8_plan> fp8_plans;

// D [n x m] (column-major, ld n) = A^T B: A the E4M3 weights, K x n column-major; B the E4M3 activations, K x m.
// d16: D in BF16 (a product the next kernel reads once)
static fp8_plan * fp8_get_plan(fp4_device & fd, const int device, const int64_t m, const int64_t n, const int64_t k, const bool d16 = false) {
    auto key = std::make_tuple(device, m, n, k, (int) d16);
    auto it = fp8_plans.find(key);
    if (it != fp8_plans.end()) {
        return &it->second;
    }
    fp8_plan & p = fp8_plans[key];
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&p.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_8F_E4M3, k, n, k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_8F_E4M3, k, m, k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.c, d16 ? CUDA_R_16BF : CUDA_R_32F, n, m, n));
    cublasLtMatmulPreference_t pref;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    size_t ws = fp4_workspace_size;
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
    cublasLtMatmulHeuristicResult_t res[64];
    int n_res = 0;
    if (cublasLtMatmulAlgoGetHeuristic(fd.lt, p.desc, p.a, p.b, p.c, p.c, pref, 64, res, &n_res) == CUBLAS_STATUS_SUCCESS && n_res > 0) {
        p.algo = res[0].algo;
        p.ok   = true;
        // the heuristic's first algorithm without split-K, as vLLM's CUTLASS kernel has none: the reduce kernels cost
        // more than the split wins (Gemma 4 E4B at 128 rows: FP8 GEMMs and reduces 1432 -> 1364 us a forward, 70 kernels
        // fewer). The block-scaled FP4 GEMMs gain nothing from it. GGML_CUDA_FP8_SPLIT=1 keeps the heuristic's first
        static const bool split_ok = getenv("GGML_CUDA_FP8_SPLIT") != nullptr;
        for (int i = 0; !split_ok && i < n_res; ++i) {
            if (lt_algo_split_k(res[i].algo) <= 1) {
                p.algo = res[i].algo;
                break;
            }
        }
        for (int i = 0; i < n_res; ++i) {
            if (res[i].state == CUBLAS_STATUS_SUCCESS) {
                p.cands.push_back(res[i].algo);
            }
        }
    }
    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));
    return &p;
}

// the quantized activations (E4M3 values, per-token scales) and the GEMM's workspace, per context. Two slots: slot 0
// for the activations a GEMM quantizes itself, slot 1 for a producer's (ggml_cuda_fp8_act_claim), which may still read
// slot 0's scales as it writes (the norm pair after the o projection applies that product's per-token scales)
// the fewest activation rows an FP8 product takes the tensor-core GEMM for (activations quantized to E4M3 by token):
// from a verify's rows on, where the matvec re-reads every activation row for every weight row and reaches a third of
// the bandwidth; one row keeps the matvec. GGML_CUDA_FP8_GEMM_MIN overrides
static int fp8_gemm_min_rows() {
    static const int n = getenv("GGML_CUDA_FP8_GEMM_MIN") ? std::max(1, atoi(getenv("GGML_CUDA_FP8_GEMM_MIN"))) : 2;
    return n;
}

struct fp8_ctx {
    uint8_t * buf      = nullptr;
    size_t    buf_size = 0;
    size_t    slot     = 0; // each slot's bytes; the workspace follows the two
    uint64_t            act_gen  = ~(uint64_t) 0;
    const ggml_tensor * act_t    = nullptr;
    const void        * act_data = nullptr;
    uint8_t           * act_q    = nullptr;
    float             * act_s    = nullptr;
};
static std::unordered_map<const void *, fp8_ctx> fp8_ctxs;

// room in fc for m activation rows of k in either slot and the GEMM's workspace; false when it would have to grow while a
// graph is captured. Under fp4_mtx
static bool fp8_ctx_reserve(ggml_backend_cuda_context & ctx, fp8_ctx & fc, const int64_t m, const int64_t k) {
    auto align = [](const size_t x) { return (x + 255) & ~(size_t) 255; };
    const size_t slot = align((size_t) m*k) + align((size_t) m*sizeof(float));
    if (fc.slot < slot || fc.buf_size < 2*fc.slot + fp4_workspace_size) {
        cudaStream_t stream = ctx.stream();
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
        if (capture != cudaStreamCaptureStatusNone) {
            return false;
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (fc.buf) {
            CUDA_CHECK(cudaFree(fc.buf));
        }
        // each slot starts 256-byte aligned, as cuBLASLt assumes of its operands
        fc.slot     = align(std::max(fc.slot, slot + slot/4));
        fc.buf_size = std::max(2*fc.slot + fp4_workspace_size, gemm_buf_floor);
        CUDA_CHECK(cudaMalloc(&fc.buf, fc.buf_size));
        fc.act_gen = ~(uint64_t) 0;
        // graphs captured before read the freed buffer
        ggml_cuda_graphs_invalidate(ctx);
    }
    return true;
}

// slot i's E4M3 rows (m of k) and per-token scales
static void fp8_ctx_slot(const fp8_ctx & fc, const int i, const int64_t m, const int64_t k, uint8_t ** q, float ** s) {
    *q = fc.buf + i*fc.slot;
    *s = (float *) (*q + (((size_t) m*k + 255) & ~(size_t) 255));
}

// under fp4_mtx. The claim takes the slot the live activations are not in: its producer may read their per-token scales
// as it writes (the norm pair after the o projection, the GEGLU after gate and up). may_grow false: a caller holding
// pointers into the buffer (those scales) gets no claim rather than a buffer that moved
static bool fp8_act_claim_locked(ggml_backend_cuda_context & ctx, const ggml_tensor * act, uint8_t ** q, float ** s, const bool may_grow) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    static const bool gemm_off = getenv("GGML_CUDA_FP8_GEMM") && atoi(getenv("GGML_CUDA_FP8_GEMM")) == 0;
    if (gemm_off || act->ne[1] < fp8_gemm_min_rows() || !GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_ADA_LOVELACE ||
            act->type != GGML_TYPE_F32 || !ggml_is_contiguous(act) || act->ne[0] % 16 != 0 || act->ne[2] != 1 || act->ne[3] != 1) {
        return false;
    }
    fp8_ctx & fc = fp8_ctxs[&ctx];
    const size_t need = (((size_t) act->ne[1]*act->ne[0] + 255) & ~(size_t) 255) + (((size_t) act->ne[1]*sizeof(float) + 255) & ~(size_t) 255);
    if (!may_grow && (fc.buf == nullptr || fc.slot < need)) {
        return false;
    }
    if (!fp8_ctx_reserve(ctx, fc, act->ne[1], act->ne[0])) {
        return false;
    }
    const bool live_in_1 = fc.act_gen == fp4_ctxs[&ctx].gen && fc.act_q == fc.buf + fc.slot;
    fp8_ctx_slot(fc, live_in_1 ? 0 : 1, act->ne[1], act->ne[0], q, s);
    fc.act_gen  = fp4_ctxs[&ctx].gen;
    fc.act_t    = act;
    fc.act_data = act->data;
    fc.act_q    = *q;
    fc.act_s    = *s;
    return true;
}

bool ggml_cuda_fp8_act_claim(ggml_backend_cuda_context & ctx, const ggml_tensor * act, uint8_t ** q, float ** s, const bool may_grow) {
    std::lock_guard<std::mutex> lock(fp4_mtx);
    return fp8_act_claim_locked(ctx, act, q, s, may_grow);
}

// d = the raw product (no scale applied; BF16 when d16), s_tok_out = the activations' per-token scales; under fp4_mtx
static bool fp8_gemm_raw(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, void * d,
        const float ** s_tok_out, const bool d16 = false) {
    fp4_device & fd = fp4_devices[ctx.device];
    if (fd.lt == nullptr) {
        CUBLAS_CHECK(cublasLtCreate(&fd.lt));
        CUDA_CHECK(cudaMalloc(&fd.beta, sizeof(float)));
        CUDA_CHECK(cudaMemset(fd.beta, 0, sizeof(float)));
    }

    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    const int64_t m = src1->ne[1];
    if (n % 4 != 0) {
        return false;
    }
    fp8_plan * plan = fp8_get_plan(fd, ctx.device, m, n, k, d16);
    if (!plan->ok) {
        return false;
    }

    cudaStream_t stream = ctx.stream();
    fp8_ctx & fc = fp8_ctxs[&ctx];
    if (!fp8_ctx_reserve(ctx, fc, m, k)) {
        return false;
    }
    uint8_t * ws = fc.buf + 2*fc.slot;

    // products of the same activations in a row (q, k and v) quantize them once; a producer may have quantized them
    const uint64_t gen = fp4_ctxs[&ctx].gen;
    if (!(fc.act_gen == gen && fc.act_t == src1 && fc.act_data == src1->data)) {
        fp8_ctx_slot(fc, 0, m, k, &fc.act_q, &fc.act_s);
        fp8_quantize_act<<<(unsigned) m, 256, 0, stream>>>((const float *) src1->data, fc.act_q, fc.act_s, k, src1->nb[1]/sizeof(float));
        CUDA_CHECK(cudaGetLastError());
        fc.act_gen  = gen;
        fc.act_t    = src1;
        fc.act_data = src1->data;
    }
    uint8_t * q_act = fc.act_q;
    float   * s_tok = fc.act_s;

    static const float one = 1.0f, zero = 0.0f;
    if (!plan->tuned) {
        if (!lt_tune_on() || plan->cands.empty()) {
            plan->tuned = true;
        } else {
            char key[128];
            snprintf(key, sizeof(key), "fp8 %lldx%lldx%lld%s", (long long) lt_tune_bucket(m), (long long) n, (long long) k, d16 ? " bf16" : "");
            auto run = [&](const cublasLtMatmulAlgo_t & al) {
                return cublasLtMatmul(fd.lt, plan->desc, &one, src0->data, plan->a, q_act, plan->b, &zero, d, plan->c, d, plan->c, &al, ws,
                        fp4_workspace_size, stream);
            };
            cublasLtMatmulAlgo_t al = plan->algo;
            if (lt_tune_select(fd.lt, plan->desc, plan->a, plan->b, plan->c, plan->cands, key, ctx.device, stream, run, al)) {
                plan->algo  = al;
                plan->tuned = true;
            }
        }
    }
    CUBLAS_CHECK(cublasLtMatmul(fd.lt, plan->desc, &one, src0->data, plan->a, q_act, plan->b, &zero,
            d, plan->c, d, plan->c, &plan->algo, ws, fp4_workspace_size, stream));
    *s_tok_out = s_tok;
    return true;
}

// two weights of one shape over the same activations as one strided-batched GEMM, as vLLM runs its merged gate_up_proj:
// the weights stay where they are, the batch stride is the distance between them. d: both products, the lower weight's
// first, n*m apart (d_lo_first tells which). (device, tokens, rows, k, BF16 D, stride) -> plan
static std::map<std::tuple<int, int64_t, int64_t, int64_t, int, int64_t>, fp8_plan> fp8_pair_plans;

static bool fp8_gemm_raw2(ggml_backend_cuda_context & ctx, const ggml_tensor * w0, const ggml_tensor * w1, const ggml_tensor * src1,
        void * d, bool * w0_first, const float ** s_tok_out, const bool d16) {
    fp4_device & fd = fp4_devices[ctx.device];
    if (fd.lt == nullptr) {
        CUBLAS_CHECK(cublasLtCreate(&fd.lt));
        CUDA_CHECK(cudaMalloc(&fd.beta, sizeof(float)));
        CUDA_CHECK(cudaMemset(fd.beta, 0, sizeof(float)));
    }
    const int64_t k = w0->ne[0];
    const int64_t n = w0->ne[1];
    const int64_t m = src1->ne[1];
    const char * p0 = (const char *) w0->data;
    const char * p1 = (const char *) w1->data;
    *w0_first = p0 < p1;
    const char *  lo     = *w0_first ? p0 : p1;
    const int64_t stride = (int64_t) (*w0_first ? p1 - p0 : p0 - p1);
    if (n % 4 != 0 || stride < (int64_t) (n*k) || stride % 16 != 0) {
        return false;
    }

    auto key = std::make_tuple(ctx.device, m, n, k, (int) d16, stride);
    auto it = fp8_pair_plans.find(key);
    fp8_plan * plan = nullptr;
    if (it != fp8_pair_plans.end()) {
        plan = &it->second;
    } else {
        fp8_plan & p = fp8_pair_plans[key];
        plan = &p;
        CUBLAS_CHECK(cublasLtMatmulDescCreate(&p.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_8F_E4M3, k, n, k));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_8F_E4M3, k, m, k));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.c, d16 ? CUDA_R_16BF : CUDA_R_32F, n, m, n));
        const int32_t batch = 2;
        const int64_t sa = stride, sb = 0, sc = n*m;
        for (cublasLtMatrixLayout_t l : { p.a, p.b, p.c }) {
            CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        }
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(p.a, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sa, sizeof(sa)));
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(p.b, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sb, sizeof(sb)));
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(p.c, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sc, sizeof(sc)));
        cublasLtMatmulPreference_t pref;
        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
        size_t ws = fp4_workspace_size;
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
        cublasLtMatmulHeuristicResult_t res[64];
        int n_res = 0;
        if (cublasLtMatmulAlgoGetHeuristic(fd.lt, p.desc, p.a, p.b, p.c, p.c, pref, 64, res, &n_res) == CUBLAS_STATUS_SUCCESS && n_res > 0) {
            p.algo = res[0].algo;
            p.ok   = true;
            for (int i = 0; i < n_res; ++i) {
                if (lt_algo_split_k(res[i].algo) <= 1) {
                    p.algo = res[i].algo;
                    break;
                }
            }
            for (int i = 0; i < n_res; ++i) {
                if (res[i].state == CUBLAS_STATUS_SUCCESS) {
                    p.cands.push_back(res[i].algo);
                }
            }
        }
        CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));
    }
    if (!plan->ok) {
        return false;
    }

    cudaStream_t stream = ctx.stream();
    fp8_ctx & fc = fp8_ctxs[&ctx];
    if (!fp8_ctx_reserve(ctx, fc, m, k)) {
        return false;
    }
    uint8_t * ws = fc.buf + 2*fc.slot;
    const uint64_t gen = fp4_ctxs[&ctx].gen;
    if (!(fc.act_gen == gen && fc.act_t == src1 && fc.act_data == src1->data)) {
        fp8_ctx_slot(fc, 0, m, k, &fc.act_q, &fc.act_s);
        fp8_quantize_act<<<(unsigned) m, 256, 0, stream>>>((const float *) src1->data, fc.act_q, fc.act_s, k, src1->nb[1]/sizeof(float));
        CUDA_CHECK(cudaGetLastError());
        fc.act_gen  = gen;
        fc.act_t    = src1;
        fc.act_data = src1->data;
    }
    uint8_t * q_act = fc.act_q;

    static const float one = 1.0f, zero = 0.0f;
    if (!plan->tuned) {
        if (!lt_tune_on() || plan->cands.empty()) {
            plan->tuned = true;
        } else {
            char tkey[128];
            snprintf(tkey, sizeof(tkey), "fp8 %lldx%lldx%lld b2%s", (long long) lt_tune_bucket(m), (long long) n, (long long) k, d16 ? " bf16" : "");
            auto run = [&](const cublasLtMatmulAlgo_t & al) {
                return cublasLtMatmul(fd.lt, plan->desc, &one, lo, plan->a, q_act, plan->b, &zero, d, plan->c, d, plan->c, &al, ws,
                        fp4_workspace_size, stream);
            };
            cublasLtMatmulAlgo_t al = plan->algo;
            if (lt_tune_select(fd.lt, plan->desc, plan->a, plan->b, plan->c, plan->cands, tkey, ctx.device, stream, run, al)) {
                plan->algo  = al;
                plan->tuned = true;
            }
        }
    }
    if (cublasLtMatmul(fd.lt, plan->desc, &one, lo, plan->a, q_act, plan->b, &zero, d, plan->c, d, plan->c, &plan->algo, ws,
            fp4_workspace_size, stream) != CUBLAS_STATUS_SUCCESS) {
        plan->ok = false;
        return false;
    }
    *s_tok_out = fc.act_s;
    return true;
}

static bool fp8_gemm(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const float * s_row, const float * s_t) {
    std::lock_guard<std::mutex> lock(fp4_mtx);
    float * d = (float *) dst->data;
    const float * s_tok = nullptr;
    if (!fp8_gemm_raw(ctx, src0, src1, d, &s_tok)) {
        return false;
    }
    const int64_t n = src0->ne[1];
    const int64_t m = src1->ne[1];
    const dim3 grid((unsigned) ((n/4 + 255)/256), (unsigned) m);
    fp8_scale_out<<<grid, 256, 0, ctx.stream()>>>(d, d, s_tok, s_row, s_t, n);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_fp8_gemm_geglu(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * src1,
        ggml_tensor * glu, const float * sg_row, const float * su_row, const float * sg_t, const float * su_t, const bool down_fp8) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_ADA_LOVELACE || src1->ne[1] < fp8_gemm_min_rows() ||
            !ggml_cuda_fp8_supported(gate, src1, glu) || !ggml_cuda_fp8_supported(up, src1, glu) ||
            gate->ne[1] != up->ne[1] || gate->ne[0] != up->ne[0] || glu->ne[0] != gate->ne[1] || glu->ne[1] != src1->ne[1]) {
        return false;
    }
    const int64_t n = gate->ne[1];
    const int64_t m = src1->ne[1];
    // the products in BF16 (vLLM's activations; GGML_CUDA_FP8_GEGLU_F32=1 keeps F32), and an FP8 down projection's
    // quantization made with the GEGLU (GGML_CUDA_FUSE_OFF bit 59 leaves it to the down GEMM)
    static const bool f32 = getenv("GGML_CUDA_FP8_GEGLU_BF16") == nullptr; // BF16 products measured no faster
    uint8_t * q8 = nullptr;
    float * q8_s = nullptr;
    const bool quant_down = down_fp8 && !(ggml_cuda_fuse_off_mask() & (1ull << 59)) && n % 4 == 0;
    std::lock_guard<std::mutex> lock(fp4_mtx);
    // the claim comes after both products are queued: they read the input's rows from the slot it takes
    auto claim = [&]() {
        if (quant_down && !fp8_act_claim_locked(ctx, glu, &q8, &q8_s, false)) {
            q8   = nullptr;
            q8_s = nullptr;
        }
    };
    const float * s_tok = nullptr;
    // both products as one batched GEMM (GGML_CUDA_FP8_PAIR=0: two)
    static const bool pair_on = getenv("GGML_CUDA_FP8_PAIR") == nullptr || atoi(getenv("GGML_CUDA_FP8_PAIR")) != 0;
    if (pair_on && f32) {
        ggml_cuda_pool_alloc<float> dgu(ctx.pool(), (size_t) 2*n*m);
        bool gate_first = true;
        if (fp8_gemm_raw2(ctx, gate, up, src1, dgu.get(), &gate_first, &s_tok, false)) {
            const float * dg = dgu.get() + (gate_first ? 0 : (size_t) n*m);
            const float * du = dgu.get() + (gate_first ? (size_t) n*m : 0);
            claim();
            fp8_geglu_quant_out<float><<<(unsigned) m, 256, 0, ctx.stream()>>>(dg, du, (float *) glu->data, s_tok, sg_row, su_row, sg_t, su_t, n, q8, q8_s);
            CUDA_CHECK(cudaGetLastError());
            return true;
        }
    }
    if (f32) {
        // the gate's product goes to glu's buffer, the up's to scratch; the GELU pass reads both and writes glu
        ggml_cuda_pool_alloc<float> du(ctx.pool(), (size_t) n*m);
        float * dg = (float *) glu->data;
        if (!fp8_gemm_raw(ctx, gate, src1, dg, &s_tok) || !fp8_gemm_raw(ctx, up, src1, du.get(), &s_tok)) {
            return false;
        }
        claim();
        fp8_geglu_quant_out<float><<<(unsigned) m, 256, 0, ctx.stream()>>>(dg, du.get(), (float *) glu->data, s_tok, sg_row, su_row, sg_t, su_t, n, q8, q8_s);
    } else {
        ggml_cuda_pool_alloc<nv_bfloat16> dgu(ctx.pool(), (size_t) 2*n*m);
        nv_bfloat16 * dg = dgu.get();
        nv_bfloat16 * du = dgu.get() + (size_t) n*m;
        if (!fp8_gemm_raw(ctx, gate, src1, dg, &s_tok, true) || !fp8_gemm_raw(ctx, up, src1, du, &s_tok, true)) {
            return false;
        }
        claim();
        fp8_geglu_quant_out<nv_bfloat16><<<(unsigned) m, 256, 0, ctx.stream()>>>(dg, du, (float *) glu->data, s_tok, sg_row, su_row, sg_t, su_t, n, q8, q8_s);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// an activation of several dimensions laid out row after row (a recurrent layer's tokens by sequence), with a product
// laid out the same way: one matrix of all its rows
static bool fp8_rows_flat(const ggml_tensor * src1, const ggml_tensor * dst) {
    return src1->ne[2]*src1->ne[3] > 1 && src1->nb[2] == src1->nb[1]*src1->ne[1] && src1->nb[3] == src1->nb[2]*src1->ne[2] &&
        ggml_is_contiguous(dst) && ggml_nrows(dst) == ggml_nrows(src1);
}

static ggml_tensor fp8_as_2d(const ggml_tensor * t) {
    ggml_tensor v = *t;
    v.ne[1] = ggml_nrows(t);
    v.ne[2] = v.ne[3] = 1;
    v.nb[2] = v.nb[3] = v.nb[1]*v.ne[1];
    return v;
}

bool ggml_cuda_fp8_mul_mat_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (ggml_cuda_fp8_supported(src0, src1, dst)) {
        return true;
    }
    if (!fp8_rows_flat(src1, dst)) {
        return false;
    }
    const ggml_tensor s2 = fp8_as_2d(src1);
    const ggml_tensor d2 = fp8_as_2d(dst);
    return ggml_cuda_fp8_supported(src0, &s2, &d2);
}

bool ggml_cuda_fp8_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const float * s_row, const float * s_t) {
    if (!ggml_cuda_fp8_supported(src0, src1, dst) && fp8_rows_flat(src1, dst)) {
        ggml_tensor s2 = fp8_as_2d(src1);
        ggml_tensor d2 = fp8_as_2d(dst);
        const bool ok = ggml_cuda_fp8_mul_mat(ctx, src0, &s2, &d2, s_row, s_t);
        // the GEMM keys its quantized activations by tensor: these views live on the stack, so no later call may match
        std::lock_guard<std::mutex> lock(fp4_mtx);
        fp8_ctxs[&ctx].act_t = nullptr;
        return ok;
    }
    if (!ggml_cuda_fp8_supported(src0, src1, dst)) {
        return false;
    }
    const int64_t ncols = src1->ne[1];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    // FP8 tensor cores from Ada; GGML_CUDA_FP8_GEMM=0 keeps every product on the matvec
    static const bool gemm_off = getenv("GGML_CUDA_FP8_GEMM") && atoi(getenv("GGML_CUDA_FP8_GEMM")) == 0;
    if (ncols >= fp8_gemm_min_rows() && !gemm_off && GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_ADA_LOVELACE &&
            fp8_gemm(ctx, src0, src1, dst, s_row, s_t)) {
        return true;
    }
    fp8_mul_mat_vec_any((const uint8_t *) src0->data, (const float *) src1->data, (float *) dst->data, s_row, s_t,
            src0->ne[0], src0->ne[1], ncols, src1->nb[1]/sizeof(float), dst->nb[1]/sizeof(float), ctx.stream());
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_fp8_mul_mat_raw(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, float * d,
        const float ** s_tok) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    static const bool gemm_off = getenv("GGML_CUDA_FP8_GEMM") && atoi(getenv("GGML_CUDA_FP8_GEMM")) == 0;
    if (gemm_off || src1->ne[1] < fp8_gemm_min_rows() || !GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_ADA_LOVELACE ||
            src0->type != GGML_TYPE_F8_E4M3 || src1->type != GGML_TYPE_F32 || !ggml_is_contiguous(src0) ||
            src1->nb[0] != sizeof(float) || src1->nb[1] % (4*sizeof(float)) != 0 || src0->ne[0] % 16 != 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(fp4_mtx);
    return fp8_gemm_raw(ctx, src0, src1, d, s_tok);
}
