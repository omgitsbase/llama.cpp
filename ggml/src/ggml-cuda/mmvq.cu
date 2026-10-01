#include "mmvq.cuh"
#include "mmvq-mma.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>
#include <type_traits>

// only enabled on DGX Spark, where it is a gain on every type below. On the higher-bandwidth parts the kernel
// has little exposed latency left to hide and the extra requests cost more than they save.
// For perf data, see https://github.com/ggml-org/llama.cpp/pull/26705#issuecomment-5569335031
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
// returns true only for those quants that benefit from prefetch and false otherwise
static constexpr __host__ __device__ bool mmvq_should_prefetch(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

static __device__ __forceinline__ void mmvq_prefetch_l2(const void * p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}
#endif

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return vec_dot_q1_0_q8_1;
        case GGML_TYPE_Q2_0:    return vec_dot_q2_0_q8_1;
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int mmvq_type_size(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return sizeof(block_q1_0);
        case GGML_TYPE_Q2_0:    return sizeof(block_q2_0);
        case GGML_TYPE_Q4_0:    return sizeof(block_q4_0);
        case GGML_TYPE_Q4_1:    return sizeof(block_q4_1);
        case GGML_TYPE_Q5_0:    return sizeof(block_q5_0);
        case GGML_TYPE_Q5_1:    return sizeof(block_q5_1);
        case GGML_TYPE_Q8_0:    return sizeof(block_q8_0);
        case GGML_TYPE_MXFP4:   return sizeof(block_mxfp4);
        case GGML_TYPE_NVFP4:   return sizeof(block_nvfp4);
        case GGML_TYPE_Q2_K:    return sizeof(block_q2_K);
        case GGML_TYPE_Q3_K:    return sizeof(block_q3_K);
        case GGML_TYPE_Q4_K:    return sizeof(block_q4_K);
        case GGML_TYPE_Q5_K:    return sizeof(block_q5_K);
        case GGML_TYPE_Q6_K:    return sizeof(block_q6_K);
        case GGML_TYPE_IQ2_XXS: return sizeof(block_iq2_xxs);
        case GGML_TYPE_IQ2_XS:  return sizeof(block_iq2_xs);
        case GGML_TYPE_IQ2_S:   return sizeof(block_iq2_s);
        case GGML_TYPE_IQ3_XXS: return sizeof(block_iq3_xxs);
        case GGML_TYPE_IQ1_S:   return sizeof(block_iq1_s);
        case GGML_TYPE_IQ1_M:   return sizeof(block_iq1_m);
        case GGML_TYPE_IQ4_NL:  return sizeof(block_iq4_nl);
        case GGML_TYPE_IQ4_XS:  return sizeof(block_iq4_xs);
        case GGML_TYPE_IQ3_S:   return sizeof(block_iq3_s);
        default:                return 0;
    }
}

// an id written by a kernel that finished before an earlier kernel's PDL wait: read from L2, past any stale L1 line
static __device__ __forceinline__ uint32_t mmvq_ld_ids_early(const int32_t * p) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    return (uint32_t) __ldcg(p);
#else
    return (uint32_t) *p;
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

static constexpr __device__ bool has_vec_dot_multi(ggml_type type) {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S;
}

template <ggml_type type, int ncols>
static __device__ __forceinline__ void vec_dot_multi(
        const void * __restrict__ vx, const block_q8_1 * __restrict__ y, const int stride_col_y,
        const int kbx, const int kqs, float * __restrict__ out) {
    if constexpr (type == GGML_TYPE_Q4_K) {
        vec_dot_q4_K_q8_1_multi<ncols>(vx, y, stride_col_y, kbx, kqs, out);
    } else if constexpr (type == GGML_TYPE_IQ2_S) {
        vec_dot_iq2_s_q8_1_multi<ncols>(vx, y, stride_col_y, kbx, kqs, out);
    } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
        vec_dot_iq3_xxs_q8_1_multi<ncols>(vx, y, stride_col_y, kbx, kqs, out);
    } else if constexpr (type == GGML_TYPE_IQ3_S) {
        vec_dot_iq3_s_q8_1_multi<ncols>(vx, y, stride_col_y, kbx, kqs, out);
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return VDR_Q1_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_0:    return VDR_Q2_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4,
    MMVQ_PARAMETERS_GB10
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA2) || defined(RDNA3_5)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
    return MMVQ_PARAMETERS_TURING;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    return MMVQ_PARAMETERS_GB10;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_TURING && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return MMVQ_PARAMETERS_TURING;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_DGX_SPARK) {
        return MMVQ_PARAMETERS_GB10;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.
// Check https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627 for details

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_pascal_older(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 4;
        case GGML_TYPE_NVFP4:   return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 6;
        case GGML_TYPE_Q4_1:    return 6;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_0:    return 6;
        case GGML_TYPE_Q5_1:    return 6;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_turing_plus(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 7;
        case GGML_TYPE_IQ3_S:   return 6;
        case GGML_TYPE_IQ3_XXS: return 7;
        case GGML_TYPE_MXFP4:   return 7;
        case GGML_TYPE_NVFP4:   return 8;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_gcn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 5;
        case GGML_TYPE_IQ1_M:   return 5;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 5;
        case GGML_TYPE_Q4_1:    return 5;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_cdna(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 5;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna1_rdna2(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_K:    return 6;
        case GGML_TYPE_Q6_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 6;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 7;
        case GGML_TYPE_IQ1_M:   return 7;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 7;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 5;
        case GGML_TYPE_NVFP4:   return 5;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 7;
        case GGML_TYPE_Q4_1:    return 7;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_0:    return 7;
        case GGML_TYPE_Q5_1:    return 7;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 5;
        case GGML_TYPE_Q8_0:    return 7;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

// Host function: returns the max batch size for the current arch+type at runtime.
int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    // NVIDIA: Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    // the device side sizes its launch bounds by the arch the code was compiled for, so this must too
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        const int arch = ggml_cuda_highest_compiled_arch(cc);
        if (arch == GGML_CUDA_CC_VOLTA || arch >= GGML_CUDA_CC_ADA_LOVELACE) {
            return MMVQ_MAX_BATCH_SIZE;
        }
        if (arch >= GGML_CUDA_CC_TURING) {
            return get_mmvq_mmid_max_batch_turing_plus(type);
        }
        return get_mmvq_mmid_max_batch_pascal_older(type);
    }

    // AMD
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            return get_mmvq_mmid_max_batch_rdna4(type);
        }
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            return get_mmvq_mmid_max_batch_rdna3(type);
        }
        if (GGML_CUDA_CC_IS_RDNA1(cc) || GGML_CUDA_CC_IS_RDNA2(cc)) {
            return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return get_mmvq_mmid_max_batch_cdna(type);
        }
        if (GGML_CUDA_CC_IS_GCN(cc)) {
            return get_mmvq_mmid_max_batch_gcn(type);
        }
    }
    return MMVQ_MAX_BATCH_SIZE;
}

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
    if (!ggml_is_quantized(type)) {
        return false;
    }
    // GGML_CUDA_MMVQ_MAX caps the batch this path will take, so the one below
    // it in the dispatch order gets a turn. mul_mat_vec_q reads each weight
    // block once and then does one integer dot product per column, so its cost
    // climbs with the batch in a way a bandwidth-bound kernel's does not; where
    // that crossover sits is worth measuring rather than assuming.
    {
        static const int cap = [] {
            const char * e = getenv("GGML_CUDA_MMVQ_MAX");
            return e ? atoi(e) : -1;
        }();
        if (cap >= 0 && ne11 > cap) {
            return false;
        }
    }
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        if (GGML_CUDA_CC_IS_CDNA1(cc)) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q5_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q8_0:
                    return ne11 <= 6;
                case GGML_TYPE_Q2_K:
                    return ne11 <= 4;
                case GGML_TYPE_Q3_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q4_K:
                    return ne11 <= 2;
                case GGML_TYPE_Q5_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q6_K:
                    return ne11 <= 4;
                case GGML_TYPE_IQ1_S:
                    return ne11 <= 5;
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ4_XS:
                    return ne11 <= 6;
                default:
                    return ne11 <= MMVQ_MAX_BATCH_SIZE;
            }
        }
        switch (type) { // tuned for CDNA2
            case GGML_TYPE_Q2_K:
                return ne11 <= 5;
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 3;
            case GGML_TYPE_Q6_K:
                return ne11 <= 5;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    return ne11 <= MMVQ_MAX_BATCH_SIZE;
}

// Device constexpr: returns the max batch size for the current arch+type at compile time.
template <ggml_type type>
static constexpr __device__ int get_mmvq_mmid_max_batch_for_device() {
#if defined(RDNA4)
    return get_mmvq_mmid_max_batch_rdna4(type);
#elif defined(RDNA3)
    return get_mmvq_mmid_max_batch_rdna3(type);
#elif defined(RDNA2) || defined(RDNA1)
    return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
#elif defined(CDNA)
    return get_mmvq_mmid_max_batch_cdna(type);
#elif defined(GCN)
    return get_mmvq_mmid_max_batch_gcn(type);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE)
    return MMVQ_MAX_BATCH_SIZE;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    return get_mmvq_mmid_max_batch_turing_plus(type);
#else
    return get_mmvq_mmid_max_batch_pascal_older(type);
#endif
}

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool small_k = false, bool halve_iters = false) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return 8;
                case GGML_TYPE_Q6_K:
                    return 2;
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_TURING) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    return 2;
                default:
                    return 4;
            }
        }
        switch (ncols_dst) {
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_GB10) {
        const int generic = calc_nwarps(type, ncols_dst, MMVQ_PARAMETERS_GENERIC);
        // Only worth the wider block when it actually retires the K loop in half the trips (Observation)
        if (ncols_dst == 1 && !small_k && halve_iters) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                    return 2 * generic;
                default:
                    break;
            }
        }
        return generic;
    }
    return 1;
}

static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING || table_id == MMVQ_PARAMETERS_GB10) {
        switch (ncols_dst) {
            case 1:
                return small_k ? nwarps : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

// fusion_mode 3 carries the convolution epilogue in place of the elementwise one
template <int fusion_mode>
using mmvq_epi_t = std::conditional_t<fusion_mode == 3, ggml_cuda_mmvq_conv, ggml_cuda_mm_epilogue_t<fusion_mode>>;

// a channel's state, taps and bias, loaded by its lane while the partial sums still reduce
struct mmvq_conv_operands {
    float s[3];
    float w[4];
    float b;
};

static __device__ __forceinline__ void mmvq_conv_load(const ggml_cuda_mmvq_conv & e, const int c, mmvq_conv_operands & o) {
    const float * s = e.state + (e.state_ids ? (int64_t) e.state_ids[0]*e.state_row : 0) + (int64_t) c*e.state_s1;
#pragma unroll
    for (int m = 0; m < 3; ++m) {
        o.s[m] = s[m];
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        o.w[j] = e.w[(int64_t) c*e.w_s1 + j];
    }
    o.b = e.bias != nullptr ? e.bias[c] : 0.0f;
}

// one channel's convolution as ssm_conv_f32 computes it, over the window [state, projected values], then its snapshots.
// The state was read before any snapshot is written, and a snapshot row that is the state row is this channel's alone.
template <int ncols_dst>
static __device__ __forceinline__ void mmvq_conv_row(const ggml_cuda_mmvq_conv & e, const int c, const float * v, const mmvq_conv_operands & o) {
    constexpr int d_conv = 4;
    float win[d_conv - 1 + ncols_dst];
#pragma unroll
    for (int m = 0; m < d_conv - 1; ++m) {
        win[m] = o.s[m];
    }
#pragma unroll
    for (int t = 0; t < ncols_dst; ++t) {
        win[d_conv - 1 + t] = v[t];
    }
    const float * w = o.w;
    const float   b = o.b;
#pragma unroll
    for (int t = 0; t < ncols_dst; ++t) {
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < d_conv; ++j) {
            sumf += win[t + j] * w[j];
        }
        sumf += b;
        e.y[(int64_t) t*e.y_s1 + c] = e.silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
    for (int k = 0; k < e.n_snap; ++k) {
        float * d = e.snap[k] + (int64_t) c*e.snap_s1;
#pragma unroll
        for (int i0 = 0; i0 <= ncols_dst; ++i0) {
            if (i0 == e.snap_idx[k]) {
#pragma unroll
                for (int m = 0; m < d_conv - 1; ++m) {
                    d[m] = win[i0 + m];
                }
            }
        }
    }
}

// fusion_mode: 0 none, 1 the lean epilogue (no gate), 2 gate, bias and scales, 3 the convolution
// one_row: each block takes a single row, for batched launches too small to fill the GPU with two
template <ggml_type type, int ncols_dst, int fusion_mode, bool small_k, bool one_row>
static __device__ __forceinline__ void mul_mat_vec_q_impl(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device & fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride, const uint32_t nrows_x, const ggml_cuda_mv_pf & pf, const bool release, const uint32_t nchannels_x,
        const mmvq_epi_t<fusion_mode> & epi, const uint32_t block_x) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr bool has_fusion = fusion_mode == 2;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id);
    constexpr int rows_per_cuda_block = one_row ? 1 : calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);
    static_assert(mmvq_type_size(type) > 0, "mmvq_type_size lacks this type");

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*block_x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    // constant weights can head into L2 while the producer finishes; an early id is final (an earlier matvec read it
    // after its own wait) and only steers the hint
#ifdef GGML_CUDA_PDL_DEVICE
    if (pf.bytes != 0 && threadIdx.y == 0) {
        const uint32_t channel_pf = ncols_dst == 1 && ids ? mmvq_ld_ids_early(ids + channel_dst) : fastdiv(channel_dst, channel_ratio);
        const int64_t  kb0 = (int64_t) fastdiv(blockIdx.z, sample_ratio)*stride_sample_x
            + (int64_t) channel_pf*stride_channel_x + (int64_t) row0*stride_row_x;
#pragma unroll
        for (int r = 0; r < rows_per_cuda_block*(has_fusion ? 2 : 1); ++r) {
            const int    i    = r % rows_per_cuda_block;
            const void * base = has_fusion && r >= rows_per_cuda_block ? pf.gate : pf.x;
            if (base != nullptr && channel_pf < nchannels_x && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < nrows_x)) {
                const int64_t kb = kb0 + (int64_t) i*stride_row_x;
                ggml_cuda_prefetch_l2_warp((const char *) base + kb*mmvq_type_size(type), pf.bytes, threadIdx.x);
            }
        }
    }
#else
    GGML_UNUSED_VARS(pf, nchannels_x);
#endif // GGML_CUDA_PDL_DEVICE
    ggml_cuda_pdl_sync();
    if (release) {
        ggml_cuda_pdl_lc();
    }
    channel_x  = ncols_dst == 1 && ids ? ids[channel_dst]                     : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    bool use_scale = false;
    bool use_gate_scale = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    const float * x_scale = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op active_glu;
    float glu_limit = 0.0f;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        glu_limit     = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            use_scale      = fusion.x_scale    != nullptr;
            use_gate_scale = fusion.gate_scale != nullptr && use_gate;
            x_scale        = (const float *) fusion.x_scale;
            gate_scale     = (const float *) fusion.gate_scale;
        }
    }


    [[maybe_unused]] float x_biases[ncols_dst]    = { 0.0f };
    [[maybe_unused]] float gate_biases[ncols_dst] = { 0.0f };
    [[maybe_unused]] float x_scales = 1.0f;
    [[maybe_unused]] float gate_scales = 1.0f;
    if constexpr (has_fusion) {
        // 1. Hide latency by prefetching bias, gates and scales here
        // 2. load only on threads that won't die after partial sum calculation
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
            if (use_bias) {
                x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if (use_gate_bias) {
                gate_bias = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if constexpr (type == GGML_TYPE_NVFP4) {
                if (use_scale) {
                    x_scales = x_scale[ids ? channel_x : 0];
                }
                if (use_gate_scale) {
                    gate_scales = gate_scale[ids ? channel_x : 0];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

        if constexpr (ncols_dst > 1 && has_vec_dot_multi(type)) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                float r[ncols_dst];
                vec_dot_multi<type, ncols_dst>(vx, &y[kby], stride_col_y, kbx_offset + i*stride_row_x + kbx, kqs, r);
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    tmp[j][i] += r[j];
                }
                if constexpr (has_fusion) {
                    if (use_gate) {
                        vec_dot_multi<type, ncols_dst>(vgate, &y[kby], stride_col_y, kbx_offset + i*stride_row_x + kbx, kqs, r);
#pragma unroll
                        for (int j = 0; j < ncols_dst; ++j) {
                            tmp_gate[j][i] += r[j];
                        }
                    }
                }
            }
        } else {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    tmp[j][i] += vec_dot_q_cuda(
                        vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            tmp_gate[j][i] += vec_dot_q_cuda(
                                vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                        }
                    }
                }
            }
        }
    }

    // the lanes that store load the epilogue's operands here, so the loads overlap the reduction
    [[maybe_unused]] float epi_arg[GGML_CUDA_MM_EPI_MAX_ARGS][ncols_dst] = {{0.0f}};
    if constexpr (fusion_mode == 1) {
        const int row = row0 + threadIdx.x;
        if (threadIdx.y == 0 && threadIdx.x < rows_per_cuda_block && (rows_per_cuda_block == 1 || uint32_t(row) < nrows_x)) {
#pragma unroll
            for (int k = 0; k < GGML_CUDA_MM_EPI_MAX_ARGS; ++k) {
                if (k < epi.n_args) {
#pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        epi_arg[k][j] = ggml_cuda_mm_epi_load(epi.arg[k], row, j, channel_dst, sample_dst);
                    }
                }
            }
        }
    }

    [[maybe_unused]] mmvq_conv_operands conv_o;
    if constexpr (fusion_mode == 3) {
        const int row = row0 + threadIdx.x;
        if (threadIdx.y == 0 && threadIdx.x < rows_per_cuda_block && (rows_per_cuda_block == 1 || uint32_t(row) < nrows_x)) {
            mmvq_conv_load(epi, row, conv_o);
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    [[maybe_unused]] __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // the convolution holds its row's values for every column until the last one is summed
    [[maybe_unused]] float conv_v[ncols_dst];

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }

            if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < nrows_x)) {
                float result = tmp[j][i];
                if constexpr (has_fusion) {
                    // the scale rounds on its own, as the separate multiply did, before a bias or residual is added
                    if constexpr (type == GGML_TYPE_NVFP4) {
                        result = __fmul_rn(result, x_scales);
                    }
                    result += x_biases[j];
                    if (use_gate) {
                        float gate_value = tmp_gate[j][i];
                        if constexpr (type == GGML_TYPE_NVFP4) {
                            gate_value = __fmul_rn(gate_value, gate_scales);
                        }
                        gate_value += gate_biases[j];
                        switch (active_glu) {
                            case GGML_GLU_OP_SWIGLU:
                                result *= ggml_cuda_op_silu_single(gate_value);
                                break;
                            case GGML_GLU_OP_GEGLU:
                                result *= ggml_cuda_op_gelu_single(gate_value);
                                break;
                            case GGML_GLU_OP_SWIGLU_OAI:
                                result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                                break;
                            case GGML_GLU_OP_SWIGLU_CLAMP:
                                result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                                break;
                            default:
                                result = result * gate_value;
                                break;
                        }
                    }
                }
                if constexpr (fusion_mode == 1) {
                    result = ggml_cuda_mm_epi_apply(epi, result, epi_arg[0][j], epi_arg[1][j], epi_arg[2][j]);
                }
                if constexpr (fusion_mode == 3) {
                    conv_v[j] = result;
                } else {
                    dst[j*stride_col_dst + i] = result;
                }
            }
        }
    }

    if constexpr (fusion_mode == 3) {
        if (threadIdx.x < rows_per_cuda_block && (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
            mmvq_conv_row<ncols_dst>(epi, row0 + threadIdx.x, conv_v, conv_o);
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_scale, use_gate_scale, active_glu, glu_limit, gate_bias, x_bias, x_scale, gate_scale, tmp_gate);
    }
    if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(use_scale, use_gate_scale, x_scale, gate_scale, x_scales, gate_scales);
    }
}

template <ggml_type type, int ncols_dst, int fusion_mode, bool small_k = false, bool one_row = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id())*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride, const uint32_t nrows_x, const ggml_cuda_mv_pf pf, const bool release, const uint32_t nchannels_x,
        const ggml_cuda_mm_epilogue_t<fusion_mode> epi) {
    mul_mat_vec_q_impl<type, ncols_dst, fusion_mode, small_k, one_row>(
        vx_ptr, vy_ptr, ids_ptr, fusion, dst_ptr, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio,
        stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, nrows_x, pf, release, nchannels_x, epi, blockIdx.x);
}

// matrices over one activation in one launch: each block belongs to one member and computes its rows
// exactly as mul_mat_vec_q does for that member alone
struct mmvq_member {
    const void * vx;
    float      * dst;
    uint32_t     nrows;
    uint32_t     stride_row_x;
    uint32_t     stride_col_dst;
    uint32_t     block_end;
};

template <int nm>
struct mmvq_members {
    mmvq_member m[nm];
};

template <ggml_type type, int ncols_dst, bool small_k, int nm>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id())*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_members(const void * vy, const uint32_t ncols_x, const uint32_t stride_col_y, const mmvq_members<nm> g) {
    const void * vx             = g.m[0].vx;
    float      * dst            = g.m[0].dst;
    uint32_t     nrows          = g.m[0].nrows;
    uint32_t     stride_row_x   = g.m[0].stride_row_x;
    uint32_t     stride_col_dst = g.m[0].stride_col_dst;
    uint32_t     block0         = 0;
#pragma unroll
    for (int k = 1; k < nm; ++k) {
        if (blockIdx.x >= g.m[k - 1].block_end) {
            vx             = g.m[k].vx;
            dst            = g.m[k].dst;
            nrows          = g.m[k].nrows;
            stride_row_x   = g.m[k].stride_row_x;
            stride_col_dst = g.m[k].stride_col_dst;
            block0         = g.m[k - 1].block_end;
        }
    }

    const ggml_cuda_mm_fusion_args_device fusion{};
    const ggml_cuda_mv_pf pf{};
    const uint3 one = make_uint3(0, 0, 1);
    mul_mat_vec_q_impl<type, ncols_dst, 0, small_k, false>(
        vx, vy, nullptr, fusion, dst, ncols_x, one, stride_row_x, stride_col_y, stride_col_dst,
        one, 0, 0, 0, one, 0, 0, 0, 0, nrows, pf, false, 1, ggml_cuda_mm_epilogue_none{}, blockIdx.x - block0);
}

// the input projection with its convolution: rows split into blocks as mul_mat_vec_q splits them, so every row sums
// as it would there
template <ggml_type type, int ncols_dst>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id())*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_conv(const void * vx, const void * vy, const uint32_t ncols_x, const uint32_t stride_row_x,
        const uint32_t stride_col_y, const uint32_t nrows, const ggml_cuda_mmvq_conv conv) {
    const ggml_cuda_mm_fusion_args_device fusion{};
    const ggml_cuda_mv_pf pf{};
    const uint3 one = make_uint3(0, 0, 1);
    mul_mat_vec_q_impl<type, ncols_dst, 3, false, false>(
        vx, vy, nullptr, fusion, conv.y, ncols_x, one, stride_row_x, stride_col_y, 0,
        one, 0, 0, 0, one, 0, 0, 0, 0, nrows, pf, false, 1, conv, blockIdx.x);
}

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / (c_rows_per_block*lane_groups)), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
// With lane_groups > 1 a row fits in a fraction of a warp, and each lane group takes its own rows.
template <ggml_type type, int c_rows_per_block, int lane_groups, bool has_fusion>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr,
        const ggml_cuda_mm_fusion_args_device fusion,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride, const bool release,
        const float * rw, const uint32_t rw_s_slot, const uint32_t rw_s_token) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    // the gate projection shares the expert layout of vx, so the same
    // offsets read it; its dot product feeds the GLU at the end
    [[maybe_unused]] const void * GGML_CUDA_RESTRICT vgate = has_fusion ? fusion.gate : nullptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    constexpr int group_size = warp_size/lane_groups;
    static_assert(group_size % (qi/vdr) == 0, "a block's lanes must fit in one lane group");

    const uint32_t token_idx   = threadIdx.y;
    const int      lane        = threadIdx.x % group_size;
    const int      row0        = c_rows_per_block*(lane_groups*blockIdx.x + threadIdx.x/group_size);
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * group_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    ggml_cuda_pdl_sync();
    if (release) {
        ggml_cuda_pdl_lc();
    }
    const uint32_t channel_x = ids[channel_dst + token_idx * ids_stride];
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    const int kbx_offset  = channel_x*stride_channel_x + row0*stride_row_x;

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};
    [[maybe_unused]] float tmp_gate[c_rows_per_block] = {0.0f};

    for (int kbx = lane / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (lane % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
            if constexpr (has_fusion) {
                tmp_gate[i] += vec_dot_q_cuda(vgate, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
            }
        }
    }

    if (!release) {
        ggml_cuda_pdl_lc();
    }

    // Warp-level reduction only - no shared memory needed.
    // the other groups' lanes would add +0 at the skipped steps; one +0 add gives the same bits
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        if constexpr (lane_groups > 1) {
            tmp[i] = __fadd_rn(tmp[i], 0.0f);
            if constexpr (has_fusion) {
                tmp_gate[i] = __fadd_rn(tmp_gate[i], 0.0f);
            }
        }
        tmp[i] = warp_reduce_sum<group_size>(tmp[i]);
        if constexpr (has_fusion) {
            tmp_gate[i] = warp_reduce_sum<group_size>(tmp_gate[i]);
        }
    }

    // Write results
    if (lane < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + lane) < nrows_x)) {
        float result = tmp[lane];
        // an expert's bias is its own row of the [nrows, n_expert] bias tensor
        const uint32_t bias_idx = channel_x*stride_channel_dst + row0 + lane;
        // an expert's own scale (NVFP4's second-level scale), applied where the separate multiply would have been
        if constexpr (type == GGML_TYPE_NVFP4) {
            if (fusion.x_scale != nullptr) {
                result *= ((const float *) fusion.x_scale)[channel_x];
            }
        }
        if (fusion.x_bias != nullptr) {
            result += ((const float *) fusion.x_bias)[bias_idx];
        }
        if constexpr (has_fusion) {
            float gate_value = tmp_gate[lane];
            if constexpr (type == GGML_TYPE_NVFP4) {
                if (fusion.gate_scale != nullptr) {
                    gate_value *= ((const float *) fusion.gate_scale)[channel_x];
                }
            }
            if (fusion.gate_bias != nullptr) {
                gate_value += ((const float *) fusion.gate_bias)[bias_idx];
            }
            switch (fusion.glu_op) {
                case GGML_GLU_OP_SWIGLU:
                    result *= ggml_cuda_op_silu_single(gate_value);
                    break;
                case GGML_GLU_OP_GEGLU:
                    result *= ggml_cuda_op_gelu_single(gate_value);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                    break;
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, fusion.glu_limit);
                    break;
                default:
                    result = result * gate_value;
                    break;
            }
        }
        if (rw != nullptr) {
            result = result * rw[channel_dst*rw_s_slot + token_idx*rw_s_token];
        }
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + lane] = result;
    }
}

template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false, const bool one_row = false) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id);
    const int rpb = one_row ? 1 : calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

template<ggml_type type, int c_ncols_dst, bool small_k = false, bool one_row = false>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion,
        const ggml_cuda_mm_epilogue * epi, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t ids_stride, const uint32_t nrows_x, const ggml_cuda_mv_pf & pf, const bool release, const uint32_t nchannels_x,
        cudaStream_t stream) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
    if constexpr (c_ncols_dst <= MMVQ_MAX_BATCH_SIZE) {
        if (has_fusion) {
            GGML_ASSERT(epi == nullptr);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, 2, small_k, one_row>, launch_params,
                 vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, nrows_x, pf, release, nchannels_x,
                 ggml_cuda_mm_epilogue_none{});
            return;
        }
        if constexpr (type != GGML_TYPE_NVFP4) {
            if (epi != nullptr) {
                const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
                ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, 1, small_k, one_row>, launch_params,
                     vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                     channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                     sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, nrows_x, pf, release, nchannels_x,
                     *epi);
                return;
            }
        }
    }

    GGML_ASSERT(!has_fusion && epi == nullptr && "fusion only supported for ncols_dst=1");

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, 0, small_k, one_row>, launch_params,
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, nrows_x, pf, release, nchannels_x,
        ggml_cuda_mm_epilogue_none{});
}

// lanes a row of this type and width needs, and the lane groups that let several rows share a warp
// (each group must hold whole blocks and a whole row, and every group must get two real rows)
template <ggml_type type>
static int mmvq_lane_groups(const int ncols_x, const int nrows_x, const int warp_size) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int lpb = ggml_cuda_type_traits<type>::qi / get_vdr_mmvq(type);
    const int     tpr = ncols_x/qk * lpb;
    for (const int g : {4, 2}) {
        if (tpr*g <= warp_size && lpb*g <= warp_size && nrows_x % (2*g) == 0) {
            return g;
        }
    }
    return 1;
}

// GGML_CUDA_MMVQ_LANE_GROUPS and GGML_CUDA_MMVQ_ONE_ROW force a CTA shape for tests; -1 leaves it to the rule
static int mmvq_force_env(const char * name) {
    const char * e = getenv(name);
    return e ? atoi(e) : -1;
}

struct mmvq_router_weights {
    const float * w       = nullptr;
    uint32_t      s_slot  = 0;
    uint32_t      s_token = 0;
};

static thread_local mmvq_router_weights g_mmvq_router_weights;
static thread_local bool                g_mmvq_router_weights_taken = false;

void ggml_cuda_mmvq_set_router_weights(const float * w, const int64_t s_slot, const int64_t s_token) {
    g_mmvq_router_weights       = { w, (uint32_t) s_slot, (uint32_t) s_token };
    g_mmvq_router_weights_taken = false;
}

bool ggml_cuda_mmvq_router_weights_taken() {
    return g_mmvq_router_weights_taken;
}

template <ggml_type type>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, const int lane_groups, const bool release, cudaStream_t stream) {

    constexpr int rows_per_block = 2; // 2 gives best perf based on tuning
    constexpr int lpb = ggml_cuda_type_traits<type>::qi / get_vdr_mmvq(type);

    // the gate, the per-expert scales and the biases all travel with this kernel

    const mmvq_router_weights rw = g_mmvq_router_weights;
    g_mmvq_router_weights = {};
    g_mmvq_router_weights_taken = rw.w != nullptr;

    const auto launch = [&](auto c_lane_groups) {
        constexpr int g = decltype(c_lane_groups)::value;
        const int64_t nblocks_rows = (nrows_x + rows_per_block*g - 1) / (rows_per_block*g);
        const dim3 block_nums(nblocks_rows, nchannels_dst);
        const dim3 block_dims(warp_size, ncols_dst);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        if (fusion.gate != nullptr) {
            ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, g, true>, launch_params,
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
                stride_row_x, stride_col_y, stride_col_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst,
                ncols_dst, ids_stride, release, rw.w, rw.s_slot, rw.s_token);
            return;
        }
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, g, false>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, release, rw.w, rw.s_slot, rw.s_token);
    };

    if constexpr (lpb <= 8) {
        if (lane_groups == 4 && warp_size == 32) {
            launch(std::integral_constant<int, 4>{});
            return;
        }
    }
    if constexpr (lpb <= 16) {
        if (lane_groups == 2 && warp_size == 32) {
            launch(std::integral_constant<int, 2>{});
            return;
        }
    }
    launch(std::integral_constant<int, 1>{});
}

// When K is small, increase rows_per_block to match nwarps so each warp has more work to do
template <ggml_type type>
static bool mmvq_use_small_k(const int ncols_x, const int ncols_dst, const int c_ncols_dst, const int cc, const int warp_size,
        const mmvq_parameter_table_id table_id) {
    // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
    constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
    constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr                   = get_vdr_mmvq(type);
    const int     blocks_per_row_x      = ncols_x / qk;
    const int     blocks_per_iter_1warp = vdr * warp_size / qi;
    const int     nwarps                = calc_nwarps(type, c_ncols_dst, table_id);
    // a row the block covers in one iteration, equality included: a warp per row is the same sum in the same order
    bool          use                   = nwarps > 1 && blocks_per_row_x <= nwarps * blocks_per_iter_1warp;

    constexpr std::array<ggml_type, 2> iq_slow_turing = {
        GGML_TYPE_IQ3_XXS,
        GGML_TYPE_IQ3_S,
    };
    constexpr std::array<ggml_type, 8> iq_slow_other = {
        GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
        GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
    };
    constexpr std::array<ggml_type, 3> slow_pascal = {
        GGML_TYPE_IQ3_S,
        GGML_TYPE_Q2_K,
        GGML_TYPE_Q3_K,
    };

    const bool is_nvidia_turing_plus  = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
    const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;

    if (is_nvidia_turing_plus) {
        if (ncols_dst == 1 &&
                std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end()) {
            use = false;
        }
    } else if ((ncols_dst == 1 && std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end()) ||
            (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
            GGML_CUDA_CC_IS_RDNA(cc)) {
        use = false;
    }

    return use;
}

template <ggml_type type>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion,
        const ggml_cuda_mm_epilogue * epi, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, const bool pf_ok, const bool release, cudaStream_t stream) {
    // pf_ok with ids: the ids are final before this launch's wait (see ggml_cuda_mul_mat_vec_q)

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int                     cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id  = get_device_table_id(cc);

    const bool has_ids = ids != nullptr;

    // How the K loop divides up at the baseline block width, both decisions below use these.
    constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
    constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr                   = get_vdr_mmvq(type);
    const int     blocks_per_row_x      = ncols_x / qk;
    const int     blocks_per_iter_1warp = vdr * warp_size / qi;

    const auto should_use_small_k = [&](int c_ncols_dst) {
        return mmvq_use_small_k<type>(ncols_x, ncols_dst, c_ncols_dst, cc, warp_size, table_id);
    };

    const int      nsm        = ggml_cuda_info().devices[device].nsm;
    const int64_t  row_bytes  = (int64_t) (ncols_x/ggml_blck_size(type))*ggml_type_size(type);
    const bool     has_gate   = fusion.gate != nullptr;
    const uint64_t fo         = ggml_cuda_fuse_off_mask();
    const bool     shape_ok   = GGML_CUDA_CC_IS_NVIDIA(cc);

    // several rows per warp when a row takes a fraction of it; only worth it while the launch is more than a wave
    static const int force_lane_groups = mmvq_force_env("GGML_CUDA_MMVQ_LANE_GROUPS");
    const int lane_groups = shape_ok && !(fo & GGML_CUDA_FUSE_MMV_LANE_GROUPS) && has_ids ? mmvq_lane_groups<type>(ncols_x, nrows_x, warp_size) : 1;
    const auto moe_waves_over_one = [&](const int ncols) {
        const void * kernel = has_gate ? (const void *) mul_mat_vec_q_moe<type, 2, 1, true> : (const void *) mul_mat_vec_q_moe<type, 2, 1, false>;
        return (int64_t) ((nrows_x + 1)/2)*nchannels_dst > (int64_t) nsm*ggml_cuda_max_active_blocks(kernel, warp_size*ncols, 0);
    };
    const auto pick_lane_groups = [&](const bool over_one_wave) {
        if (force_lane_groups > 0) {
            return lane_groups > 1 ? std::min(force_lane_groups, lane_groups) : 1;
        }
        return over_one_wave ? lane_groups : 1;
    };


    if (has_ids && ncols_dst > 1) {
        GGML_ASSERT(epi == nullptr);
        // Multi-token MUL_MAT_ID path - dedicated MoE kernel
        mul_mat_vec_q_moe_launch<type>(
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst,
            lane_groups > 1 ? pick_lane_groups(moe_waves_over_one(ncols_dst)) : 1, release, stream);
        return;
    }

    static const int force_one_row = mmvq_force_env("GGML_CUDA_MMVQ_ONE_ROW");

    const auto launch = [&](auto c_ncols_dst, auto c_small_k) {
        constexpr int  ncols   = decltype(c_ncols_dst)::value;
        constexpr bool small_k = decltype(c_small_k)::value;
        const std::pair<dim3, dim3> dims = calc_launch_params<type>(ncols, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id, small_k);
        const int     threads = dims.second.x*dims.second.y;
        const int64_t nctas   = (int64_t) dims.first.x*dims.first.y*dims.first.z;

        // a single token's experts, their rows short enough to share a warp: the MoE kernel takes them in
        // lane groups, which is the same sum in the same order, when today's grid is more than a wave
        if constexpr (ncols == 1) {
            if (lane_groups > 1 && nsamples_dst == 1 && epi == nullptr && fusion.x_bias == nullptr && fusion.gate_bias == nullptr &&
                    fusion.x_scale == nullptr && fusion.gate_scale == nullptr) {
                const void * kernel = has_gate ? (const void *) mul_mat_vec_q<type, 1, 2, small_k> : (const void *) mul_mat_vec_q<type, 1, 0, small_k>;
                const int g = pick_lane_groups(nctas > (int64_t) nsm*ggml_cuda_max_active_blocks(kernel, threads, 0));
                if (g > 1) {
                    mul_mat_vec_q_moe_launch<type>(
                        vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x,
                        stride_row_x, stride_col_y, stride_col_dst,
                        stride_channel_x, stride_channel_y, stride_channel_dst,
                        1, ids_stride, warp_size, nchannels_dst, g, release, stream);
                    return;
                }
            }
        }

        // MSVC does not take the enclosing lambda's constexpr locals as template arguments here, so run derives its own
        const auto run = [&](auto c_n, auto c_sk, auto c_one_row) {
            constexpr int  n   = decltype(c_n)::value;
            constexpr bool sk  = decltype(c_sk)::value;
            constexpr bool or1 = decltype(c_one_row)::value;
            const std::pair<dim3, dim3> d = or1 ? calc_launch_params<type>(n, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id, false, true) : dims;
            ggml_cuda_mv_pf pf;
            if (pf_ok && ggml_cuda_kernel_uses_pdl((const void *) mul_mat_vec_q<type, n, 0, sk>)) {
                const int rpb = or1 ? 1 : calc_rows_per_block(n, table_id, sk, d.second.y);
                pf.x     = vx;
                pf.gate  = fusion.gate;
                pf.bytes = ggml_cuda_mv_prefetch_bytes(row_bytes, rpb*(has_gate ? 2 : 1), (int64_t) d.first.x*d.first.y*d.first.z, nsm);
            }
            mul_mat_vec_q_switch_fusion<type, n, sk, or1>(
                vx, vy, ids, fusion, epi, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                stride_sample_x, stride_sample_y, stride_sample_dst, d.first, d.second, 0, ids_stride, nrows_x,
                pf, release, nchannels_x, stream);
        };

        // one row per block when two-row blocks fill at most half a wave and one-row blocks still fit in one
        if constexpr (ncols > 1 && !small_k) {
            bool one_row = false;
            if (shape_ok && !(fo & GGML_CUDA_FUSE_MMV_ONE_ROW)) {
                if (force_one_row >= 0) {
                    one_row = force_one_row > 0;
                } else {
                    const bool    fused  = has_gate || fusion.x_bias || fusion.gate_bias || fusion.x_scale || fusion.gate_scale;
                    const int64_t nctas1 = (int64_t) nrows_x*dims.first.y*dims.first.z;
                    const void *  k2     = fused ? (const void *) mul_mat_vec_q<type, ncols, 2, false, false>
                                                 : (const void *) mul_mat_vec_q<type, ncols, 0, false, false>;
                    const void *  k1     = fused ? (const void *) mul_mat_vec_q<type, ncols, 2, false, true>
                                                 : (const void *) mul_mat_vec_q<type, ncols, 0, false, true>;
                    one_row = 2*nctas <= (int64_t) nsm*ggml_cuda_max_active_blocks(k2, threads, 0) &&
                                nctas1 <= (int64_t) nsm*ggml_cuda_max_active_blocks(k1, threads, 0);
                }
            }
            if (one_row) {
                run(c_ncols_dst, c_small_k, std::true_type{});
                return;
            }
        }
        run(c_ncols_dst, c_small_k, std::false_type{});
    };

    switch (ncols_dst) {
        case 1:
            if (should_use_small_k(1)) {
                launch(std::integral_constant<int, 1>{}, std::true_type{});
            } else {
                launch(std::integral_constant<int, 1>{}, std::false_type{});
            }
            break;
        case 2: launch(std::integral_constant<int, 2>{}, std::false_type{}); break;
        case 3: launch(std::integral_constant<int, 3>{}, std::false_type{}); break;
        case 4: launch(std::integral_constant<int, 4>{}, std::false_type{}); break;
        case 5: launch(std::integral_constant<int, 5>{}, std::false_type{}); break;
        case 6: launch(std::integral_constant<int, 6>{}, std::false_type{}); break;
        case 7: launch(std::integral_constant<int, 7>{}, std::false_type{}); break;
        case 8: launch(std::integral_constant<int, 8>{}, std::false_type{}); break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}
static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion,
        const ggml_cuda_mm_epilogue * epi, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, const bool pf_ok, const bool release, cudaStream_t stream) {
    switch (type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q1_0>
                (vx, vy, ids, fusion, epi, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, pf_ok, release, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_0>
                (vx, vy, ids, fusion, epi, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, pf_ok, release, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, fusion, epi, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, pf_ok, release, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, fusion, epi, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, pf_ok, release, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, fusion, epi, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, pf_ok, release, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, fusion, epi, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, pf_ok, release, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, fusion, epi, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, pf_ok, release, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4>
                (vx, vy, ids, fusion, epi, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, pf_ok, release, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NVFP4>
                (vx, vy, ids, fusion, epi, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, pf_ok, release, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, fusion, epi, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, pf_ok, release, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, fusion, epi, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, pf_ok, release, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, fusion, epi, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, pf_ok, release, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, fusion, epi, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, pf_ok, release, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, fusion, epi, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, pf_ok, release, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, fusion, epi, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, pf_ok, release, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, fusion, epi, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, pf_ok, release, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, fusion, epi, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, pf_ok, release, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, fusion, epi, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, pf_ok, release, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, fusion, epi, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, pf_ok, release, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, fusion, epi, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, pf_ok, release, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, fusion, epi, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, pf_ok, release, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, fusion, epi, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, pf_ok, release, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, fusion, epi, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, pf_ok, release, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion, ggml_cuda_norm_fold * fold, const void * q8_pre) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE);

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};
    const ggml_cuda_mm_epilogue * epi = fusion ? fusion->epi : nullptr;
    GGML_ASSERT(!epi || (!ids && src0->type != GGML_TYPE_NVFP4));

    if (fusion) {
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        GGML_ASSERT( !ids || dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc));
        GGML_ASSERT(  ids || dst->ne[1] <= MMVQ_MAX_BATCH_SIZE);
        // Scale fusion is only allowed for NVFP4 currently as the cost of checking this at run-time in the prologue is
        // non-negligible for some models such as gpt-oss-20b
        GGML_ASSERT((fusion->x_scale == nullptr && fusion->gate_scale == nullptr) || src0->type == GGML_TYPE_NVFP4);

        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        if (fusion->x_scale) {
            GGML_ASSERT(fusion->x_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->x_scale));
            GGML_ASSERT(ggml_nelements(fusion->x_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.x_scale = fusion->x_scale->data;
        }
        if (fusion->gate_scale) {
            GGML_ASSERT(fusion->gate_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->gate_scale));
            GGML_ASSERT(ggml_nelements(fusion->gate_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.gate_scale = fusion->gate_scale->data;
        }
        fusion_local.glu_op = fusion->glu_op;
        fusion_local.glu_limit = fusion->glu_limit;
    }

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    // without PDL the matvec starts only after its producer ends, so the quantize asks for its first rows
    // (opt-in with GGML_CUDA_PF_PRODUCER=1 until measured on a card without PDL)
    ggml_cuda_l2_pf l2_pf;
    {
        static const bool   pf_producer = [] { const char * e = getenv("GGML_CUDA_PF_PRODUCER"); return e && atoi(e) != 0; }();
        const ggml_tensor * gate = fusion ? fusion->gate : nullptr;
        const int           nsm  = ggml_cuda_info().devices[ctx.device].nsm;
        if (pf_producer && !ids && ne02 == 1 && ne03 == 1 && !(ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_MMV_PRODUCER_PF) &&
                GGML_CUDA_CC_IS_NVIDIA(ggml_cuda_info().devices[ctx.device].cc) &&
                ggml_cuda_tensor_is_constant(src0) && (gate == nullptr || ggml_cuda_tensor_is_constant(gate))) {
            const int     nstreams  = gate ? 2 : 1;
            const int64_t row_bytes = ne00/ggml_blck_size(src0->type)*ts_src0;
            const int64_t pf_row    = ggml_cuda_mv_prefetch_bytes(row_bytes, nstreams, ne01, nsm);
            l2_pf.base[0]    = (const char *) src0->data;
            l2_pf.base[1]    = gate ? (const char *) gate->data : nullptr;
            l2_pf.row_bytes  = (uint32_t) pf_row;
            l2_pf.row_stride = nb01;
            l2_pf.nrows      = (uint32_t) std::min<int64_t>(ne01, (int64_t) nsm*32768/(pf_row*nstreams));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t  q8_bytes    = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
    char * q8 = nullptr;
    const int64_t s11_src = src1->nb[1] / ts_src1;
    const int64_t s12_src = src1->nb[2] / ts_src1;
    const int64_t s13_src = src1->nb[3] / ts_src1;
    if (q8_pre) {
        GGML_ASSERT(!fold);
        q8 = (char *) q8_pre;
    } else if (fold) {
        GGML_ASSERT(fold->x->type == GGML_TYPE_F32 && fold->w->type == GGML_TYPE_F32 && fold->q8_bytes == q8_bytes);
        ggml_cuda_norm_fold_reserve(ctx, *fold);
        // the first quantized reader quantizes, and writes the row factors for any float reader.
        // src1 may be a plain view of the product, so its own strides walk the norm's input
        if (!fold->q8_ready) {
            quantize_row_q8_1_rms_cuda((const float *) fold->x->data, (const float *) fold->w->data,
                ggml_cuda_tensor_is_constant(fold->w), fold->eps, fold->q8, fold->scale,
                ne10, s11_src, s12_src, s13_src, ne10_padded, ne11, ne12, ne13, l2_pf, stream);
            fold->q8_ready    = true;
            fold->scale_ready = true;
        }
        q8 = (char *) fold->q8;
    } else {
        q8 = src1_q8_1.alloc(q8_bytes);
        quantize_row_q8_1_pf_cuda(src1_d, q8, ne10, s11_src, s12_src, s13_src, ne10_padded, ne11, ne12, ne13, l2_pf, stream);
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    // expert ids are final before this launch's wait when an earlier matvec on this stream read them after its
    // own: every kernel after that one started after its wait returned, so their producer had completed
    const uint64_t fo        = ggml_cuda_fuse_off_mask();
    const bool     ids_early = ids && ncols_dst == 1 && !(fo & GGML_CUDA_FUSE_MMV_IDS_EARLY) &&
        ctx.ids_ready == ids && ctx.ids_ready_stream == ctx.curr_stream_no;
    const bool     pf_ok     = (!ids || ids_early) && !(fo & GGML_CUDA_FUSE_MMV_PREFETCH) && ggml_cuda_tensor_is_constant(src0) &&
        (fusion == nullptr || fusion->gate == nullptr || ggml_cuda_tensor_is_constant(fusion->gate));
    const bool     release   = !(fo & GGML_CUDA_FUSE_MMV_RELEASE);

    if (!ids && ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1 &&
            ggml_cuda_mmvq_mma_supported(src0->type, ggml_cuda_info().devices[ctx.device].cc, ncols_dst)) {
        ggml_cuda_mmvq_mma(src0->data, src0->type, q8, fusion_local, epi, dst_d, ne00, ne01, ncols_dst, s01, stride_col_y, stride_col_dst, stream);
        return;
    }

    mul_mat_vec_q_switch_type(
        src0->data, src0->type, q8, ids_d, fusion_local, epi, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           s03, s13,              s3,               ids_stride, pf_ok, release, stream);

    if (ids) {
        ctx.ids_ready        = ids;
        ctx.ids_ready_stream = ctx.curr_stream_no;
    }
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    ggml_cuda_mm_fusion_args_device fusion_local{};
    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, fusion_local, nullptr, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, false, false, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}

// launches a pair, or with wave_rows set returns the rows one wave covers. Four member slots: with two, nvcc picks the
// member through the parameters' address and the loads lose their global address space
template <ggml_type type, int ncols_dst, bool small_k>
static void mul_mat_vec_q_pair_launch(const void * vy, const uint32_t ncols_x, const uint32_t stride_col_y, const mmvq_member * mb,
        const int nwarps, const int rows_per_block, const int device, const int warp_size, cudaStream_t stream, int64_t * wave_rows) {
    const auto kernel = mul_mat_vec_q_members<type, ncols_dst, small_k, 4>;
    if (wave_rows) {
        int blocks_per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, nwarps*warp_size, 0));
        *wave_rows = (int64_t) blocks_per_sm * ggml_cuda_info().devices[device].nsm * rows_per_block;
        return;
    }
    const mmvq_members<4> g = {{ mb[0], mb[1], mb[1], mb[1] }};
    const ggml_cuda_kernel_launch_params launch_params(dim3(mb[1].block_end, 1, 1), dim3(warp_size, nwarps, 1), 0, stream);
    ggml_cuda_kernel_launch(kernel, launch_params, vy, ncols_x, stride_col_y, g);
}

template <ggml_type type>
static void mul_mat_vec_q_pair_switch_ncols_dst(const void * vy, const uint32_t ncols_x, const uint32_t stride_col_y, mmvq_member * mb,
        const int ncols_dst, const int device, cudaStream_t stream, int64_t * wave_rows) {
    const int cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id = get_device_table_id(cc);
    const bool small_k = ncols_dst == 1 && mmvq_use_small_k<type>(ncols_x, ncols_dst, 1, cc, warp_size, table_id);
    const int  nwarps  = calc_nwarps(type, ncols_dst, table_id);
    const int  rpb     = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    mb[0].block_end = (mb[0].nrows + rpb - 1) / rpb;
    mb[1].block_end = mb[0].block_end + (mb[1].nrows + rpb - 1) / rpb;
    switch (ncols_dst) {
        case 1:
            if (small_k) {
                mul_mat_vec_q_pair_launch<type, 1, true>(vy, ncols_x, stride_col_y, mb, nwarps, rpb, device, warp_size, stream, wave_rows);
            } else {
                mul_mat_vec_q_pair_launch<type, 1, false>(vy, ncols_x, stride_col_y, mb, nwarps, rpb, device, warp_size, stream, wave_rows);
            }
            break;
        case 2: mul_mat_vec_q_pair_launch<type, 2, false>(vy, ncols_x, stride_col_y, mb, nwarps, rpb, device, warp_size, stream, wave_rows); break;
        case 3: mul_mat_vec_q_pair_launch<type, 3, false>(vy, ncols_x, stride_col_y, mb, nwarps, rpb, device, warp_size, stream, wave_rows); break;
        case 4: mul_mat_vec_q_pair_launch<type, 4, false>(vy, ncols_x, stride_col_y, mb, nwarps, rpb, device, warp_size, stream, wave_rows); break;
        case 5: mul_mat_vec_q_pair_launch<type, 5, false>(vy, ncols_x, stride_col_y, mb, nwarps, rpb, device, warp_size, stream, wave_rows); break;
        case 6: mul_mat_vec_q_pair_launch<type, 6, false>(vy, ncols_x, stride_col_y, mb, nwarps, rpb, device, warp_size, stream, wave_rows); break;
        case 7: mul_mat_vec_q_pair_launch<type, 7, false>(vy, ncols_x, stride_col_y, mb, nwarps, rpb, device, warp_size, stream, wave_rows); break;
        case 8: mul_mat_vec_q_pair_launch<type, 8, false>(vy, ncols_x, stride_col_y, mb, nwarps, rpb, device, warp_size, stream, wave_rows); break;
        default: GGML_ABORT("fatal error");
    }
}

static void mul_mat_vec_q_pair_switch_type(const ggml_type type, const void * vy, const uint32_t ncols_x, const uint32_t stride_col_y,
        mmvq_member * mb, const int ncols_dst, const int device, cudaStream_t stream, int64_t * wave_rows) {
    switch (type) {
#define MMVQ_PAIR_CASE(t) case t: mul_mat_vec_q_pair_switch_ncols_dst<t>(vy, ncols_x, stride_col_y, mb, ncols_dst, device, stream, wave_rows); break;
        MMVQ_PAIR_CASE(GGML_TYPE_Q1_0)
        MMVQ_PAIR_CASE(GGML_TYPE_Q2_0)
        MMVQ_PAIR_CASE(GGML_TYPE_Q4_0)
        MMVQ_PAIR_CASE(GGML_TYPE_Q4_1)
        MMVQ_PAIR_CASE(GGML_TYPE_Q5_0)
        MMVQ_PAIR_CASE(GGML_TYPE_Q5_1)
        MMVQ_PAIR_CASE(GGML_TYPE_Q8_0)
        MMVQ_PAIR_CASE(GGML_TYPE_MXFP4)
        MMVQ_PAIR_CASE(GGML_TYPE_Q2_K)
        MMVQ_PAIR_CASE(GGML_TYPE_Q3_K)
        MMVQ_PAIR_CASE(GGML_TYPE_Q4_K)
        MMVQ_PAIR_CASE(GGML_TYPE_Q5_K)
        MMVQ_PAIR_CASE(GGML_TYPE_Q6_K)
        MMVQ_PAIR_CASE(GGML_TYPE_IQ2_XXS)
        MMVQ_PAIR_CASE(GGML_TYPE_IQ2_XS)
        MMVQ_PAIR_CASE(GGML_TYPE_IQ2_S)
        MMVQ_PAIR_CASE(GGML_TYPE_IQ3_XXS)
        MMVQ_PAIR_CASE(GGML_TYPE_IQ1_S)
        MMVQ_PAIR_CASE(GGML_TYPE_IQ1_M)
        MMVQ_PAIR_CASE(GGML_TYPE_IQ4_NL)
        MMVQ_PAIR_CASE(GGML_TYPE_IQ4_XS)
        MMVQ_PAIR_CASE(GGML_TYPE_IQ3_S)
#undef MMVQ_PAIR_CASE
        default: GGML_ABORT("fatal error");
    }
}

template <ggml_type type, int ncols_dst>
static void mul_mat_vec_q_conv_launch(const void * vx, const void * vy, const uint32_t ncols_x, const uint32_t stride_row_x,
        const uint32_t stride_col_y, const uint32_t nrows, const ggml_cuda_mmvq_conv & conv, const int device, cudaStream_t stream) {
    const int cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id = get_device_table_id(cc);
    const int nwarps = calc_nwarps(type, ncols_dst, table_id);
    const int rpb    = calc_rows_per_block(ncols_dst, table_id, false, nwarps);
    const ggml_cuda_kernel_launch_params launch_params(dim3((nrows + rpb - 1) / rpb, 1, 1), dim3(warp_size, nwarps, 1), 0, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_q_conv<type, ncols_dst>, launch_params, vx, vy, ncols_x, stride_row_x, stride_col_y, nrows, conv);
}

template <ggml_type type>
static void mul_mat_vec_q_conv_switch_ncols_dst(const void * vx, const void * vy, const uint32_t ncols_x, const uint32_t stride_row_x,
        const uint32_t stride_col_y, const uint32_t nrows, const int ncols_dst, const ggml_cuda_mmvq_conv & conv, const int device,
        cudaStream_t stream) {
    switch (ncols_dst) {
        case 2: mul_mat_vec_q_conv_launch<type, 2>(vx, vy, ncols_x, stride_row_x, stride_col_y, nrows, conv, device, stream); break;
        case 3: mul_mat_vec_q_conv_launch<type, 3>(vx, vy, ncols_x, stride_row_x, stride_col_y, nrows, conv, device, stream); break;
        case 4: mul_mat_vec_q_conv_launch<type, 4>(vx, vy, ncols_x, stride_row_x, stride_col_y, nrows, conv, device, stream); break;
        case 5: mul_mat_vec_q_conv_launch<type, 5>(vx, vy, ncols_x, stride_row_x, stride_col_y, nrows, conv, device, stream); break;
        case 6: mul_mat_vec_q_conv_launch<type, 6>(vx, vy, ncols_x, stride_row_x, stride_col_y, nrows, conv, device, stream); break;
        case 7: mul_mat_vec_q_conv_launch<type, 7>(vx, vy, ncols_x, stride_row_x, stride_col_y, nrows, conv, device, stream); break;
        case 8: mul_mat_vec_q_conv_launch<type, 8>(vx, vy, ncols_x, stride_row_x, stride_col_y, nrows, conv, device, stream); break;
        default: GGML_ABORT("fatal error");
    }
}

#define MMVQ_CONV_TYPES(X) \
    X(GGML_TYPE_Q4_0) X(GGML_TYPE_Q4_1) X(GGML_TYPE_Q5_0) X(GGML_TYPE_Q5_1) X(GGML_TYPE_Q8_0) X(GGML_TYPE_MXFP4) \
    X(GGML_TYPE_Q2_K) X(GGML_TYPE_Q3_K) X(GGML_TYPE_Q4_K) X(GGML_TYPE_Q5_K) X(GGML_TYPE_Q6_K) \
    X(GGML_TYPE_IQ2_XXS) X(GGML_TYPE_IQ2_XS) X(GGML_TYPE_IQ2_S) X(GGML_TYPE_IQ3_XXS) X(GGML_TYPE_IQ1_S) X(GGML_TYPE_IQ1_M) \
    X(GGML_TYPE_IQ4_NL) X(GGML_TYPE_IQ4_XS) X(GGML_TYPE_IQ3_S)

bool ggml_cuda_mmvq_conv_supported(const ggml_type type, const int cc, const int64_t ncols_dst) {
    if (ncols_dst < 2 || ncols_dst > MMVQ_MAX_BATCH_SIZE || !ggml_cuda_should_use_mmvq(type, cc, ncols_dst) ||
            ggml_cuda_mmvq_mma_supported(type, cc, ncols_dst)) {
        return false;
    }
    switch (type) {
#define MMVQ_CONV_CASE(t) case t: return true;
        MMVQ_CONV_TYPES(MMVQ_CONV_CASE)
#undef MMVQ_CONV_CASE
        default: return false;
    }
}

void ggml_cuda_mul_mat_vec_q_conv(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, ggml_cuda_norm_fold * fold,
        const ggml_cuda_mmvq_conv & conv) {
    const ggml_tensor * src0 = mm->src[0];
    const ggml_tensor * src1 = mm->src[1];
    const ggml_type     type = src0->type;
    cudaStream_t stream = ctx.stream();

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t  q8_bytes    = ne11*ne10_padded * sizeof(block_q8_1)/QK8_1;
    const int64_t s11_src = src1->nb[1] / sizeof(float);
    const int64_t s12_src = src1->nb[2] / sizeof(float);
    const int64_t s13_src = src1->nb[3] / sizeof(float);

    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
    char * q8 = nullptr;
    if (fold) {
        GGML_ASSERT(fold->x->type == GGML_TYPE_F32 && fold->w->type == GGML_TYPE_F32 && fold->q8_bytes == q8_bytes);
        ggml_cuda_norm_fold_reserve(ctx, *fold);
        if (!fold->q8_ready) {
            quantize_row_q8_1_rms_cuda((const float *) fold->x->data, (const float *) fold->w->data,
                ggml_cuda_tensor_is_constant(fold->w), fold->eps, fold->q8, fold->scale,
                ne10, s11_src, s12_src, s13_src, ne10_padded, ne11, 1, 1, ggml_cuda_l2_pf{}, stream);
            fold->q8_ready    = true;
            fold->scale_ready = true;
        }
        q8 = (char *) fold->q8;
    } else {
        q8 = src1_q8_1.alloc(q8_bytes);
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, q8, type, ne10, s11_src, s12_src, s13_src, ne10_padded, ne11, 1, 1, stream);
    }

    const uint32_t stride_row_x = src0->nb[1] / ggml_type_size(type);
    const uint32_t stride_col_y = ne10_padded / QK8_1;
    switch (type) {
#define MMVQ_CONV_CASE(t) case t: mul_mat_vec_q_conv_switch_ncols_dst<t>(src0->data, q8, ne10, stride_row_x, stride_col_y, \
            src0->ne[1], (int) ne11, conv, ctx.device, stream); break;
        MMVQ_CONV_TYPES(MMVQ_CONV_CASE)
#undef MMVQ_CONV_CASE
        default: GGML_ABORT("fatal error");
    }
}

static mmvq_member mmvq_member_of(const ggml_tensor * mm) {
    const ggml_tensor * src0 = mm->src[0];
    mmvq_member m = {};
    m.vx             = src0->data;
    m.dst            = (float *) mm->data;
    m.nrows          = src0->ne[1];
    m.stride_row_x   = src0->nb[1] / ggml_type_size(src0->type);
    m.stride_col_dst = mm->nb[1] / sizeof(float);
    return m;
}

bool ggml_cuda_mmvq_pair_ok(const ggml_tensor * a, const ggml_tensor * b, int device) {
    const ggml_type type  = a->src[0]->type;
    const int64_t   ncols = a->src[1]->ne[1];
    if (b->src[0]->type != type || b->src[1] != a->src[1] || ggml_cuda_mmvq_mma_supported(type, ggml_cuda_info().devices[device].cc, ncols)) {
        return false;
    }
    // NVFP4 is left to the fusion that applies its tensor scale
    switch (type) {
        case GGML_TYPE_Q1_0: case GGML_TYPE_Q2_0: case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0: case GGML_TYPE_MXFP4: case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K: case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ1_S: case GGML_TYPE_IQ1_M: case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ3_S:
            break;
        default:
            return false;
    }

    // both members must fit in one wave of the pair kernel: larger pairs measured slower than two launches
    static std::mutex mutex;
    static std::map<std::tuple<int, int, int64_t, int64_t>, int64_t> wave_rows_cache;
    const auto key = std::make_tuple(device, (int) type, a->src[1]->ne[0], ncols);
    int64_t wave_rows = 0;
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto it = wave_rows_cache.find(key);
        if (it != wave_rows_cache.end()) {
            wave_rows = it->second;
        } else {
            mmvq_member mb[2] = { mmvq_member_of(a), mmvq_member_of(b) };
            mul_mat_vec_q_pair_switch_type(type, nullptr, a->src[1]->ne[0], 0, mb, ncols, device, nullptr, &wave_rows);
            wave_rows_cache[key] = wave_rows;
        }
    }
    return a->src[0]->ne[1] <= wave_rows && b->src[0]->ne[1] <= wave_rows;
}

void ggml_cuda_mul_mat_vec_q_pair(ggml_backend_cuda_context & ctx, const ggml_tensor * a, const ggml_tensor * b, ggml_cuda_norm_fold * fold) {
    const ggml_tensor * src1 = a->src[1];
    const ggml_type     type = a->src[0]->type;
    GGML_ASSERT(b->src[1] == src1 && b->src[0]->type == type && a->src[0]->ne[0] == src1->ne[0] && b->src[0]->ne[0] == src1->ne[0]);
    cudaStream_t stream = ctx.stream();

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t  q8_bytes    = ne11*ne10_padded * sizeof(block_q8_1)/QK8_1;
    const int64_t s11_src = src1->nb[1] / sizeof(float);
    const int64_t s12_src = src1->nb[2] / sizeof(float);
    const int64_t s13_src = src1->nb[3] / sizeof(float);

    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
    char * q8 = nullptr;
    if (fold) {
        GGML_ASSERT(fold->x->type == GGML_TYPE_F32 && fold->w->type == GGML_TYPE_F32 && fold->q8_bytes == q8_bytes);
        ggml_cuda_norm_fold_reserve(ctx, *fold);
        if (!fold->q8_ready) {
            quantize_row_q8_1_rms_cuda((const float *) fold->x->data, (const float *) fold->w->data,
                ggml_cuda_tensor_is_constant(fold->w), fold->eps, fold->q8, fold->scale,
                ne10, s11_src, s12_src, s13_src, ne10_padded, ne11, 1, 1, ggml_cuda_l2_pf{}, stream);
            fold->q8_ready    = true;
            fold->scale_ready = true;
        }
        q8 = (char *) fold->q8;
    } else {
        q8 = src1_q8_1.alloc(q8_bytes);
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, q8, type, ne10, s11_src, s12_src, s13_src, ne10_padded, ne11, 1, 1, stream);
    }

    mmvq_member mb[2] = { mmvq_member_of(a), mmvq_member_of(b) };
    mul_mat_vec_q_pair_switch_type(type, q8, ne10, ne10_padded / QK8_1, mb, ne11, ctx.device, stream, nullptr);
}
