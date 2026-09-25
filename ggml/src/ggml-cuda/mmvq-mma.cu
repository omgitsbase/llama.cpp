#include "mmvq-mma.cuh"
#include "mma.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

using namespace ggml_cuda_mma;

// Each warp holds a 16-row tile; the eight warps of a block take turns over the rows' superblocks and add up
// at the end. Within a 32-weight chunk a lane feeds the mma k-groups 2t and 2t+1 of its two rows (t = lane%4),
// and the same two groups of its column's activations: which k positions the hardware pairs is free as long
// as both sides agree, and two adjacent groups keep a lane's loads to one index word and one sign byte.
// The integer sum of a chunk is exactly the vector kernel's, and so is the integer scale applied to it.
static constexpr int mmvq_mma_nwarps = 8;

template <ggml_type type> struct mmvq_mma_traits;

template <> struct mmvq_mma_traits<GGML_TYPE_IQ3_S> {
    typedef block_iq3_s block;
    static __device__ __forceinline__ void decode(const block * b, const int c, const int t, int & lo, int & hi, int & isc) {
        const int qs16 = ((const uint16_t *) b->qs)[4*c + t];
        const int qh   = b->qh[c];
        const int i0   = (qs16 & 0xFF) | (((qh >> (2*t + 0)) & 1) << 8);
        const int i1   = (qs16 >>   8) | (((qh >> (2*t + 1)) & 1) << 8);
        const int s    = b->signs[4*c + t];
        const int sg0  = __vcmpne4(((s & 0x03) << 7) | ((s & 0x0C) << 21), 0x00000000);
        const int sg1  = __vcmpne4(((s & 0x30) << 3) | ((s & 0xC0) << 17), 0x00000000);
        lo  = __vsub4(iq3s_grid[i0] ^ sg0, sg0);
        hi  = __vsub4(iq3s_grid[i1] ^ sg1, sg1);
        isc = 1 + 2*((b->scales[c/2] >> ((c & 1)*4)) & 0x0F);
    }
    static __device__ __forceinline__ int scale_sum(const int sumi, const int isc) {
        return sumi*isc;
    }
};

template <> struct mmvq_mma_traits<GGML_TYPE_IQ3_XXS> {
    typedef block_iq3_xxs block;
    static __device__ __forceinline__ void decode(const block * b, const int c, const int t, int & lo, int & hi, int & isc) {
        const int      q16   = ((const uint16_t *) b->qs)[4*c + t];
        const uint32_t aux32 = get_int_b2(b->qs, QK_K/16 + c);
        const uint32_t signs = unpack_ksigns(aux32 >> (7*t));
        const int sg0 = __vcmpne4(signs & 0x08040201, 0);
        const int sg1 = __vcmpne4(signs & 0x80402010, 0);
        lo  = __vsub4(iq3xxs_grid[q16 & 0xFF] ^ sg0, sg0);
        hi  = __vsub4(iq3xxs_grid[q16 >>   8] ^ sg1, sg1);
        isc = aux32 >> 28;
    }
    static __device__ __forceinline__ int scale_sum(const int sumi, const int isc) {
        return (isc*sumi + sumi/2)/2;
    }
};

template <ggml_type type, int ncols_dst, bool has_fusion>
__launch_bounds__(mmvq_mma_nwarps*WARP_SIZE, 1)
static __global__ void mul_mat_vec_q_mma(
        const void * __restrict__ vx, const block_q8_1 * __restrict__ y, const ggml_cuda_mm_fusion_args_device fusion,
        float * __restrict__ dst, const int nblocks_row, const int nrows_x, const int stride_row_x, const int stride_col_y,
        const int stride_col_dst) {
#if defined(TURING_MMA_AVAILABLE)
    typedef mmvq_mma_traits<type>   traits;
    typedef typename traits::block block;
    constexpr int blocks_q8 = QK_K/QK8_1;

    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int g    = lane / 4;
    const int t    = lane % 4;
    // rows past the end read the last row and are not written
    const int row0 = 16*blockIdx.x;

    [[maybe_unused]] bool use_gate = false;
    if constexpr (has_fusion) {
        use_gate = fusion.gate != nullptr;
    }

    // the activations this lane feeds the mma are column g's; its results are columns 2t and 2t+1
    const bool feeds = g < ncols_dst;
    const int  jb    = feeds ? g : 0;
    const int  j0    = 2*t;
    const int  j1    = 2*t + 1;
    const block_q8_1 * yb  = y + (int64_t) jb*stride_col_y;
    const block_q8_1 * yd0 = y + (int64_t) (j0 < ncols_dst ? j0 : 0)*stride_col_y;
    const block_q8_1 * yd1 = y + (int64_t) (j1 < ncols_dst ? j1 : 0)*stride_col_y;

    float acc[4]  = {0.0f};
    float accg[4] = {0.0f};

    ggml_cuda_pdl_sync();
    for (int kb = warp; kb < nblocks_row; kb += mmvq_mma_nwarps) {
        const block * xs0 = (const block *) vx + (int64_t) min(row0 + g,     nrows_x - 1)*stride_row_x + kb;
        const block * xs1 = (const block *) vx + (int64_t) min(row0 + g + 8, nrows_x - 1)*stride_row_x + kb;
        [[maybe_unused]] const block * gs0 = nullptr;
        [[maybe_unused]] const block * gs1 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gs0 = (const block *) fusion.gate + (int64_t) min(row0 + g,     nrows_x - 1)*stride_row_x + kb;
                gs1 = (const block *) fusion.gate + (int64_t) min(row0 + g + 8, nrows_x - 1)*stride_row_x + kb;
            }
        }
        const float dx0 = __half2float(xs0->d);
        const float dx1 = __half2float(xs1->d);
        [[maybe_unused]] float dg0 = 0.0f;
        [[maybe_unused]] float dg1 = 0.0f;
        if constexpr (has_fusion) {
            if (use_gate) {
                dg0 = __half2float(gs0->d);
                dg1 = __half2float(gs1->d);
            }
        }
#pragma unroll
        for (int c = 0; c < QK_K/32; ++c) {
            const int kby = kb*blocks_q8 + c;

            tile<8, 8, int> B;
            B.x[0] = feeds ? get_int_b4(yb[kby].qs, 2*t + 0) : 0;
            B.x[1] = feeds ? get_int_b4(yb[kby].qs, 2*t + 1) : 0;
            const float dy0 = j0 < ncols_dst ? __low2float(yd0[kby].ds) : 0.0f;
            const float dy1 = j1 < ncols_dst ? __low2float(yd1[kby].ds) : 0.0f;

            tile<16, 8, int> A;
            tile<16, 8, int> C;
            int isc0;
            int isc1;
            traits::decode(xs0, c, t, A.x[0], A.x[2], isc0);
            traits::decode(xs1, c, t, A.x[1], A.x[3], isc1);
            mma(C, A, B);
            acc[0] += (dx0*dy0) * (float) traits::scale_sum(C.x[0], isc0);
            acc[1] += (dx0*dy1) * (float) traits::scale_sum(C.x[1], isc0);
            acc[2] += (dx1*dy0) * (float) traits::scale_sum(C.x[2], isc1);
            acc[3] += (dx1*dy1) * (float) traits::scale_sum(C.x[3], isc1);

            if constexpr (has_fusion) {
                if (use_gate) {
                    tile<16, 8, int> G;
                    traits::decode(gs0, c, t, A.x[0], A.x[2], isc0);
                    traits::decode(gs1, c, t, A.x[1], A.x[3], isc1);
                    mma(G, A, B);
                    accg[0] += (dg0*dy0) * (float) traits::scale_sum(G.x[0], isc0);
                    accg[1] += (dg0*dy1) * (float) traits::scale_sum(G.x[1], isc0);
                    accg[2] += (dg1*dy0) * (float) traits::scale_sum(G.x[2], isc1);
                    accg[3] += (dg1*dy1) * (float) traits::scale_sum(G.x[3], isc1);
                }
            }
        }
    }
    ggml_cuda_pdl_lc();

    __shared__ float red[mmvq_mma_nwarps - 1][4][WARP_SIZE];
    [[maybe_unused]] __shared__ float red_gate[has_fusion ? mmvq_mma_nwarps - 1 : 1][4][WARP_SIZE];
    if (warp > 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            red[warp - 1][i][lane] = acc[i];
            if constexpr (has_fusion) {
                red_gate[warp - 1][i][lane] = accg[i];
            }
        }
    }
    __syncthreads();
    if (warp > 0) {
        return;
    }
#pragma unroll
    for (int w = 0; w < mmvq_mma_nwarps - 1; ++w) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc[i] += red[w][i][lane];
            if constexpr (has_fusion) {
                accg[i] += red_gate[w][i][lane];
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = row0 + g + 8*(i/2);
        const int col = (i % 2) ? j1 : j0;
        if (row >= nrows_x || col >= ncols_dst) {
            continue;
        }
        float result = acc[i];
        if constexpr (has_fusion) {
            if (fusion.x_bias) {
                result += ((const float *) fusion.x_bias)[col*stride_col_dst + row];
            }
            if (use_gate) {
                float gate_value = accg[i];
                if (fusion.gate_bias) {
                    gate_value += ((const float *) fusion.gate_bias)[col*stride_col_dst + row];
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
                    default:
                        result = result * gate_value;
                        break;
                }
            }
        }
        dst[col*stride_col_dst + row] = result;
    }
#else
    GGML_UNUSED_VARS(vx, y, fusion, dst, nblocks_row, nrows_x, stride_row_x, stride_col_y, stride_col_dst);
    NO_DEVICE_CODE;
#endif // defined(TURING_MMA_AVAILABLE)
}

template <ggml_type type, int ncols_dst>
static void mmvq_mma_launch(
        const void * vx, const void * vy, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        cudaStream_t stream) {
    const dim3 block_nums((nrows_x + 15)/16, 1, 1);
    const dim3 block_dims(WARP_SIZE, mmvq_mma_nwarps, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
    const int nblocks_row = ncols_x / QK_K;
    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    if (has_fusion) {
        ggml_cuda_kernel_launch(mul_mat_vec_q_mma<type, ncols_dst, true>, launch_params,
            vx, (const block_q8_1 *) vy, fusion, dst, nblocks_row, nrows_x, stride_row_x, stride_col_y, stride_col_dst);
    } else {
        ggml_cuda_kernel_launch(mul_mat_vec_q_mma<type, ncols_dst, false>, launch_params,
            vx, (const block_q8_1 *) vy, fusion, dst, nblocks_row, nrows_x, stride_row_x, stride_col_y, stride_col_dst);
    }
}

bool ggml_cuda_mmvq_mma_supported(ggml_type type, int cc, int64_t ncols_dst) {
    static const bool off = getenv("GGML_CUDA_NO_MMVQ_MMA") != nullptr;
    // measured end to end on Qwen3.8-27B: at 4-6 columns the vector kernel, which decodes a block once for all
    // columns, is as fast or faster; at 8 (seven drafts) this path is 4-10% faster
    if (off || ncols_dst != 8 || !turing_mma_available(cc)) {
        return false;
    }
    switch (type) {
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ3_XXS:
            return true;
        default:
            return false;
    }
}

void ggml_cuda_mmvq_mma(
        const void * vx, ggml_type type, const void * vy, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        int ncols_x, int nrows_x, int ncols_dst, int stride_row_x, int stride_col_y, int stride_col_dst, cudaStream_t stream) {
    GGML_ASSERT(ncols_x % QK_K == 0 && ncols_dst == 8);
    switch (type) {
        case GGML_TYPE_IQ3_S:
            mmvq_mma_launch<GGML_TYPE_IQ3_S, 8>(vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mmvq_mma_launch<GGML_TYPE_IQ3_XXS, 8>(vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}
