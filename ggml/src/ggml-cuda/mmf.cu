#include "ggml.h"
#include "mmf.cuh"
#include "mmid.cuh"

static __forceinline__ int mmf_get_rows_per_block(const int cc) {
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMF_ROWS_PER_BLOCK_CDNA;
    } else {
        return MMF_ROWS_PER_BLOCK;
    }
}

static __global__ void mmf_sum_k_slices(const float * __restrict__ partial, float * __restrict__ dst, const int nrows,
        const int ncols, const int nsplit, const int64_t stride_col_dst) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y;
    if (i >= nrows) {
        return;
    }
    float sum = 0.0f;
    for (int s = 0; s < nsplit; ++s) {
        sum += partial[((int64_t) s*ncols + j)*nrows + i];
    }
    dst[j*stride_col_dst + i] = sum;
}

template <typename T>
static void mmf_split_k_launch(ggml_backend_cuda_context & ctx, const T * x, const float * y, float * dst, const int64_t k,
        const int64_t nrows, const int64_t ncols, const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const int64_t nsplit, const int rows_per_block, const int vals_per_T) {
    const int64_t klen = k / nsplit;
    ggml_cuda_pool_alloc<float> partial(ctx.pool(), nsplit*ncols*nrows);

    // slice s of K is sample s: x and y step along K by one slice, each slice writes its own partial product
    mul_mat_f_switch_rows_per_block<T>(
        rows_per_block, x, y, nullptr, partial.get(), klen, nrows, ncols, stride_row, stride_col_y, nrows,
        0, 0, 1, 1, 1, 0, 0, 0,
        nsplit, nsplit, klen, klen*vals_per_T, ncols*nrows, ctx.stream(), nullptr);

    const dim3 block_nums((nrows + 255) / 256, ncols, 1);
    mmf_sum_k_slices<<<block_nums, 256, 0, ctx.stream()>>>(partial.get(), dst, (int) nrows, (int) ncols, (int) nsplit, stride_col_dst);
}

// A weight with few row blocks leaves most SMs idle while each block walks all of K (a router, a gate projection):
// K is split into slices computed as samples of the same kernel, summed in a fixed order afterwards (deterministic,
// unlike atomics). Splits only while the row blocks fill under a quarter of the SMs, up to about one block per SM.
static bool mmf_split_k(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const int rows_per_block) {
    const int device    = ggml_cuda_get_device();
    const int nsm       = ggml_cuda_info().devices[device].nsm;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;

    const int vals_per_T = src0->type == GGML_TYPE_F32 ? 1 : 2;
    const int64_t k            = src0->ne[0] / vals_per_T;
    const int64_t nrows        = src0->ne[1];
    const int64_t n_row_blocks = nrows / rows_per_block;
    if (n_row_blocks * 4 > nsm) {
        return false;
    }
    int64_t nsplit = 1;
    while (n_row_blocks * nsplit * 2 <= nsm && k % (nsplit * 2 * warp_size) == 0) {
        nsplit *= 2;
    }
    if (nsplit == 1) {
        return false;
    }

    const int64_t s01 = src0->nb[1] / ggml_type_size(src0->type);
    const int64_t s11 = src1->nb[1] / ggml_type_size(src1->type);
    const int64_t s1  =  dst->nb[1] / ggml_type_size(dst->type);
    const float * y = (const float *) src1->data;
    float       * d = (float *) dst->data;

    switch (src0->type) {
        case GGML_TYPE_F32:
            mmf_split_k_launch<float>(ctx, (const float *) src0->data, y, d, k, nrows, src1->ne[1], s01, s11, s1, nsplit, rows_per_block, 1);
            return true;
        case GGML_TYPE_F16:
            mmf_split_k_launch<half2>(ctx, (const half2 *) src0->data, y, d, k, nrows, src1->ne[1], s01/2, s11/2, s1, nsplit, rows_per_block, 2);
            return true;
        case GGML_TYPE_BF16:
            mmf_split_k_launch<nv_bfloat162>(ctx, (const nv_bfloat162 *) src0->data, y, d, k, nrows, src1->ne[1], s01/2, s11/2, s1, nsplit, rows_per_block, 2);
            return true;
        default:
            return false;
    }
}

void ggml_cuda_mul_mat_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(!ids ||  ids->type == GGML_TYPE_I32);
    GGML_ASSERT(         dst->type == GGML_TYPE_F32);


    GGML_TENSOR_BINARY_OP_LOCALS;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(ne13 == ne3);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));
    GGML_ASSERT(        nb0        == ts_dst);

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = src1->nb[1] / ts_src1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s12 = src1->nb[2] / ts_src1;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s13 = src1->nb[3] / ts_src1;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t ids_s0 = ids ? ids->nb[0] / ggml_type_size(ids->type) : 0;
    const int64_t ids_s1 = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    mmf_ids_data ids_info{};
    mmf_ids_data * ids_info_ptr = nullptr;
    ggml_cuda_pool_alloc<int32_t> ids_src_compact_dev;
    ggml_cuda_pool_alloc<int32_t> ids_dst_compact_dev;
    ggml_cuda_pool_alloc<int32_t> expert_bounds_dev;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_dst      = ids ? ne1 : ne2;

    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1 : s2;

    int64_t stride_channel_y         = ids ? s11  : s12;
    int64_t nchannels_y              = ids ? ne11 : ne12;

    //mul_mat_id: handle broadcast
    if (ids && nchannels_y == 1) {
        stride_channel_y = 0;
        nchannels_y      = ids->ne[0];
    }

    if (ids && ncols_dst > 16) {
        const int64_t n_expert_used = ids->ne[0];
        const int64_t n_experts     = ne02;
        const int64_t n_tokens      = ne12;
        const int64_t ne_get_rows   = n_tokens * n_expert_used;

        ids_src_compact_dev.alloc(ctx.pool(), ne_get_rows);
        ids_dst_compact_dev.alloc(ctx.pool(), ne_get_rows);
        expert_bounds_dev.alloc(ctx.pool(), n_experts + 1);

        const int si1  = static_cast<int>(ids_s1);
        const int sis1 = static_cast<int>(src1->nb[2] / src1->nb[1]);

        GGML_ASSERT(sis1 > 0);

        ggml_cuda_launch_mm_ids_helper(ids_d, ids_src_compact_dev.get(), ids_dst_compact_dev.get(), expert_bounds_dev.get(),
            static_cast<int>(n_experts), static_cast<int>(n_tokens), static_cast<int>(n_expert_used), static_cast<int>(ne11), si1, sis1, /*write_inverse =*/ false, ctx.stream());
        CUDA_CHECK(cudaGetLastError());

        ids_info.ids_src_compact   = ids_src_compact_dev.get();
        ids_info.ids_dst_compact   = ids_dst_compact_dev.get();
        ids_info.expert_bounds_dev = expert_bounds_dev.get();
        ids_info.n_experts         = static_cast<int>(n_experts);
        ids_info.sis1              = sis1;
        ids_info_ptr = &ids_info;
    }

    const int device    = ggml_cuda_get_device();
    const int cc        = ggml_cuda_info().devices[device].cc;
    const int rows_per_block = mmf_get_rows_per_block(cc);

    if (!ids && ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1 && mmf_split_k(ctx, src0, src1, dst, rows_per_block)) {
        return;
    }

    switch (src0->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *) src0->data;
            constexpr int vals_per_T = 1;
            mul_mat_f_switch_rows_per_block<float>(
                rows_per_block, src0_d, src1_d, ids_d, dst_d, ne00/vals_per_T, ne01, ncols_dst, s01/vals_per_T, stride_col_y/vals_per_T, stride_col_dst,
                ids_s0, ids_s1, ne02, nchannels_y, nchannels_dst, s02/vals_per_T, stride_channel_y, stride_channel_dst,
                ne03, ne3, s03/vals_per_T, s13, s3, ctx.stream(), ids_info_ptr);
        } break;
        case GGML_TYPE_F16: {
            const half2 * src0_d = (const half2 *) src0->data;
            constexpr int vals_per_T = 2;
            mul_mat_f_switch_rows_per_block<half2>(
                rows_per_block, src0_d, src1_d, ids_d, dst_d, ne00/vals_per_T, ne01, ncols_dst, s01/vals_per_T, stride_col_y/vals_per_T, stride_col_dst,
                ids_s0, ids_s1, ne02, nchannels_y, nchannels_dst, s02/vals_per_T, stride_channel_y, stride_channel_dst,
                ne03, ne3, s03/vals_per_T, s13, s3, ctx.stream(), ids_info_ptr);
        } break;
        case GGML_TYPE_BF16: {
            const nv_bfloat162 * src0_d = (const nv_bfloat162 *) src0->data;
            constexpr int vals_per_T = 2;
            mul_mat_f_switch_rows_per_block<nv_bfloat162>(
                rows_per_block, src0_d, src1_d, ids_d, dst_d, ne00/vals_per_T, ne01, ncols_dst, s01/vals_per_T, stride_col_y/vals_per_T, stride_col_dst,
                ids_s0, ids_s1, ne02, nchannels_y, nchannels_dst, s02/vals_per_T, stride_channel_y, stride_channel_dst,
                ne03, ne3, s03/vals_per_T, s13, s3, ctx.stream(), ids_info_ptr);
        } break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }
}

bool ggml_cuda_should_use_mmf(enum ggml_type type, int cc, int warp_size, const int64_t * src0_ne,
        const size_t * src0_nb, const int src1_ncols, bool mul_mat_id) {
    if (ggml_is_quantized(type) || type == GGML_TYPE_F8_E4M3) {
        return false;
    }

    const size_t ts = ggml_type_size(type);
    if (src0_ne[0] % (warp_size * (4/ts)) != 0) {
        return false;
    }

    if (src0_nb[0] != ts) {
        return false;
    }

    // Pointers not aligned to the size of half2/nv_bfloat162/float2 would result in a crash:
    for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
        if (src0_nb[i] % (2*ts) != 0) {
            return false;
        }
    }
    if (src0_ne[1] % mmf_get_rows_per_block(cc) != 0) {
        return false;
    }

    if (GGML_CUDA_CC_IS_CDNA3(cc) && type == GGML_TYPE_BF16) {
        return false;
    }

    if (mul_mat_id) {
        if (src0_ne[1] <= 1024 && src1_ncols > 512) {
            return false;
        } else if(src0_ne[1] > 1024 && src1_ncols > 128) {
            return false;
        }
    } else {
        if (GGML_CUDA_CC_IS_RDNA3_0(cc) && src1_ncols > 8) {
            return false;
        } else if (GGML_CUDA_CC_IS_CDNA2(cc) && (type == GGML_TYPE_F16 || type == GGML_TYPE_BF16)) {
            //TODO: truse CDNA2 as CDNA1, tune the perf when CDNA2 is available.
            return false;
        } else if (GGML_CUDA_CC_IS_CDNA1(cc) && (type == GGML_TYPE_F16 || type == GGML_TYPE_BF16)) {
            return false;
        } else if (src1_ncols > 16) {
            return false;
        }
    }

    switch (type) {
        case GGML_TYPE_F32:
            return ampere_mma_available(cc) || amd_mfma_available(cc);
        case GGML_TYPE_F16:
            return volta_mma_available(cc) || turing_mma_available(cc) || amd_wmma_available(cc) || amd_mfma_available(cc);
        case GGML_TYPE_BF16:
            return ampere_mma_available(cc) || amd_wmma_available(cc) || amd_mfma_available(cc);
        default:
            return false;
    }
}
