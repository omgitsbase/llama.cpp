#include <algorithm>
#include <cstdint>

#include "argmax.cuh"
#include "common.cuh"
#include "sum.cuh"
#include "unary.cuh"

static __global__ void argmax_f32(const float * __restrict__ x, int32_t * __restrict__ dst, const int64_t ncols) {
    const int64_t row = blockIdx.x;

    float maxval = -FLT_MAX;
    int   argmax = -1;
    const float * rowx = x + row * ncols;

    for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        const float val = rowx[col];
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
        const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (n_warps > 1) {
        constexpr int    max_warps = 1024 / WARP_SIZE;
        __shared__ float shared_maxval[max_warps];
        __shared__ int   shared_argmax[max_warps];
        if (lane_id == 0) {
            shared_maxval[warp_id] = maxval;
            shared_argmax[warp_id] = argmax;
        }

        __syncthreads();

        if (warp_id == 0) {
            if (lane_id < n_warps) {
                maxval = shared_maxval[lane_id];
                argmax = shared_argmax[lane_id];
            }
#pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
                const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
                if (val > maxval) {
                    maxval = val;
                    argmax = col;
                }
            }
        }
    }

    if (warp_id == 0 && lane_id == 0) {
        dst[row] = argmax;
    }
}

// A vocabulary-sized row read by one block is latency bound: each thread
// walks hundreds of elements. Blocks take a slice each and one warp settles
// the slices, the lowest index winning a tie.
template <bool softcap>
static __global__ void argmax_f32_partial(const float * __restrict__ x, float * __restrict__ part_val, int32_t * __restrict__ part_idx,
                                          const int64_t ncols, const int64_t chunk, const ggml_cuda_softcap_params sc) {
    const int64_t row   = blockIdx.y;
    const int64_t start = (int64_t) blockIdx.x * chunk;
    const int64_t end   = min(start + chunk, ncols);

    const float * rowx = x + row * ncols;

    float maxval = -FLT_MAX;
    int   argmax = -1;

    for (int64_t col = start + threadIdx.x; col < end; col += blockDim.x) {
        float val = rowx[col];
        if constexpr (softcap) {
            val = sc.b0 * op_tanh(sc.a0 * val + sc.a1) + sc.b1;
        }
        if (val > maxval || (val == maxval && (int) col < argmax)) {
            maxval = val;
            argmax = (int) col;
        }
    }

#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
        const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
        if (val > maxval || (val == maxval && col < argmax && col >= 0)) {
            maxval = val;
            argmax = col;
        }
    }

    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (n_warps > 1) {
        constexpr int    max_warps = 1024 / WARP_SIZE;
        __shared__ float shared_maxval[max_warps];
        __shared__ int   shared_argmax[max_warps];
        if (lane_id == 0) {
            shared_maxval[warp_id] = maxval;
            shared_argmax[warp_id] = argmax;
        }
        __syncthreads();
        if (warp_id == 0) {
            maxval = lane_id < n_warps ? shared_maxval[lane_id] : -FLT_MAX;
            argmax = lane_id < n_warps ? shared_argmax[lane_id] : -1;
#pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
                const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
                if (val > maxval || (val == maxval && col < argmax && col >= 0)) {
                    maxval = val;
                    argmax = col;
                }
            }
        }
    }

    if (warp_id == 0 && lane_id == 0) {
        part_val[row * gridDim.x + blockIdx.x] = maxval;
        part_idx[row * gridDim.x + blockIdx.x] = argmax;
    }
}

static __global__ void argmax_f32_final(const float * __restrict__ part_val, const int32_t * __restrict__ part_idx,
                                        int32_t * __restrict__ dst, const int nparts, float * __restrict__ val_out) {
    const int row = blockIdx.x;

    float maxval = -FLT_MAX;
    int   argmax = -1;

    for (int i = threadIdx.x; i < nparts; i += WARP_SIZE) {
        const float val = part_val[row * nparts + i];
        const int   col = part_idx[row * nparts + i];
        if (val > maxval || (val == maxval && col < argmax && col >= 0)) {
            maxval = val;
            argmax = col;
        }
    }

#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
        const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
        if (val > maxval || (val == maxval && col < argmax && col >= 0)) {
            maxval = val;
            argmax = col;
        }
    }

    if (threadIdx.x == 0) {
        dst[row] = argmax;
        if (val_out) {
            val_out[row] = maxval;
        }
    }
}

void ggml_cuda_argmax_softcap(ggml_backend_cuda_context & ctx, const ggml_tensor * x, int32_t * ids, float * val_out,
                              const ggml_cuda_softcap_params & sc) {
    GGML_ASSERT(x->type == GGML_TYPE_F32 && ggml_is_contiguous(x) && x->ne[0] >= 16384);

    const int64_t ne00  = x->ne[0];
    const int64_t nrows = ggml_nrows(x);

    const int64_t chunk  = 4096;
    const int     nparts = (int) std::min<int64_t>(64, (ne00 + chunk - 1) / chunk);
    const int64_t per    = (ne00 + nparts - 1) / nparts;

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<float>   part_val(ctx.pool(), nrows * nparts);
    ggml_cuda_pool_alloc<int32_t> part_idx(ctx.pool(), nrows * nparts);

    const dim3 grid(nparts, nrows, 1);
    argmax_f32_partial<true><<<grid, 1024, 0, stream>>>((const float *) x->data, part_val.get(), part_idx.get(), ne00, per, sc);
    argmax_f32_final<<<nrows, WARP_SIZE, 0, stream>>>(part_val.get(), part_idx.get(), ids, nparts, val_out);
}

void ggml_cuda_argmax(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    int32_t     * dst_d  = (int32_t     *) dst->data;

    cudaStream_t stream = ctx.stream();

    if (ne00 >= 16384) {
        const int64_t chunk  = 4096;
        const int     nparts = (int) std::min<int64_t>(64, (ne00 + chunk - 1) / chunk);
        const int64_t per    = (ne00 + nparts - 1) / nparts;

        ggml_cuda_pool_alloc<float>   part_val(ctx.pool(), nrows * nparts);
        ggml_cuda_pool_alloc<int32_t> part_idx(ctx.pool(), nrows * nparts);

        const dim3 grid(nparts, nrows, 1);
        argmax_f32_partial<false><<<grid, 1024, 0, stream>>>(src0_d, part_val.get(), part_idx.get(), ne00, per, {});
        argmax_f32_final<<<nrows, WARP_SIZE, 0, stream>>>(part_val.get(), part_idx.get(), dst_d, nparts, nullptr);
        return;
    }

    const int64_t num_blocks = nrows;
    const int64_t num_threads = std::min<int64_t>(1024, (ne00 + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    const dim3 blocks_dim(num_threads, 1, 1);
    const dim3 blocks_num(num_blocks, 1, 1);

    argmax_f32<<<blocks_num, blocks_dim, 0, stream>>>(src0_d, dst_d, ne00);
}
