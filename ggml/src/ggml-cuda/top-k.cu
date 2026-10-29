#include "argsort.cuh"
#include "top-k.cuh"
#include "unary.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

// the radix select: every row at once (a sampler's rows of one graph), where CUB's top-k takes a row a call

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
    int equal_count;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0, 0};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

static __global__ void top_k_radix_reset_counters(top_k_radix_state * states, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row].greater_count = 0;
        states[row].equal_count = 0;
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        top_k_radix_state * __restrict__ states,
        int ncols,
        int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_radix_state * state = &states[row];

    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > state->prefix) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if (key == state->prefix) {
            const int pos = atomicAdd(&state->equal_count, 1);
            if (pos < state->rank) {
                row_dst[k - state->rank + pos] = col;
            }
        }
    }
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = std::min((ncols + 1023) / 1024, 64);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    top_k_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);

    const dim3 row_grid(blocks_per_row * nrows);
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift);
        top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
    }

    top_k_radix_reset_counters
        <<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows);
    top_k_radix_gather<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, states, ncols, k, blocks_per_row);
}


// the two-stage select for a small k (a sampler's candidates): a block a chunk of a row keeps the chunk's keys in
// registers and finds the chunk's top k by radix passes in shared memory, then a block a row selects the row's top k
// from the chunks'. The logits are read once, and the input alone fixes the order of the output (positions come from
// scans, not atomics)

template<int BLOCK>
static __device__ __forceinline__ int top_k_block_scan_incl(int v, int * s_warp, int & total) {
    // inclusive prefix sum of v in thread order, and the sum over the block
    constexpr int nwarps = BLOCK / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
#pragma unroll
    for (int off = 1; off < WARP_SIZE; off <<= 1) {
        const int t = __shfl_up_sync(0xffffffff, v, off);
        if (lane >= off) {
            v += t;
        }
    }
    if (lane == WARP_SIZE - 1) {
        s_warp[warp] = v;
    }
    __syncthreads();
    int add = 0;
    total = 0;
#pragma unroll
    for (int w = 0; w < nwarps; ++w) {
        add   += w < warp ? s_warp[w] : 0;
        total += s_warp[w];
    }
    __syncthreads();
    return v + add;
}

// the k-th largest of the block's keys (thr) and how many keys equal to it belong to the top k (n_eq)
template<int BLOCK, int ITEMS>
static __device__ __forceinline__ void top_k_block_select(const uint32_t (&key)[ITEMS], const int k, uint32_t & thr, int & n_eq) {
    constexpr int NBINS = 256;
    static_assert(BLOCK >= NBINS, "a thread a bin");
    __shared__ int hist[NBINS];
    __shared__ int s_warp[BLOCK / WARP_SIZE];
    __shared__ uint32_t s_prefix;
    __shared__ int s_rank;

    const int tid = threadIdx.x;
    uint32_t prefix = 0, pmask = 0;
    int rank = k;
#pragma unroll
    for (int shift = 24; shift >= 0; shift -= 8) {
        if (tid < NBINS) {
            hist[tid] = 0;
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            const bool     in    = (key[i] & pmask) == prefix;
            const int      bin   = in ? int((key[i] >> shift) & (NBINS - 1)) : -1;
            const unsigned peers = __match_any_sync(0xffffffff, bin);
            if (in && int(threadIdx.x % WARP_SIZE) == __ffs(peers) - 1) {
                atomicAdd(&hist[bin], __popc(peers));
            }
        }
        __syncthreads();
        // thread t holds bin NBINS-1-t: its inclusive sum counts the keys at or above its bin
        const int cnt = tid < NBINS ? hist[NBINS - 1 - tid] : 0;
        int total;
        const int above = top_k_block_scan_incl<BLOCK>(cnt, s_warp, total);
        if (tid < NBINS && above >= rank && above - cnt < rank) {
            s_prefix = prefix | ((uint32_t) (NBINS - 1 - tid) << shift);
            s_rank   = rank - (above - cnt);
        }
        __syncthreads();
        prefix = s_prefix;
        rank   = s_rank;
        pmask |= (uint32_t) (NBINS - 1) << shift;
    }
    thr  = prefix;
    n_eq = rank;
}

// the block's top k in a fixed order: the keys above the threshold, then the first n_eq equal to it
template<int BLOCK, int ITEMS>
static __device__ __forceinline__ void top_k_block_write(
        const uint32_t (&key)[ITEMS], const int (&idx)[ITEMS], const uint32_t thr, const int n_eq,
        uint32_t * out_key, int * out_idx) {
    __shared__ int s_warp[BLOCK / WARP_SIZE];
    int n_gt_t = 0, n_eq_t = 0;
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        n_gt_t += key[i] >  thr;
        n_eq_t += key[i] == thr;
    }
    int n_gt;
    int gt_pos = top_k_block_scan_incl<BLOCK>(n_gt_t, s_warp, n_gt) - n_gt_t;
    int n_eq_all;
    int eq_pos = top_k_block_scan_incl<BLOCK>(n_eq_t, s_warp, n_eq_all) - n_eq_t;
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        if (key[i] > thr) {
            if (out_key) {
                out_key[gt_pos] = key[i];
            }
            out_idx[gt_pos++] = idx[i];
        } else if (key[i] == thr) {
            if (eq_pos < n_eq) {
                if (out_key) {
                    out_key[n_gt + eq_pos] = key[i];
                }
                out_idx[n_gt + eq_pos] = idx[i];
            }
            eq_pos++;
        }
    }
}

template<int BLOCK, int ITEMS, bool softcap>
__launch_bounds__(BLOCK)
static __global__ void top_k_chunk_select(
        const float * __restrict__ src, uint32_t * __restrict__ cand_key, int * __restrict__ cand_idx, const int ncols, const int k,
        const ggml_cuda_softcap_params sc) {
    const int row   = blockIdx.y;
    const int chunk = blockIdx.x;
    const float * row_src = src + (size_t) row*ncols;
    const int c0 = chunk*BLOCK*ITEMS;

    uint32_t key[ITEMS];
    int      idx[ITEMS];
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        const int col = c0 + i*BLOCK + threadIdx.x;
        float v = col < ncols ? row_src[col] : 0.0f;
        if constexpr (softcap) {
            // the logits' cap as the argmax applies it (argmax_f32_partial): the keys of the capped logits
            v = sc.b0 * op_tanh(sc.a0 * v + sc.a1) + sc.b1;
        }
        // past the row: below every logit (-inf maps above 0)
        key[i] = col < ncols ? top_k_float_to_ordered(v) : 0u;
        idx[i] = col < ncols ? col : -1;
    }
    uint32_t thr;
    int n_eq;
    top_k_block_select<BLOCK, ITEMS>(key, k, thr, n_eq);
    const size_t off = ((size_t) row*gridDim.x + chunk)*k;
    top_k_block_write<BLOCK, ITEMS>(key, idx, thr, n_eq, cand_key + off, cand_idx + off);
}

template<int BLOCK, int ITEMS>
__launch_bounds__(BLOCK)
static __global__ void top_k_row_select(
        const uint32_t * __restrict__ cand_key, const int * __restrict__ cand_idx, int * __restrict__ dst, const int ncand, const int k) {
    const int row = blockIdx.x;
    const uint32_t * rk = cand_key + (size_t) row*ncand;
    const int      * ri = cand_idx + (size_t) row*ncand;

    uint32_t key[ITEMS];
    int      idx[ITEMS];
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        const int c = i*BLOCK + threadIdx.x;
        key[i] = c < ncand ? rk[c] : 0u;
        idx[i] = c < ncand ? ri[c] : -1;
    }
    uint32_t thr;
    int n_eq;
    top_k_block_select<BLOCK, ITEMS>(key, k, thr, n_eq);
    top_k_block_write<BLOCK, ITEMS>(key, idx, thr, n_eq, nullptr, dst + (size_t) row*k);
}

static __global__ void top_k_gather_softcap(
        const float * __restrict__ x, const int * __restrict__ ids, float * __restrict__ vals, const int ncols, const int k, const int n,
        const ggml_cuda_softcap_params sc) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float v = x[(size_t) (i / k)*ncols + ids[i]];
    vals[i] = sc.b0 * op_tanh(sc.a0 * v + sc.a1) + sc.b1;
}

static void top_k_two_stage_cuda(
        ggml_cuda_pool & pool, const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream,
        const ggml_cuda_softcap_params * sc = nullptr) {
    constexpr int BLOCK1 = 256, ITEMS1 = 16; // a chunk of 4096 logits
    constexpr int BLOCK2 = 512;
    const int nchunks = (ncols + BLOCK1*ITEMS1 - 1) / (BLOCK1*ITEMS1);
    const int ncand   = nchunks*k;

    ggml_cuda_pool_alloc<uint32_t> keys_alloc(pool, (size_t) nrows*ncand);
    ggml_cuda_pool_alloc<int>      idxs_alloc(pool, (size_t) nrows*ncand);

    if (sc) {
        top_k_chunk_select<BLOCK1, ITEMS1, true><<<dim3(nchunks, nrows), BLOCK1, 0, stream>>>(src, keys_alloc.get(), idxs_alloc.get(), ncols, k, *sc);
    } else {
        top_k_chunk_select<BLOCK1, ITEMS1, false><<<dim3(nchunks, nrows), BLOCK1, 0, stream>>>(src, keys_alloc.get(), idxs_alloc.get(), ncols, k, {});
    }
    GGML_ASSERT(ncand <= BLOCK2*16);
    if (ncand <= BLOCK2*4) {
        top_k_row_select<BLOCK2, 4><<<nrows, BLOCK2, 0, stream>>>(keys_alloc.get(), idxs_alloc.get(), dst, ncand, k);
    } else if (ncand <= BLOCK2*8) {
        top_k_row_select<BLOCK2, 8><<<nrows, BLOCK2, 0, stream>>>(keys_alloc.get(), idxs_alloc.get(), dst, ncand, k);
    } else {
        top_k_row_select<BLOCK2, 16><<<nrows, BLOCK2, 0, stream>>>(keys_alloc.get(), idxs_alloc.get(), dst, ncand, k);
    }
}

bool ggml_cuda_top_k_softcap_supported(int64_t ncols, int64_t nrows, int64_t k) {
    return nrows >= 1 && ncols > 4096 && k <= 128 && ((ncols + 4095) / 4096) * k <= 8192 && ncols < (1 << 24);
}

void ggml_cuda_top_k_softcap(ggml_backend_cuda_context & ctx, const float * x, int ncols, int nrows, int k,
        const ggml_cuda_softcap_params & sc, int32_t * ids, float * vals) {
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool & pool = ctx.pool();
    // the outputs may lie where the allocator reuses the logits' memory: then they go through the pool
    const char * x0 = (const char *) x;
    const char * x1 = x0 + (size_t) ncols*nrows*sizeof(float);
    const size_t n  = (size_t) k*nrows;
    const auto overlaps = [&](const void * p, size_t bytes) {
        return (const char *) p < x1 && (const char *) p + bytes > x0;
    };
    const bool tmp = overlaps(ids, n*sizeof(int32_t)) || overlaps(vals, n*sizeof(float));
    ggml_cuda_pool_alloc<int32_t> ids_tmp(pool, tmp ? n : 0);
    ggml_cuda_pool_alloc<float>   vals_tmp(pool, tmp ? n : 0);
    int32_t * ids_w  = tmp ? ids_tmp.get()  : ids;
    float   * vals_w = tmp ? vals_tmp.get() : vals;

    top_k_two_stage_cuda(pool, x, ids_w, ncols, nrows, k, stream, &sc);
    top_k_gather_softcap<<<(int) ((n + 255) / 256), 256, 0, stream>>>(x, ids_w, vals_w, ncols, k, (int) n, sc);
    if (tmp) {
        CUDA_CHECK(cudaMemcpyAsync(ids,  ids_w,  n*sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(vals, vals_w, n*sizeof(float),   cudaMemcpyDeviceToDevice, stream));
    }
}

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
    // several long rows (a batch of sampled rows): one radix select over all of them. GGML_CUDA_TOPK_RADIX=0 keeps a
    // call per row
    static const bool radix_on = getenv("GGML_CUDA_TOPK_RADIX") == nullptr || atoi(getenv("GGML_CUDA_TOPK_RADIX")) != 0;
    if (radix_on && nrows > 1 && ncols > 4096 && k <= 1024) {
        // a small k: the logits read once (top_k_two_stage_cuda); GGML_CUDA_TOPK_TWO_STAGE=0 keeps the radix passes
        static const bool two_stage = getenv("GGML_CUDA_TOPK_TWO_STAGE") == nullptr || atoi(getenv("GGML_CUDA_TOPK_TWO_STAGE")) != 0;
        if (two_stage && k <= 128 && ((ncols + 4095) / 4096) * k <= 8192) {
            top_k_two_stage_cuda(pool, src0_d, dst_d, (int) ncols, (int) nrows, (int) k, stream);
            return;
        }
        top_k_radix_cuda(pool, src0_d, dst_d, (int) ncols, (int) nrows, (int) k, stream);
        return;
    }
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
#if defined(GGML_USE_HIP)
    if (ncols > 1024) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    } else {
#endif // defined(GGML_USE_HIP)
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
#if defined(GGML_USE_HIP)
    }
#endif // defined(GGML_USE_HIP)
#endif
}
