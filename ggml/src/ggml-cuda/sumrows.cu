#include "reduce_rows.cuh"
#include "sumrows.cuh"

void sum_rows_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const int  id  = ggml_cuda_get_device();
    const int  nsm = ggml_cuda_info().devices[id].nsm;
    const dim3 block_nums(nrows, 1, 1);
    if ((nrows / nsm) < 2) {
        const dim3 block_dims(512, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(reduce_rows_f32</*norm=*/false>, launch_params, x, dst, ncols);
    } else {
        const dim3 block_dims(ncols < 1024 ? 32 : 128, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(reduce_rows_f32</*norm=*/false>, launch_params, x, dst, ncols);
    }
}

// dst[:, t] = sum over e of x[:, e, t]; one block per (token, column chunk).
// The columns of one token's experts are ne0 apart, tokens are s2 apart.
// dst[:, t] = sum over e of x[:, e, t]. Each thread owns four consecutive
// columns so a warp reads 512 bytes per expert row, and the expert loop is
// unrolled; columns of one token's experts are s1 apart, tokens s2 apart.
template <int n_expert>
static __global__ void sum_expert_rows_f32(const float * x_ptr, float * dst_ptr,
                                           const int ne0, const int64_t s1, const int64_t s2) {
    ggml_cuda_pdl_sync();
    const float * GGML_CUDA_RESTRICT x   = x_ptr;
    float       * GGML_CUDA_RESTRICT dst = dst_ptr;
    const int t    = blockIdx.y;
    const int col4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (col4 >= ne0) {
        return;
    }
    const float * src = x + t*s2 + col4;
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int e = 0; e < n_expert; ++e) {
        const float4 v = *(const float4 *) (src + e*s1);
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    *(float4 *) (dst + t*ne0 + col4) = acc;
}

static __global__ void sum_expert_rows_f32_any(const float * x_ptr, float * dst_ptr,
                                               const int ne0, const int n_expert, const int64_t s1, const int64_t s2) {
    ggml_cuda_pdl_sync();
    const float * GGML_CUDA_RESTRICT x   = x_ptr;
    float       * GGML_CUDA_RESTRICT dst = dst_ptr;
    const int t   = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ne0) {
        return;
    }
    const float * src = x + t*s2 + col;
    float acc = 0.0f;
    for (int e = 0; e < n_expert; ++e) {
        acc += src[e*s1];
    }
    dst[t*ne0 + col] = acc;
}

void ggml_cuda_op_sum_expert_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * experts, ggml_tensor * dst, int n_expert) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int     ne0 = experts->ne[0];
    const int     nt  = dst->ne[1];
    const int64_t s1  = experts->nb[1] / sizeof(float);
    const int64_t s2  = experts->nb[2] / sizeof(float);
    cudaStream_t stream = ctx.stream();

    const bool vec4 = (ne0 % 4 == 0) && (s1 % 4 == 0) && (s2 % 4 == 0) &&
                      ((uintptr_t) experts->data % 16 == 0) && ((uintptr_t) dst->data % 16 == 0);
    if (vec4 && (n_expert == 4 || n_expert == 6 || n_expert == 8 || n_expert == 10 || n_expert == 16)) {
        const dim3 block_dims(128, 1, 1);
        const dim3 block_nums((ne0/4 + 127) / 128, nt, 1);
        const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        switch (n_expert) {
            case  4: ggml_cuda_kernel_launch(sum_expert_rows_f32< 4>, lp, (const float *) experts->data, (float *) dst->data, ne0, s1, s2); break;
            case  6: ggml_cuda_kernel_launch(sum_expert_rows_f32< 6>, lp, (const float *) experts->data, (float *) dst->data, ne0, s1, s2); break;
            case  8: ggml_cuda_kernel_launch(sum_expert_rows_f32< 8>, lp, (const float *) experts->data, (float *) dst->data, ne0, s1, s2); break;
            case 10: ggml_cuda_kernel_launch(sum_expert_rows_f32<10>, lp, (const float *) experts->data, (float *) dst->data, ne0, s1, s2); break;
            case 16: ggml_cuda_kernel_launch(sum_expert_rows_f32<16>, lp, (const float *) experts->data, (float *) dst->data, ne0, s1, s2); break;
        }
        return;
    }
    const dim3 block_dims(256, 1, 1);
    const dim3 block_nums((ne0 + 255) / 256, nt, 1);
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
    ggml_cuda_kernel_launch(sum_expert_rows_f32_any, lp, (const float *) experts->data, (float *) dst->data, ne0, n_expert, s1, s2);
}

void ggml_cuda_op_sum_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const dim3 block_nums(nrows, 1, 1);

    const int id  = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;
    if ((nrows / nsm) < 2) {
        // Increase num threads to 512 for small nrows to better hide the latency
        const dim3 block_dims(512, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(reduce_rows_f32</*norm=*/false>, launch_params, src0_d, dst_d, ncols);
    } else {
        // Enough active SMs to hide latency, use smaller blocks to allow better scheduling
        const dim3 block_dims(ncols < 1024 ? 32 : 128, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(reduce_rows_f32</*norm=*/false>, launch_params, src0_d, dst_d, ncols);
    }
}
