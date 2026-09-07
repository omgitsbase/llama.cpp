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
static __global__ void sum_expert_rows_f32(const float * __restrict__ x, float * __restrict__ dst,
                                           const int ne0, const int n_expert, const int64_t s1, const int64_t s2) {
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

    const dim3 block_dims(256, 1, 1);
    const dim3 block_nums((ne0 + 255) / 256, nt, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, ctx.stream());
    ggml_cuda_kernel_launch(sum_expert_rows_f32, launch_params, (const float *) experts->data, (float *) dst->data, ne0, n_expert, s1, s2);
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
