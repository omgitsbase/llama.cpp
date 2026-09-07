#include "common.cuh"

void sum_rows_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, cudaStream_t stream);
void ggml_cuda_op_sum_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// sum the expert rows of a [n_embd, n_expert, n_tokens] tensor into dst [n_embd, n_tokens]
void ggml_cuda_op_sum_expert_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * experts, ggml_tensor * dst, int n_expert);
