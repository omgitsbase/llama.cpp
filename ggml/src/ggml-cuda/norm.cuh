#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// a residual add folded into the rms_norm and weight-multiply that follow it
void ggml_cuda_op_add_rms_norm_mul(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * norm, ggml_tensor * mul);

// out, when given, receives the product in place of mul_tensor's buffer
void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor, float * out = nullptr);

// rms_norm > scale(const) > mul(weight), the constant folded into the epilogue
void ggml_cuda_op_rms_norm_fused_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor, ggml_tensor * mul_tensor);

// rms_norm > mul > add [> mul(scalar)] > rms_norm > mul in one launch, see norm.cu. q8, q8_s: the FP8 GEMM reading the
// second product takes its rows quantized here (ggml_cuda_fp8_act_claim)
void ggml_cuda_op_rms_norm_pair(ggml_backend_cuda_context & ctx, ggml_tensor * norm1, ggml_tensor * mul1, ggml_tensor * add1,
                                ggml_tensor * scale1, ggml_tensor * norm2, ggml_tensor * mul2,
                                const float * x_tok = nullptr, const float * x_row = nullptr, const float * x_t = nullptr,
                                uint8_t * q8 = nullptr, float * q8_s = nullptr);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor,
                                     ggml_tensor *               scale_tensor = nullptr,
                                     nv_bfloat16 *               dst_bf16     = nullptr);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// two l2_norms over adjacent views of one tensor, as one launch
// two l2 norms of one shape and eps whose sources share strides, anywhere: one launch
void ggml_cuda_op_l2_norm_pair2(ggml_backend_cuda_context & ctx, ggml_tensor * a, ggml_tensor * b);

void ggml_cuda_op_l2_norm_pair(ggml_backend_cuda_context & ctx, ggml_tensor * a, ggml_tensor * b);
