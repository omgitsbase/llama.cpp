#pragma once
#include "common.cuh"

#define CUDA_NEG_BLOCK_SIZE 256
#define CUDA_STEP_BLOCK_SIZE 256
#define CUDA_GELU_BLOCK_SIZE 256
#define CUDA_SILU_BLOCK_SIZE 256
#define CUDA_SILU_BACK_BLOCK_SIZE 256
#define CUDA_TANH_BLOCK_SIZE 256
#define CUDA_RELU_BLOCK_SIZE 256
#define CUDA_SIGMOID_BLOCK_SIZE 256
#define CUDA_HARDSIGMOID_BLOCK_SIZE 256
#define CUDA_EXP_BLOCK_SIZE 256
#define CUDA_HARDSWISH_BLOCK_SIZE 256
#define CUDA_SQR_BLOCK_SIZE 256
#define CUDA_SQRT_BLOCK_SIZE 256
#define CUDA_SIN_BLOCK_SIZE 256
#define CUDA_COS_BLOCK_SIZE 256
#define CUDA_GLU_BLOCK_SIZE 256
#define CUDA_XIELU_BLOCK_SIZE 256

void ggml_cuda_op_abs(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sgn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_neg(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_step(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_gelu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_silu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_silu_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_gelu_erf(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_gelu_quick(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_tanh(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_hardsigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_exp(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_hardswish(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_leaky_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sqrt(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sin(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_cos(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_log(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_expm1(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_softplus(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_elu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_floor(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_ceil(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_round(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_trunc(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_reglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_geglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_swiglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_swiglu_oai(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_geglu_erf(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_geglu_quick(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_xielu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_unary_mul(ggml_backend_cuda_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node);

void ggml_cuda_op_relu_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * relu_node, ggml_tensor * sqr_node);

__device__ __forceinline__ float ggml_cuda_op_silu_single(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float ggml_cuda_op_gelu_single(float x) {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;

    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

__device__ __forceinline__ float ggml_cuda_op_swiglu_oai_single(float x, float g, float alpha = 1.702f, float limit = 7.0f) {
    x = fminf(x, limit);
    g = fmaxf(fminf(g, limit), -limit);

    float out_glu = x / (1.0f + expf(-x * alpha));
    out_glu = out_glu * (1.0f + g);
    return out_glu;
}

// A chain of elementwise ops run as one launch. Each step applies one op
// to the running value, with an optional second operand read through
// broadcast strides. The expressions are the ones the separate kernels use,
// so the result is the same to the bit.
#define GGML_CUDA_CHAIN_MAX_STEPS 8
#define GGML_CUDA_CHAIN_MAX_ARGS  8

enum ggml_cuda_chain_op {
    GGML_CUDA_CHAIN_ADD = 1, GGML_CUDA_CHAIN_SUB, GGML_CUDA_CHAIN_MUL, GGML_CUDA_CHAIN_DIV,
    GGML_CUDA_CHAIN_RSUB, GGML_CUDA_CHAIN_RDIV,
    GGML_CUDA_CHAIN_SCALE = 10, GGML_CUDA_CHAIN_CLAMP, GGML_CUDA_CHAIN_SQR, GGML_CUDA_CHAIN_SQRT,
    GGML_CUDA_CHAIN_LOG, GGML_CUDA_CHAIN_SIN, GGML_CUDA_CHAIN_COS,
    GGML_CUDA_CHAIN_UNARY = 100, // + ggml_unary_op
};

struct ggml_cuda_chain_arg {
    const float * data;
    int64_t       s[4]; // element strides, 0 where the operand broadcasts
};

struct ggml_cuda_chain_step {
    int   op;
    int   arg; // index into arg[], or -1
    float p0, p1;
};

struct ggml_cuda_chain_args {
    ggml_cuda_chain_arg  x;
    ggml_cuda_chain_arg  arg[GGML_CUDA_CHAIN_MAX_ARGS];
    ggml_cuda_chain_step step[GGML_CUDA_CHAIN_MAX_STEPS];
    int     n_steps;
    float * dst;
    int64_t ne[4];
};

bool ggml_cuda_chain_unary_supported(int op);
void ggml_cuda_op_elt_chain(ggml_backend_cuda_context & ctx, const ggml_cuda_chain_args & a);
