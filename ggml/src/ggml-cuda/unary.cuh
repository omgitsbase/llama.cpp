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

void ggml_cuda_op_swiglu_clamp(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

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

__device__ __forceinline__ float ggml_cuda_op_swiglu_clamp_single(float gate, float up, float limit) {
    gate = fminf(gate, limit);
    up = fmaxf(fminf(up, limit), -limit);

    return ggml_cuda_op_silu_single(gate) * up;
}

static __device__ __forceinline__ float op_abs(float x) {
    return fabsf(x);
}

static __device__ __forceinline__ float op_sgn(float x) {
    return (x > 0.f ? 1.f : ((x < 0.f ? -1.f : 0.f)));
}

static __device__ __forceinline__ float op_neg(float x) {
    return -x;
}

static __device__ __forceinline__ float op_step(float x) {
    return x > 0.0f;
}

static __device__ __forceinline__ float op_gelu(float x) {
    return ggml_cuda_op_gelu_single(x);
}

static __device__ __forceinline__ float op_gelu_erf(float x) {
    const float SQRT_2_INV = 0.70710678118654752440084436210484f;

    return 0.5f*x*(1.0f + erff(x*SQRT_2_INV));
}

static __device__ __forceinline__ float op_gelu_quick(float x) {
    const float GELU_QUICK_COEF = -1.702f;

    return x * (1.0f / (1.0f + expf(GELU_QUICK_COEF * x)));
}

static __device__ __forceinline__ float op_silu(float x) {
    return ggml_cuda_op_silu_single(x);
}

static __device__ __forceinline__ float op_tanh(float x) {
    return tanhf(x);
}

static __device__ __forceinline__ float op_relu(float x) {
    return fmaxf(x, 0);
}

static __device__ __forceinline__ float op_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static __device__ __forceinline__ float op_hardsigmoid(float x) {
    return fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static __device__ __forceinline__ float op_hardswish(float x) {
    return x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static __device__ __forceinline__ float op_exp(float x) {
    return expf(x);
}

static __device__ __forceinline__ float op_sqr(float x) {
    return x * x;
}

static __device__ __forceinline__ float op_relu_sqr(float x) {
    const float r = fmaxf(x, 0.0f);
    return r * r;
}

static __device__ __forceinline__ float op_sqrt(float x) {
    return sqrtf(x);
}

static __device__ __forceinline__ float op_sin(float x) {
    return sinf(x);
}

static __device__ __forceinline__ float op_cos(float x) {
    return cosf(x);
}

static __device__ __forceinline__ float op_log(float x) {
    return logf(x);
}

static __device__ __forceinline__ float op_expm1(float x) {
    return expm1f(x);
}

static __device__ __forceinline__ float op_softplus(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

static __device__ __forceinline__ float op_elu(float x) {
    return (x > 0.f) ? x : expm1f(x);
}

static __device__ __forceinline__ float op_floor(float x) {
    return floorf(x);
}

static __device__ __forceinline__ float op_ceil(float x) {
    return ceilf(x);
}

static __device__ __forceinline__ float op_round(float x) {
    return round(x);
}

static __device__ __forceinline__ float op_trunc(float x) {
    return trunc(x);
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
    GGML_CUDA_CHAIN_GLU   = 200, // + ggml_glu_op: the operand is the gate, the running value the up
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

static __device__ __forceinline__ float ggml_cuda_chain_unary(int op, float x) {
    switch (op) {
        case GGML_UNARY_OP_ABS:         return op_abs(x);
        case GGML_UNARY_OP_SGN:         return op_sgn(x);
        case GGML_UNARY_OP_NEG:         return op_neg(x);
        case GGML_UNARY_OP_STEP:        return op_step(x);
        case GGML_UNARY_OP_TANH:        return op_tanh(x);
        case GGML_UNARY_OP_ELU:         return op_elu(x);
        case GGML_UNARY_OP_RELU:        return op_relu(x);
        case GGML_UNARY_OP_SIGMOID:     return op_sigmoid(x);
        case GGML_UNARY_OP_GELU:        return op_gelu(x);
        case GGML_UNARY_OP_GELU_QUICK:  return op_gelu_quick(x);
        case GGML_UNARY_OP_SILU:        return op_silu(x);
        case GGML_UNARY_OP_HARDSWISH:   return op_hardswish(x);
        case GGML_UNARY_OP_HARDSIGMOID: return op_hardsigmoid(x);
        case GGML_UNARY_OP_EXP:         return op_exp(x);
        case GGML_UNARY_OP_EXPM1:       return op_expm1(x);
        case GGML_UNARY_OP_SOFTPLUS:    return op_softplus(x);
        case GGML_UNARY_OP_GELU_ERF:    return op_gelu_erf(x);
        case GGML_UNARY_OP_FLOOR:       return op_floor(x);
        case GGML_UNARY_OP_CEIL:        return op_ceil(x);
        case GGML_UNARY_OP_ROUND:       return op_round(x);
        case GGML_UNARY_OP_TRUNC:       return op_trunc(x);
        default:                        return x;
    }
}

// a gated unit on one element, in the operand order of the separate kernels: op(gate) * up
static __device__ __forceinline__ float ggml_cuda_glu_single(const int op, const float g, const float u, const float alpha, const float limit) {
    switch (op) {
        case GGML_GLU_OP_REGLU:       return op_relu(g) * u;
        case GGML_GLU_OP_GEGLU:       return op_gelu(g) * u;
        case GGML_GLU_OP_SWIGLU:      return op_silu(g) * u;
        case GGML_GLU_OP_SWIGLU_OAI:  return ggml_cuda_op_swiglu_oai_single(g, u, alpha, limit);
        case GGML_GLU_OP_SWIGLU_CLAMP: return ggml_cuda_op_swiglu_clamp_single(g, u, limit);
        case GGML_GLU_OP_GEGLU_ERF:   return op_gelu_erf(g) * u;
        case GGML_GLU_OP_GEGLU_QUICK: return op_gelu_quick(g) * u;
        default:                      return u;
    }
}

// one chain step on the running value v; o is the step's operand, if it has one
static __device__ __forceinline__ float ggml_cuda_chain_apply(const ggml_cuda_chain_step & st, const float v, const float o) {
    switch (st.op) {
        case GGML_CUDA_CHAIN_ADD:   return v + o;
        case GGML_CUDA_CHAIN_SUB:   return v - o;
        case GGML_CUDA_CHAIN_MUL:   return v * o;
        case GGML_CUDA_CHAIN_DIV:   return v / o;
        case GGML_CUDA_CHAIN_RSUB:  return o - v;
        case GGML_CUDA_CHAIN_RDIV:  return o / v;
        case GGML_CUDA_CHAIN_SCALE: return st.p0 * v + st.p1;
        case GGML_CUDA_CHAIN_CLAMP: return fminf(fmaxf(v, st.p0), st.p1);
        case GGML_CUDA_CHAIN_SQR:   return v * v;
        case GGML_CUDA_CHAIN_SQRT:  return sqrtf(v);
        case GGML_CUDA_CHAIN_LOG:   return logf(v);
        case GGML_CUDA_CHAIN_SIN:   return sinf(v);
        case GGML_CUDA_CHAIN_COS:   return cosf(v);
        default:
            return st.op >= GGML_CUDA_CHAIN_GLU ? ggml_cuda_glu_single(st.op - GGML_CUDA_CHAIN_GLU, o, v, st.p0, st.p1) :
                                                  ggml_cuda_chain_unary(st.op - GGML_CUDA_CHAIN_UNARY, v);
    }
}

// The elementwise tail a matvec applies to its own result before the store (mul_mat_vec_q fusion=1). Operand k
// is read at row*s_row + col*s_col + channel*s_channel + sample*s_sample of the matvec's output index.
#define GGML_CUDA_MM_EPI_MAX_STEPS 6
#define GGML_CUDA_MM_EPI_MAX_ARGS  3

struct ggml_cuda_mm_epi_arg {
    const float * data;
    int s_row, s_col, s_channel, s_sample;
};

struct ggml_cuda_mm_epilogue {
    ggml_cuda_mm_epi_arg arg[GGML_CUDA_MM_EPI_MAX_ARGS];
    ggml_cuda_chain_step step[GGML_CUDA_MM_EPI_MAX_STEPS];
    int n_args;
    int n_steps;
};

// the kernels without an epilogue take this in its place, so their parameters do not grow
struct ggml_cuda_mm_epilogue_none {};

template <int fusion>
using ggml_cuda_mm_epilogue_t = std::conditional_t<fusion == 1, ggml_cuda_mm_epilogue, ggml_cuda_mm_epilogue_none>;

static __device__ __forceinline__ float ggml_cuda_mm_epi_load(const ggml_cuda_mm_epi_arg & a, const int row, const int col, const int channel, const int sample) {
    return a.data[row*a.s_row + col*a.s_col + channel*a.s_channel + sample*a.s_sample];
}

static __device__ __forceinline__ float ggml_cuda_mm_epi_apply(const ggml_cuda_mm_epilogue & epi, float v, const float o0, const float o1, const float o2) {
#pragma unroll 1
    for (int s = 0; s < epi.n_steps; ++s) {
        const ggml_cuda_chain_step st = epi.step[s];
        v = ggml_cuda_chain_apply(st, v, st.arg == 0 ? o0 : st.arg == 1 ? o1 : o2);
    }
    return v;
}

bool ggml_cuda_chain_unary_supported(int op);
void ggml_cuda_op_elt_chain(ggml_backend_cuda_context & ctx, const ggml_cuda_chain_args & a);
