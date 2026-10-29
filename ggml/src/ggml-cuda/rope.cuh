#include "common.cuh"

#define CUDA_ROPE_BLOCK_SIZE 256

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * set_rows);

void ggml_cuda_op_rms_norm_mul_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * rope, ggml_tensor * set_rows, const ggml_tensor * hadamard = nullptr);

// a fused q/k/v projection's epilogue in one launch: d holds the product ([n_tok][ld_d], q heads then k then v, hd
// each), times s_tok per token, s_row per column and s_t (any may be nullptr); the q and k heads RMS-normed with
// norm_q's/norm_k's eps, times wq/wk and roped (rope's parameters), the v heads normed with norm_v's eps. q goes to
// q_out ([n_tok][n_q][hd]), k and v to rows k_idx/v_idx of F16 caches with k_stride/v_stride elements a row
void ggml_cuda_qkv_norm_rope_store(ggml_backend_cuda_context & ctx, const float * d, int64_t ld_d,
        const float * s_tok, const float * s_row, const float * s_t, int hd, int n_q, int n_k, int64_t n_tok,
        const ggml_tensor * norm_q, const ggml_tensor * norm_k, const ggml_tensor * norm_v,
        const float * wq, const float * wk, float * q_out,
        void * k_cache, const int64_t * k_idx, int64_t k_stride, void * v_cache, const int64_t * v_idx, int64_t v_stride,
        const ggml_tensor * rope, bool f8 = false, float k_mul = 1.0f, float k_add = 0.0f, float v_mul = 1.0f, float v_add = 0.0f);
