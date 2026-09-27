#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

// fold: src1 was never written; quantize its norm input with the norm applied
// instead, once for every reader of the fold. q8_pre: src1 already quantized by the caller
void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr,
    ggml_cuda_norm_fold * fold = nullptr, const void * q8_pre = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

// a delta-net layer's input projection and the causal convolution after it: each row's lane takes the channel's
// projected values for every token, convolves [state, values] with the channel's taps as ssm_conv does, stores the
// result and publishes the state snapshots, so neither the projection nor the window is ever written
#define GGML_CUDA_CONV_SNAP_MAX 8

struct ggml_cuda_mmvq_conv {
    const float   * state;     // channel c's d_conv-1 state values at c*state_s1
    const int32_t * state_ids; // a folded gather: the state row is state_ids[0], state_row floats apart
    int64_t         state_row;
    int             state_s1;
    const float   * w;         // channel c's d_conv taps at c*w_s1
    int             w_s1;
    const float   * bias;      // one per channel, or nullptr
    float         * y;         // token t of channel c at t*y_s1 + c
    int             y_s1;
    int             silu;
    int             n_snap;
    int             snap_s1;   // floats between channels in a snapshot
    float         * snap[GGML_CUDA_CONV_SNAP_MAX];
    int             snap_idx[GGML_CUDA_CONV_SNAP_MAX]; // window position of each snapshot's first value
};

bool ggml_cuda_mmvq_conv_supported(ggml_type type, int cc, int64_t ncols_dst);
void ggml_cuda_mul_mat_vec_q_conv(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, ggml_cuda_norm_fold * fold,
    const ggml_cuda_mmvq_conv & conv);

// two quantized matvecs over one activation, each within one wave, in one launch
bool ggml_cuda_mmvq_pair_ok(const ggml_tensor * a, const ggml_tensor * b, int device);
void ggml_cuda_mul_mat_vec_q_pair(ggml_backend_cuda_context & ctx, const ggml_tensor * a, const ggml_tensor * b, ggml_cuda_norm_fold * fold);
