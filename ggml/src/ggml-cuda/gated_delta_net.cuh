#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

// the l2 norms of q and k a delta-net layer runs just before its recurrence, done by the recurrence at dst as it loads
// them, each as l2_norm_f32 computes it: dst then reads q and k from these unnormalized sources. Taken by that launch.
void ggml_cuda_gated_delta_net_set_qk_norm(const ggml_tensor * dst, const ggml_tensor * q, const ggml_tensor * k, float eps);
// how many are still waiting for their recurrence; clear drops them
int  ggml_cuda_gated_delta_net_qk_norm_pending();
void ggml_cuda_gated_delta_net_qk_norm_clear();
