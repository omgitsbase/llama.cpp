#pragma once

#include "common.cuh"

void ggml_cuda_argmax(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// The logits' softcap (scale a0, a1; tanh; scale b0, b1) applied to each element as the argmax reads it, in the
// expressions of the fused elementwise kernel, so the values compared are the ones the capped logits would hold.
// val_out takes each row's capped maximum: the chosen logit. Rows of x hold at least 16384 elements.
struct ggml_cuda_softcap_params {
    float a0, a1, b0, b1;
};

void ggml_cuda_argmax_softcap(ggml_backend_cuda_context & ctx, const ggml_tensor * x, int32_t * ids, float * val_out,
                              const ggml_cuda_softcap_params & sc);

