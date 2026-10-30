#include "common.cuh"

// rows one block routes; up to this many rows the outputs may alias the logits
#define GGML_CUDA_TOPK_MOE_ROWS_PER_BLOCK 4
#include "ggml.h"

#include <initializer_list>

// Rows that one CUDA block handles: upstream's name for the constant above, which this fork's kernel (warp
// reduce, every warp reaching the barrier) is launch-bounded on.
#define TOPK_MOE_ROWS_PER_BLOCK GGML_CUDA_TOPK_MOE_ROWS_PER_BLOCK

struct ggml_cuda_topk_moe_args {
    bool sigmoid{};
    bool sqrt_softplus{};
    bool softmax{};
    bool delayed_softmax{};
    bool prob_bias{};
    bool norm{};
    bool scale{};
};

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args,
                           const ggml_tensor *             extra_logit = nullptr, // a stacked shared expert's gate logits
                           int                             extra_id    = 0);

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids);
