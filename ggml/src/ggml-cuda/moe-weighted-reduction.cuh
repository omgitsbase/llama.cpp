#include "common.cuh"

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst);

// the same sum with each slot's rows times its expert's scale, scale[ids[token, slot]] (an NVFP4 expert's second level)
void ggml_cuda_op_moe_weighted_reduction_ids(ggml_backend_cuda_context & ctx,
                                             const ggml_tensor *         experts,
                                             const ggml_tensor *         scale,
                                             const ggml_tensor *         ids,
                                             const ggml_tensor *         weights,
                                             ggml_tensor *               dst);
