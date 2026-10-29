#include "common.cuh"
#include "argmax.cuh"

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// the top k of soft-capped logits and the capped candidates, read from the logits before their cap (the batched sampler
// with its logits no graph output, see ggml-cuda.cu): whether the select takes this shape, and the select and gather
bool ggml_cuda_top_k_softcap_supported(int64_t ncols, int64_t nrows, int64_t k);
void ggml_cuda_top_k_softcap(ggml_backend_cuda_context & ctx, const float * x, int ncols, int nrows, int k,
        const ggml_cuda_softcap_params & sc, int32_t * ids, float * vals);
