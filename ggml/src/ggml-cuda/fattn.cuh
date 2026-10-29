#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// the attention times a scale on its output (an FP8 cache's V scale), the scale applied as the separate op would
void ggml_cuda_flash_attn_ext_scaled(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float scale);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);

// a new graph: the mask scans kept across a graph's layers (fattn-common.cuh) are stale
void ggml_cuda_kv_max_graph_begin(ggml_backend_cuda_context & ctx);
