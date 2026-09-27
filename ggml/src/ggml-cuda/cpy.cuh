#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);

// copies of 4-byte elements that neither read nor write what another of them writes, launched as one kernel: each
// item is ggml_cpy's element order, a flat index walked through the source's shape and then the destination's
#define GGML_CUDA_CPY_RUN_MAX 8

struct ggml_cuda_cpy_run_item {
    const char * src;
    char       * dst;
    int64_t      ne;
    int64_t      ne0[3]; // source ne[0..2]
    int64_t      nb0[4];
    int64_t      ne1[3]; // destination ne[0..2]
    int64_t      nb1[4];
};

struct ggml_cuda_cpy_run {
    int                    n;
    ggml_cuda_cpy_run_item item[GGML_CUDA_CPY_RUN_MAX];
};

void ggml_cuda_cpy_run_4b(const ggml_cuda_cpy_run & run, cudaStream_t stream);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
