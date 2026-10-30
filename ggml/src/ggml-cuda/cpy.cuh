#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);

// copies of 4-byte elements that neither read nor write what another of them writes, launched as one kernel: each
// item is ggml_cpy's element order, a flat index walked through the source's shape and then the destination's
#define GGML_CUDA_CPY_RUN_MAX 8

struct ggml_cuda_cpy_run_item {
    const char * src;
    char       * dst;
    int64_t      ne;     // under 2^31
    uint3        fd0[3]; // source ne[0], ne[0]*ne[1], ne[0]*ne[1]*ne[2] (init_fastdiv_values)
    int64_t      nb0[4];
    uint3        fd1[3]; // the destination's
    int64_t      nb1[4];
    bool         dst_contig;
};

struct ggml_cuda_cpy_run {
    int                    n;
    ggml_cuda_cpy_run_item item[GGML_CUDA_CPY_RUN_MAX];
};

void ggml_cuda_cpy_run_4b(const ggml_cuda_cpy_run & run, cudaStream_t stream);

// copies of windows of one source's rows (each item's source the same rows from a further element of dim 0) into
// contiguous destinations: a thread a row reads it once for every window
struct ggml_cuda_cpy_windows {
    const char * base;   // the rows at the first window
    int64_t      nb1;
    int64_t      nb2;
    uint3        fd_ne1; // rows a dim-2 index (init_fastdiv_values)
    uint32_t     n_rows; // ne1*ne2
    int          ne0;    // a window's elements
    int          n;
    int          off[GGML_CUDA_CPY_RUN_MAX]; // each window's first element
    char *       dst[GGML_CUDA_CPY_RUN_MAX];
};

void ggml_cuda_cpy_windows_4b(const ggml_cuda_cpy_windows & w, cudaStream_t stream);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
