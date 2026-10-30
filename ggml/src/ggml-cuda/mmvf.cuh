#include "common.cuh"

#define MMVF_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVF kernels.

// two float matvecs over one activation with their epilogues in one launch; each row sums as mul_mat_vec_f does
void ggml_cuda_mul_mat_vec_f_pair_epi(ggml_backend_cuda_context & ctx, const ggml_tensor * wa, const ggml_tensor * wb,
        const ggml_tensor * src1, const ggml_cuda_norm_fold * fold, float * dst_a, float * dst_b,
        const ggml_cuda_mm_epilogue & epi_a, const ggml_cuda_mm_epilogue & epi_b, int ncols_dst);
// the column counts it is built for
bool ggml_cuda_mul_mat_vec_f_pair_ncols_ok(int ncols_dst);

void ggml_cuda_mul_mat_vec_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion = nullptr, ggml_cuda_norm_fold * fold = nullptr);

void ggml_cuda_op_mul_mat_vec_f(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

bool ggml_cuda_should_use_mmvf(enum ggml_type type, int cc, const int64_t * src0_ne, const size_t * src0_nb, int64_t ne11);
