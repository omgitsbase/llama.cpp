#pragma once

#include "common.cuh"

// A batch of 8 columns (a verify of seven drafted tokens) against IQ3 weights with int8 tensor cores. The
// vector kernel repeats its integer dot products and activation loads for every column, which by eight columns
// costs it nearly twice a single column's time; one mma here covers 16 rows, 32 weights and all eight columns.

// whether this path takes a multiplication of src0 type `type` with `ncols_dst` columns on a device of compute
// capability `cc` (plain 2-D, no expert ids)
bool ggml_cuda_mmvq_mma_supported(ggml_type type, int cc, int64_t ncols_dst);

void ggml_cuda_mmvq_mma(
        const void * vx, ggml_type type, const void * vy, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        int ncols_x, int nrows_x, int ncols_dst, int stride_row_x, int stride_col_y, int stride_col_dst, cudaStream_t stream);
