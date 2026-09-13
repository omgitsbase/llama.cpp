#pragma once

#include "common.cuh"

// ggml_quantize_chunk's signature, run on the device. Returns the bytes
// written, or 0 for anything this file does not handle, which is the caller's
// signal to quantize on the CPU instead.
extern "C" GGML_BACKEND_API size_t ggml_cuda_quantize_chunk(int type, const float * src, void * dst, int64_t start,
                                                            int64_t nrows, int64_t n_per_row, const float * imatrix);

// Whether the call above would take this tensor, so a caller can keep its own
// threads on the rows this file leaves to the CPU.
extern "C" GGML_BACKEND_API bool ggml_cuda_quantize_supported(int type, int64_t n_per_row, bool has_imatrix);
