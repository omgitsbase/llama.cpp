#pragma once

#include "common.cuh"

// The block-scaled FP4 GEMM of Blackwell's tensor cores through cuBLASLt, as vLLM runs NVFP4 checkpoints: for NVFP4 and
// MXFP4 weights in a step over several sequences, the activations quantized on the fly to the weights' format and the
// weights repacked once into the layout cuBLASLt reads, kept next to the originals.

// GGML_CUDA_FP4_GEMM=2: for one sequence's products too, to test it
bool ggml_cuda_fp4_gemm_forced();

// whether ggml_cuda_fp4_gemm can take dst = src0 x src1
bool ggml_cuda_fp4_gemm_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

// two to four NVFP4 matmuls over one activation as one GEMM (q, k and v; gate and up): every matrix but the last with
// a multiple of 128 rows. false when the group cannot be prepared now (capturing), the matmuls then run alone.
// w_s: each matrix's per-tensor scale on the device (the graph's mul after it), or nullptr
bool ggml_cuda_fp4_gemm_group(ggml_backend_cuda_context & ctx, const ggml_tensor * const * src0s, int n_mats, const ggml_tensor * src1,
        ggml_tensor * const * dsts, const float * const * w_s = nullptr);
// a gate and up pair whose only reader is a GEGLU (vLLM's merged gate_up_proj and its gelu_and_mul): the product goes
// straight to gelu(gate)*up in glu, which also leaves its rows' |max| for the down projection's quantizer.
// gate_part is the gate's index in src0s; the parts' own outputs are not written. down_w, the GEGLU's only reader, and
// down_ws, its weight scale: an NVFP4 down projection with a static input scale has its activation quantized here.
bool ggml_cuda_fp4_gemm_group_geglu(ggml_backend_cuda_context & ctx, const ggml_tensor * const * src0s, const ggml_tensor * src1,
        int gate_part, ggml_tensor * glu, const float * const * w_s = nullptr, const ggml_tensor * down_w = nullptr,
        const float * down_ws = nullptr);

// false when it cannot run now (no cuBLASLt algorithm for the shape, or weights still to repack while a graph is
// captured); the caller then runs its usual path. w_s: the weight's per-tensor scale, folded into alpha
bool ggml_cuda_fp4_gemm(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const float * w_s = nullptr);

// an NVFP4 weight's static input scale (1/the checkpoint's input global scale, F32 on the device): its activations are
// quantized with it, as vLLM does, instead of by their |max|. nullptr forgets it
void ggml_cuda_fp4_set_input_scale(const void * w_data, const float * s_data);

// A producer that writes a tensor row by row (the fused rms_norm) leaves each row's |max| here, and a consumer
// that quantizes the tensor as a whole (the NVFP4 GEMM) takes the tensor's |max| from those instead of a pass of
// its own. The rows are written on the device in graph order, so a captured graph replays them like any kernel.
// Row |max| of tensor t, n_rows rows written in this graph evaluation: the device array to fill, or nullptr
float * ggml_cuda_row_amax_begin(ggml_backend_cuda_context & ctx, const ggml_tensor * t, const void * data, int64_t n_rows);
// the array a producer filled for tensor t in this graph evaluation, or nullptr
const float * ggml_cuda_row_amax_find(ggml_backend_cuda_context & ctx, const ggml_tensor * t, const void * data, int64_t n_rows);
void ggml_cuda_row_amax_graph_begin(ggml_backend_cuda_context & ctx);

// A small quantized weight (the per-layer embedding gate and projection, a drafter's layers) over several sequences'
// rows: an F32 copy of it, dequantized once and kept, through a TF32 GEMM, as vLLM runs those in BF16. The quantized
// kernels' activation quantization, split-K reduce and stream-k fixup cost more than such a GEMM. One sequence keeps
// its kernels; GGML_CUDA_SMALL_GEMM=0 turns it off.
bool ggml_cuda_small_gemm_applies(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
// false when it cannot run now (the copy still to make while a graph is captured); the caller then runs its usual path
bool ggml_cuda_small_gemm(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// drop the repacked copies of weights in [base, base + size), called when their buffer is freed
void ggml_cuda_fp4_gemm_forget(int device, const void * base, size_t size);

// a graph is about to run: activations quantized for an earlier graph are not reused, whatever their address
void ggml_cuda_fp4_gemm_graph_begin(ggml_backend_cuda_context & ctx);

// FP8 E4M3 weights (GGML_TYPE_F8_E4M3): whether dst = src0 x src1 runs here, and the product with each weight row's
// scale s_row (src0->ne[1] floats on the device, or nullptr) and a tensor scale s_t (one float, or nullptr)
bool ggml_cuda_fp8_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
bool ggml_cuda_fp8_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const float * s_row, const float * s_t);
// an FP8 gate and up pair over one activation whose only reader is a GEGLU: gelu(gate)*up into glu, each weight's
// scales (per row, or one) applied on the way; false when it cannot run here
bool ggml_cuda_fp8_gemm_geglu(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * src1,
        ggml_tensor * glu, const float * sg_row, const float * su_row, const float * sg_t, const float * su_t,
        bool down_fp8 = false);
// the product of an FP8 GEMM over several sequences' rows (more than 8 columns) without its scales: d [ncols][rows], and
// the activations' per-token scales in s_tok (device); false when it cannot run here
bool ggml_cuda_fp8_mul_mat_raw(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, float * d,
        const float ** s_tok);

// vLLM's dynamic per-token FP8 scale from a row's |max|: the GEMM's own quantizer and the producers that quantize their
// output for it share it, so their bytes are the same
static __device__ __forceinline__ float ggml_cuda_fp8_act_scale(const float amax) {
    return fmaxf(amax / 448.0f, 1.0f / (448.0f*512.0f));
}

// The E4M3 rows and per-token scales the next FP8 GEMM over act (ne[1] rows of ne[0]) reads, for a producer that
// quantizes its own output as it writes it (the norm pair): the GEMM then takes them in place of its quantize pass, as
// vLLM's compiled norm quantizes for its GEMM. false when the GEMM would not read them (one sequence's matvec) or the
// buffer cannot grow now (a graph being captured)
// may_grow false: the caller holds pointers into the buffer (another activation's scales), so it must not move
bool ggml_cuda_fp8_act_claim(ggml_backend_cuda_context & ctx, const ggml_tensor * act, uint8_t ** q, float ** s, bool may_grow = true);

