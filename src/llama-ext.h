#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"

#include <cstdint>
#include <map>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// Chained drafting for the DECODER_MTP graph: the next n draft steps are
// built into one graph, each step taking the argmax of the previous step's
// logits on the device, so a draft round costs one decode. The batch must
// hold exactly n positions, the first carrying the anchor token and hidden
// state. 0 restores one step per decode.
LLAMA_API void llama_set_mtp_chain(struct llama_context * ctx, int32_t n);

// For the next chained decodes of a drafter over the target's cache (Gemma 4), until reset: the round is launched right
// behind the target's verify, before the host has accepted anything, as vLLM's runner drafts from the rejection
// sampler's output on the device. The batch holds n + 1 rows per sequence, row j at the position of the anchor had the
// target accepted j drafts and carrying the verify's token j; the round's sequences (llama_set_mtp_round_seqs, with
// tgt_row/tgt_out from llama_get_last_rows) name where the target left the verify's samples and hidden rows. The graph
// counts the accepted drafts from those samples and drafts from that row; its last pending id row is the count.
LLAMA_API void llama_set_mtp_dev_anchors(struct llama_context * ctx, bool value);

// ctx's next graph waits, on the device, for the work queued on ctx_src so far
LLAMA_API void llama_order_after(struct llama_context * ctx, struct llama_context * ctx_src);

// tokens the target produced for seq, noted on the drafter's context; their counts rank the reduced draft head's rows
// at the next load (LLAMA_DRAFT_VOCAB sets where they are kept, "off" keeps none)
LLAMA_API void llama_draft_note_tokens(struct llama_context * ctx, llama_seq_id seq_id, const llama_token * tokens, int32_t n_tokens);

// ids drafted by the last chained MTP decode, n of them
LLAMA_API const llama_token * llama_get_mtp_chain_ids(struct llama_context * ctx, int32_t * n);

// one sequence of a chained MTP round: its replay rows, then one row per chain step. With c_ids nullptr the verify's
// ids and hidden rows come from the target's last graph (llama_get_last_rows); h_pend nullptr reuses the last round's
struct llama_mtp_round_seq {
    llama_seq_id    seq_id  = 0;
    const int32_t * c_ids   = nullptr; // [n_chain + 1]
    const float   * h_tgt   = nullptr; // [n_embd, n_chain + 1]
    const float   * h_pend  = nullptr; // [n_embd]
    int32_t         off     = 0;       // 1 in a normal round, 0 in the first one
    int32_t         tgt_row = 0;
    int32_t         tgt_out = 0;
};

// the sequences of the next chained decode; its batch holds n*(2*n_chain + 1) rows. After the decode every sequence
// takes the device route with off 1, as a single round does
LLAMA_API void llama_set_mtp_round_seqs(struct llama_context * ctx, const struct llama_mtp_round_seq * seqs, int32_t n);

// sequences of the last chained decode: llama_get_mtp_chain_ids returns n_chain + 1 rows of that many ids
LLAMA_API int32_t llama_get_mtp_chain_n_seq(const struct llama_context * ctx);

// Whether the last ubatch of the last decode holds n rows of seq at positions pos0 .. pos0 + n - 1 in a row, all
// of them outputs; row0 and out0 receive the first one's ubatch row and output row.
LLAMA_API bool llama_get_last_rows(const struct llama_context * ctx, llama_seq_id seq, llama_pos pos0, int32_t n, int32_t * row0, int32_t * out0);

// The next decode's embedding input takes, for the first row of seq seqs[i], ctx_src's last hidden row (h_nextn) of
// that sequence at position pos[i], copied on the device; the sequence's other rows are not uploaded either. ok[i]
// says whether that row is there (see llama_get_last_rows). Returns how many are. A sequence not there takes what
// the batch holds.
LLAMA_API int32_t llama_set_embd_device_rows(struct llama_context * ctx, struct llama_context * ctx_src, const llama_seq_id * seqs, const llama_pos * pos, int32_t n, bool * ok);

// For the next decode only: false leaves the hidden rows (h_nextn) on the device when the batch is one ubatch whose
// rows all sampled there, for a drafter that takes them with llama_set_embd_device_rows; the host copy
// (llama_get_embeddings_nextn_ith) is then not updated. Default true.
LLAMA_API void llama_set_embd_nextn_readback(struct llama_context * ctx, bool value);

// whether the sequence's sampler runs on the backend, so its samples are on the device
LLAMA_API bool llama_sampler_on_backend(struct llama_context * ctx, llama_seq_id seq_id);

// the target context a chained MTP context takes its rounds' inputs from
LLAMA_API void llama_set_mtp_round_target(struct llama_context * ctx, struct llama_context * ctx_tgt);

// the next llama_decode on ctx takes seq's positions pos0 + 1 .. pos0 + n from column col of the ids ctx_src's last
// chained MTP draft left on the device, for any number of sequences; false when ctx_src drafted fewer than n
LLAMA_API bool llama_set_tokens_device_seq(struct llama_context * ctx, struct llama_context * ctx_src, llama_seq_id seq, llama_pos pos0, int32_t n, int32_t col);

// The first draft left on the device turns the device form of the token input on for every later small decode. A
// server warming its graphs up before the first request turns it on first, so they are built in that form. False,
// and nothing changes, when the embeddings are not on the device.
LLAMA_API bool llama_set_tokens_device_enable(struct llama_context * ctx);

// The next llama_decode may run its batch as one forward over groups of sequences with equal token counts (a verify's
// rows and the prompt chunks of the same step), each group attending through its own cache view, as vLLM batches its
// decodes and prefills. Only for a cache with a stream per sequence; the batch must fit one ubatch.
LLAMA_API void llama_set_next_decode_grouped(struct llama_context * ctx, bool value);

// A sequence's state copied out beside the decodes. With LLAMA_STATE_SEQ_FLAGS_ASYNC, llama_state_seq_get_data_ext
// returns once the device has the copies queued on a stream of its own (the host part of the state is written at
// once). The buffer holds the state after llama_state_async_wait(ctx, dst) returned and must stay allocated until
// then. The copies read the sequence's cells: llama_state_async_wait_seq(ctx, seq_id) comes before anything writes
// them again. Without a device stream for it the copy is made synchronously, as without the flag.
#define LLAMA_STATE_SEQ_FLAGS_ASYNC 8
LLAMA_API void llama_state_async_wait(struct llama_context * ctx, const void * dst);
LLAMA_API void llama_state_async_wait_seq(struct llama_context * ctx, llama_seq_id seq_id);
LLAMA_API void llama_state_async_wait_all(struct llama_context * ctx);
LLAMA_API bool llama_state_async_pending_seq(struct llama_context * ctx, llama_seq_id seq_id);

// A sequence's state copied out beside the decodes. With LLAMA_STATE_SEQ_FLAGS_ASYNC, llama_state_seq_get_data_ext
// returns once the device has the copies queued on a stream of its own (the host part of the state is written at
// once). The buffer holds the state after llama_state_async_wait(ctx, dst) returned and must stay allocated until
// then. The copies read the sequence's cells: llama_state_async_wait_seq(ctx, seq_id) comes before anything writes
// them again. Without a device stream for it the copy is made synchronously, as without the flag.
#define LLAMA_STATE_SEQ_FLAGS_ASYNC 8
LLAMA_API void llama_state_async_wait(struct llama_context * ctx, const void * dst);
LLAMA_API void llama_state_async_wait_seq(struct llama_context * ctx, llama_seq_id seq_id);
LLAMA_API void llama_state_async_wait_all(struct llama_context * ctx);
LLAMA_API bool llama_state_async_pending_seq(struct llama_context * ctx, llama_seq_id seq_id);

// A sequence's state copied out beside the decodes. With LLAMA_STATE_SEQ_FLAGS_ASYNC, llama_state_seq_get_data_ext
// returns once the device has the copies queued on a stream of its own (the host part of the state is written at
// once). The buffer holds the state after llama_state_async_wait(ctx, dst) returned and must stay allocated until
// then. The copies read the sequence's cells: llama_state_async_wait_seq(ctx, seq_id) comes before anything writes
// them again. Without a device stream for it the copy is made synchronously, as without the flag.
#define LLAMA_STATE_SEQ_FLAGS_ASYNC 8
LLAMA_API void llama_state_async_wait(struct llama_context * ctx, const void * dst);
LLAMA_API void llama_state_async_wait_seq(struct llama_context * ctx, llama_seq_id seq_id);
LLAMA_API void llama_state_async_wait_all(struct llama_context * ctx);
LLAMA_API bool llama_state_async_pending_seq(struct llama_context * ctx, llama_seq_id seq_id);

// Decode chain: a decode queued before the host has read the tokens of the one before it.
// Turns the chain form on for the context's small decodes. False, and nothing changes, when the context cannot chain.
LLAMA_API bool llama_decode_chain_enable(struct llama_context * ctx);

// Arms the next llama_decode: row rows[k] takes on the device the id sampled for index idxs[k] of the last decode.
// It must repeat that layout, or llama_decode returns 3. False before llama_outputs_advance of an armed decode.
LLAMA_API bool llama_set_tokens_sampled(struct llama_context * ctx, const int32_t * rows, const int32_t * idxs, int32_t n);

// Waits for the outputs the accessors read: while an armed decode is queued, those of the decode before it.
LLAMA_API void llama_outputs_wait(struct llama_context * ctx);

// The accessors move on to the outputs of the queued armed decode.
LLAMA_API void llama_outputs_advance(struct llama_context * ctx);

// The last decode's outputs go to the host now, ahead of work another context queues on this one's stream (a drafter
// round over its cache): the accessors wait for them alone, llama_synchronize still for everything.
LLAMA_API void llama_outputs_fence(struct llama_context * ctx);

// the decode loop on the device: n_max greedy steps from token0 at pos0 into
// out, each step fed by the last; needs flash attention, n_rs_seq 0 and a
// backend sampler on sequence 0. Returns the count, or a negative on failure.
LLAMA_API int32_t llama_decode_loop(struct llama_context * ctx, llama_token token0, llama_pos pos0, int32_t n_max, llama_token stop, llama_token * out);

// whether the last llama_decode_loop ran as one graph on the device (false: the host stepped it)
LLAMA_API bool llama_decode_loop_on_device(const struct llama_context * ctx);

// Whether the model's DECODER_MTP graph supports chained drafting.
LLAMA_API bool llama_model_has_mtp_chain(const struct llama_model * model);

// Whether its chained round runs over a drafter cache of one stream per sequence (else the drafter's cache is one
// stream for every sequence).
LLAMA_API bool llama_model_mtp_round_streams(const struct llama_model * model);

// LLAMA_ROUND_TIMING=2: timeline marks over a round; the server flushes them (print: to stderr)
LLAMA_API void llama_round_mark(const char * what, int32_t tag);
LLAMA_API void llama_round_marks_flush(bool print);

// Whether this context runs attention through the flash-attention path. The
// chained MTP round is written against it and cannot be built without it.
LLAMA_API bool llama_context_flash_attn(const struct llama_context * ctx);

// Whether this context keeps every sequence in one KV stream. A chained MTP round over several sequences needs it.
LLAMA_API bool llama_context_kv_unified(const struct llama_context * ctx);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

// llama_memory_seq_cp_prefix with the row copies queued on the context's first backend: they run ahead of its next
// decode and the host does not wait for them
LLAMA_API bool llama_seq_cp_prefix_async(struct llama_context * ctx, llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos n);

//
// model/context data extraction
//

LLAMA_API int32_t llama_model_dflash_selector_top_k(const struct llama_model * model);

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);
