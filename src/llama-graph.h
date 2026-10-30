#pragma once

#include "llama-arch.h"
#include "llama-batch.h"
#include "llama-hparams.h"
#include "llama-adapter.h"
#include "llama-ext.h"

#include "ggml-backend.h"

#include <cstdint>
#include <cstdlib>
#include <array>
#include <vector>
#include <cstdio>
#include <cstring>
#include <memory>
#include <set>
#include <functional>
#include <map>

struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

struct llama_cparams;
struct llama_draft_head;
struct llama_layer;

struct llama_memory_context_i;

class llama_kv_cache_context;
class llama_kv_cache_dsa_context;
class llama_kv_cache_dsa_iswa_context;
class llama_kv_cache_msa_context;
class llama_kv_cache_dsv4_raw_context;
class llama_kv_cache_dsv4_context;
class llama_kv_cache_iswa_context;
class llama_memory_recurrent_context;
class llama_memory_hybrid_context;
class llama_memory_hybrid_iswa_context;

// certain models (typically multi-modal) can produce different types of graphs
enum llm_graph_type {
    LLM_GRAPH_TYPE_DEFAULT,
    LLM_GRAPH_TYPE_ENCODER,
    LLM_GRAPH_TYPE_DECODER,
    LLM_GRAPH_TYPE_DECODER_MTP,
};

enum llm_fused_op {
    LLM_FUSED_OP_FLASH_ATTN,
    LLM_FUSED_OP_GDN_AR,
    LLM_FUSED_OP_GDN_CH,
    LLM_FUSED_OP_LIGHTNING_INDEXER,
    LLM_FUSED_OP_DSV4_HC_PRE,
    LLM_FUSED_OP_DSV4_HC_COMB,
    LLM_FUSED_OP_DSV4_HC_POST,
};

enum llm_ffn_op_type : int {
    LLM_FFN_NONE = 0,           // sentinel: unset; archs must assign before use
    LLM_FFN_SILU,
    LLM_FFN_GELU,
    LLM_FFN_RELU,
    LLM_FFN_RELU_SQR,
    LLM_FFN_SWIGLU,
    LLM_FFN_GEGLU,
    LLM_FFN_REGLU,
    LLM_FFN_SWIGLU_OAI_MOE,
    LLM_FFN_SITU,           // kimi-k3
};

enum llm_ffn_gate_type {
    LLM_FFN_SEQ,
    LLM_FFN_PAR, // ffn_gate is parallel to ffn_up
};

enum llm_norm_type {
    LLM_NORM,
    LLM_NORM_RMS,
    LLM_NORM_GROUP,
};

// TODO: tmp - need something better to pass the data from the encoder to the decoder
struct llama_cross {
    // the output embeddings from the encoder as a ggml tensor
    // TODO: this needs more work to be correct, for now copy the embeddings data to host memory
    //       ref: https://github.com/ggml-org/llama.cpp/pull/11213#discussion_r1969892524
    //ggml_tensor * t_embd = nullptr;

    int64_t n_embd = 0;
    int64_t n_enc  = 0;

    // embeddings data copied to host memory (tmp)
    std::vector<float> v_embd;

    // needed to construct the cross-attention mask in the decoder
    std::vector<std::set<llama_seq_id>> seq_ids_enc;
};

struct llm_graph_params;
struct llama_tokens_dev_state;
struct llama_embd_dev_state;
struct llama_decode_chain_state;

//
// llm_graph_input
//

class llm_graph_input_i {
public:
    llm_graph_input_i() {
        const char * LLAMA_GRAPH_INPUT_DEBUG = getenv("LLAMA_GRAPH_INPUT_DEBUG");
        debug = LLAMA_GRAPH_INPUT_DEBUG ? atoi(LLAMA_GRAPH_INPUT_DEBUG) : 0;
    }

    virtual ~llm_graph_input_i() = default;

    virtual void set_input(const llama_ubatch * ubatch) = 0;

    // return true if the resulting input tensors using the provided graph parameters would be
    //   the same as the previous input tensors that we have currently stored in the object
    virtual bool can_reuse(const llm_graph_params & params) {
        // returning false here by default will prevent from reusing the graph if the check
        //   for the input type has not been implemented yet
        GGML_UNUSED(params);
        return false;
    }

    // an input whose host data depends on the batch's token ids; a decode that takes its ids on the device
    // cannot upload them to it
    virtual bool reads_token_ids() const { return true; }
protected:
    // env: LLAMA_GRAPH_INPUT_DEBUG
    int debug = 0;
};

using llm_graph_input_ptr = std::unique_ptr<llm_graph_input_i>;

class llm_graph_input_embd : public llm_graph_input_i {
public:
    llm_graph_input_embd(int64_t n_embd) : n_embd(n_embd) {}
    virtual ~llm_graph_input_embd() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * tokens = nullptr; // I32 [n_batch]
    ggml_tensor * embd   = nullptr; // F32 [n_embd, n_batch]

    bool device = false; // a loop body reads its token on the device

    llama_tokens_dev_state * tokens_dev = nullptr;
    bool dev_tokens = false; // tokens is a view of tokens_dev->tok_dev
    ggml_backend_sched_t sched = nullptr;
    llama_embd_dev_state * embd_dev = nullptr; // rows of embd copied on the device from another context, see set_input
    ggml_tensor * idx     = nullptr; // I32 [S]: a chained draft's anchors as rows of embd_dev->stage; embd is then not read
    int32_t       chain_k = 0;       // rows per sequence of such a draft

    // chain form: I32 [2*n_tokens], the host ids and then the row map into chain->tok_io
    const llama_decode_chain_state * chain = nullptr;
    ggml_tensor * tok_in = nullptr;

    const int64_t n_embd = 0;
};

// similar to llm_graph_input_embd but with an additional hidden state input
class llm_graph_input_embd_h : public llm_graph_input_i {
public:
    llm_graph_input_embd_h(int64_t n_embd) : n_embd(n_embd) {}
    virtual ~llm_graph_input_embd_h() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * tokens = nullptr; // I32 [n_batch]
    ggml_tensor * embd   = nullptr; // F32 [n_embd, n_batch]
    ggml_tensor * h      = nullptr; // F32 [n_embd, n_batch]

    const int64_t n_embd = 0;
};

// The decode loop on the device. The host reserves a window of cache cells
// and uploads, once, the position, cache row and mask of every step; the
// body picks its step's row with a counter it advances itself, feeds the
// token it sampled back into its own input, and appends it to a ring.
struct llama_loop_state {
    ggml_tensor * tok      = nullptr; // I32 [n_tok]         the tokens this step reads
    ggml_tensor * counter  = nullptr; // F32 [1]             the window row of the step's first token, advanced by the body
    ggml_tensor * ring     = nullptr; // F32 [n_ring]        the ids sampled so far
    ggml_tensor * pos_win  = nullptr; // I32 [n_pos, n_rows] positions of every window row
    ggml_tensor * kidx_win = nullptr; // I32 [1, n_rows]     cache row every window row writes
    ggml_tensor * vidx_win = nullptr; // I32 [1, n_rows]
    ggml_tensor * mask_win = nullptr; // F16 [n_kv, n_rows]  mask row of every window row
    int32_t       n_max    = 0;       // steps
    int32_t       n_tok    = 0;       // tokens a step reads
    int32_t       n_rows   = 0;       // window rows
};

// the rows of seq at pos0 + 1 .. pos0 + n
struct llama_tokens_dev_span {
    llama_seq_id seq  = -1;
    llama_pos    pos0 = 0;
    int32_t      n    = 0;
    int32_t      col  = 0;  // the sequence's element in each of the drafter's id tensors
};

// Batch rows of the next decode that another context's chained MTP draft fills on the device: the verify is
// queued behind the draft graph instead of waiting for its ids to reach the host. Armed for one decode.
struct llama_embd_dev_state {
    llama_context *                        src = nullptr;
    std::vector<std::array<int32_t, 2>>    rows;  // {sequence whose first row takes it, row of src's last hidden rows}
    // a chained draft gathers its anchors from here: the source's hidden rows come over as one block, the host's
    // rows go behind it (llm_graph_input_embd::idx); the drafter allocates it on first use
    ggml_tensor *                          stage     = nullptr; // F32 [n_embd, n_src_max + n_seq_max]
    int32_t                                n_src_max = 0;
};

struct llama_tokens_dev_state {
    llama_context        * src       = nullptr; // the drafter whose last chained ids fill the rows
    ggml_backend_sched_t   sched_dst = nullptr;
    std::vector<llama_tokens_dev_span> spans;   // empty: the host uploads every row
    int32_t                n_batch   = 0;       // tokens of the decode in flight

    // a decode of up to n_max tokens in one ubatch reads the drafted ids from tok_dev, so a copy from the drafter does
    // not synchronize; a decode of several ubatches uploads them from the host
    int32_t                n_max     = 16;
    bool                   enabled   = false;
    ggml_context         * ctx_dev   = nullptr;
    ggml_backend_buffer_t  buf_dev   = nullptr;
    ggml_tensor          * tok_dev   = nullptr; // I32 [n_max], in the token embedding's buffer type

    void disarm() { src = nullptr; sched_dst = nullptr; spans.clear(); n_batch = 0; }

    bool use_dev(const llama_ubatch & ubatch) const {
        return enabled && ubatch.token && (int32_t) ubatch.n_tokens <= n_max && (int32_t) ubatch.n_tokens == n_batch;
    }

    llama_tokens_dev_state() = default;
    llama_tokens_dev_state(const llama_tokens_dev_state &) = delete;
    llama_tokens_dev_state & operator=(const llama_tokens_dev_state &) = delete;
    ~llama_tokens_dev_state() {
        if (buf_dev) { ggml_backend_buffer_free(buf_dev); }
        if (ctx_dev) { ggml_free(ctx_dev); }
    }
};

// the inputs of a round over S sequences
struct llama_mtp_round_stage {
    ggml_context          * ctx_dev    = nullptr;
    ggml_backend_buffer_t   buf_dev    = nullptr;
    ggml_tensor           * c_ids_dev  = nullptr; // I32 [n_verify*S]
    ggml_tensor           * h_tgt_dev  = nullptr; // F32 [n_embd, n_verify*S]
    ggml_tensor           * h_pend_dev = nullptr; // F32 [n_embd, S]
};

// tok_io rows [0, n_cm): the ids the last small decode sampled, by output row; rows [n_cm, 2*n_cm): the host's ids.
// A row map uploaded with the host ids picks one of the two for every batch row.
struct llama_decode_chain_state {
    int32_t                n_cm   = 0;       // 0: no decode takes the chain form
    ggml_context         * ctx_io = nullptr;
    ggml_backend_buffer_t  buf_io = nullptr;
    ggml_tensor          * tok_io = nullptr; // I32 [2*n_cm], on the output device

    // for the decode being built: the output row of the last decode each sequence takes its token from, or -1
    std::vector<int32_t>   seq_row;

    llama_decode_chain_state() = default;
    llama_decode_chain_state(const llama_decode_chain_state &) = delete;
    llama_decode_chain_state & operator=(const llama_decode_chain_state &) = delete;
    ~llama_decode_chain_state() {
        if (buf_io) { ggml_backend_buffer_free(buf_io); }
        if (ctx_io) { ggml_free(ctx_io); }
    }
};

// the candidates a drafter that samples its drafts keeps of each (speculative sampling, see build_sampling)
#define LLAMA_MTP_DRAFT_TOP_K 20

struct llama_mtp_round_state {
    llama_context * ctx_tgt   = nullptr;
    ggml_tensor   * hpend_dev = nullptr;  // F32 [n_embd, n_seq_max], by sequence id, written after every chained decode

    // The round's inputs live on the device for the life of the context. As
    // graph inputs they would sit in a host buffer, and a copy into a host
    // buffer is not one ggml can do asynchronously: it synchronizes the source
    // backend first, so every round waited for the target's whole forward pass
    // before the draft graph could even be queued. On the device the same copy
    // is ordered by an event and the host waits for nothing.
    std::vector<llama_mtp_round_stage> stages; // indexed by S - 1, filled on the first round over S sequences
    int64_t n_embd_dev   = 0;
    int64_t n_verify_dev = 0;

    // the next round's sequences; after a round each one takes the device route
    std::vector<llama_mtp_round_seq> seqs = { llama_mtp_round_seq() };

    // a drafter over the target's cache takes only the verify's samples and hidden rows (see
    // llama_set_mtp_dev_anchors): no pending row of its own
    bool anchors_only = false;

    void free_stages() {
        for (auto & st : stages) {
            if (st.buf_dev) { ggml_backend_buffer_free(st.buf_dev); }
            if (st.ctx_dev) { ggml_free(st.ctx_dev); }
        }
        stages.clear();
    }

    llama_mtp_round_state() = default;
    llama_mtp_round_state(const llama_mtp_round_state &) = delete;
    llama_mtp_round_state & operator=(const llama_mtp_round_state &) = delete;
    ~llama_mtp_round_state() { free_stages(); }
};

// A model joins the one-trip speculative round by implementing these three
// hooks over its own MTP block; llm_graph_context::build_mtp_round does the
// rest, and it is the same for every architecture.
class llm_graph_mtp_i {
public:
    virtual ~llm_graph_mtp_i() = default;

    // one pass of the MTP block over the rows of tok_embd/h, starting at row
    // `row` of the batch; it writes those K/V rows before attending. When
    // kv_only nothing reads the result, so only the cache is refreshed. When
    // mask is set it replaces the batch's own mask row.
    virtual ggml_tensor * mtp_step(
            ggml_tensor * tok_embd,
            ggml_tensor * h,
            ggml_tensor * pos,
                    int   row,
                   bool   kv_only,
            ggml_tensor * mask) = 0;

    // the embedding matrix a drafted token id goes through
    virtual ggml_tensor * mtp_embd_weights() const = 0;

    // the MTP head: hidden state to logits
    virtual ggml_tensor * mtp_head(ggml_tensor * h) = 0;
};

class llm_graph_input_mtp_round : public llm_graph_input_i {
public:
    llm_graph_input_mtp_round(llama_mtp_round_state * state, ggml_backend_sched_t sched, int64_t n_embd, int64_t n_verify, int64_t n_seq)
        : state(state), sched(sched), n_embd(n_embd), n_verify(n_verify), n_seq(n_seq) {}
    virtual ~llm_graph_input_mtp_round() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * c_ids  = nullptr; // I32 [n_verify*n_seq]
    ggml_tensor * h_tgt  = nullptr; // F32 [n_embd, n_verify*n_seq]
    ggml_tensor * h_pend = nullptr; // F32 [n_embd, n_seq]
    ggml_tensor * off    = nullptr; // F32 [n_seq]

    llama_mtp_round_state * state;
    ggml_backend_sched_t sched;
    const int64_t n_embd;
    const int64_t n_verify;
    const int64_t n_seq;
};

// a round over a cache of one stream per sequence: the batch holds each sequence's rows together, a step takes the
// replay's rows (or one chain row) of every sequence. The K/V cells in step order, and the replay rows' places in a
// mask over the view's streams
class llm_graph_input_mtp_rows : public llm_graph_input_i {
public:
    llm_graph_input_mtp_rows(const llama_kv_cache_context * mctx, int64_t n_replay) : mctx(mctx), n_replay(n_replay) {}
    virtual ~llm_graph_input_mtp_rows() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * k_idxs = nullptr; // I64 [n_tokens]
    ggml_tensor * v_idxs = nullptr; // I64 [n_tokens], null where the cache holds no separate V
    ggml_tensor * hide   = nullptr; // I64 [n_replay*n_seq]

    const llama_kv_cache_context * mctx;
    const int64_t n_replay;
};

// a round that samples its drafts: every sequence's temperature and top-p from the target's sampler, a uniform a step
class llm_graph_input_mtp_sample : public llm_graph_input_i {
public:
    llm_graph_input_mtp_sample(llama_mtp_round_state * state, int K, int V) : state(state), K(K), V(V) {}
    virtual ~llm_graph_input_mtp_sample() = default;

    bool reads_token_ids() const override { return false; }

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override { GGML_UNUSED(params); return true; }

    ggml_tensor * par = nullptr; // F32 [2, S]: 1/temperature (large for a greedy one), top-p
    ggml_tensor * u   = nullptr; // F32 [S, K]: step j's uniforms in row j

    llama_mtp_round_state * state;
    const int K;
    const int V;
};

class llm_graph_input_pos : public llm_graph_input_i {
public:
    llm_graph_input_pos(uint32_t n_pos_per_embd) : n_pos_per_embd(n_pos_per_embd) {}
    virtual ~llm_graph_input_pos() = default;

    bool reads_token_ids() const override { return false; }

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * pos = nullptr; // I32 [n_batch]

    bool device = false; // a loop body selects its position on the device

    const uint32_t n_pos_per_embd = 1;
};

// temperature tuning, used by llama4
class llm_graph_input_attn_temp : public llm_graph_input_i {
public:
    llm_graph_input_attn_temp(uint32_t n_attn_temp_floor_scale, float f_attn_temp_scale, float f_attn_temp_offset)
        : n_attn_temp_floor_scale(n_attn_temp_floor_scale), f_attn_temp_scale(f_attn_temp_scale), f_attn_temp_offset(f_attn_temp_offset) {}
    virtual ~llm_graph_input_attn_temp() = default;

    bool reads_token_ids() const override { return false; }

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * attn_scale = nullptr; // F32 [n_batch]

    const uint32_t n_attn_temp_floor_scale;
    const float    f_attn_temp_scale;
    const float    f_attn_temp_offset;
};

class llm_graph_input_pos_bucket : public llm_graph_input_i {
public:
    llm_graph_input_pos_bucket(const llama_hparams & hparams) : hparams(hparams) {}
    virtual ~llm_graph_input_pos_bucket() = default;

    bool reads_token_ids() const override { return false; }

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * pos_bucket = nullptr; // I32 [n_batch, n_batch]

    const llama_hparams hparams;
};

class llm_graph_input_pos_bucket_kv : public llm_graph_input_i {
public:
    llm_graph_input_pos_bucket_kv(
            const llama_hparams & hparams,
            const llama_kv_cache_context * mctx) : hparams(hparams), mctx(mctx) {}
    virtual ~llm_graph_input_pos_bucket_kv() = default;

    bool reads_token_ids() const override { return false; }

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * pos_bucket = nullptr; // I32 [n_kv, n_batch]

    const llama_hparams hparams;

    const llama_kv_cache_context * mctx;
};

class llm_graph_input_out_ids : public llm_graph_input_i {
public:
    llm_graph_input_out_ids(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            uint32_t n_outputs) : hparams(hparams), cparams(cparams), n_outputs(n_outputs) {}
    virtual ~llm_graph_input_out_ids() = default;

    bool reads_token_ids() const override { return false; }

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * out_ids; // I32 [n_outputs]

    const llama_hparams hparams;
    const llama_cparams cparams;

    const uint32_t n_outputs;
};

// a ubatch whose sequences skip idle streams: which sequence's queries each stream of the span takes (a stream
// between them takes the first sequence's, masked out), and which stream each sequence's output comes back from
class llm_graph_input_strm : public llm_graph_input_i {
public:
    llm_graph_input_strm(int64_t n_seqs, int64_t n_span, int64_t n_stream) : n_seqs(n_seqs), n_span(n_span), n_stream(n_stream) {}
    virtual ~llm_graph_input_strm() = default;

    bool reads_token_ids() const override { return false; }

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * strm_q = nullptr; // I32 [n_span]
    ggml_tensor * strm_o = nullptr; // I32 [n_seqs]

    const int64_t n_seqs;
    const int64_t n_span;
    const int64_t n_stream; // the cache's streams: a span of them all starts at stream 0

    // in a grouped forward, the remap of one group's rows: its ubatch is the group view's, not the graph's
    int group = -1;
    const llama_memory_context_i * gmctx = nullptr;
};

class llm_graph_input_mean : public llm_graph_input_i {
public:
    llm_graph_input_mean(const llama_cparams & cparams) : cparams(cparams) {}
    virtual ~llm_graph_input_mean() = default;

    bool reads_token_ids() const override { return false; }

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * mean; // F32 [n_batch, n_batch]

    const llama_cparams cparams;
};

class llm_graph_input_cls : public llm_graph_input_i {
public:
    llm_graph_input_cls(const llama_cparams & cparams, const llm_arch arch) : cparams(cparams), arch(arch) {}
    virtual ~llm_graph_input_cls() = default;

    bool reads_token_ids() const override { return false; }

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * cls; // I32 [n_batch]

    const llama_cparams cparams;
    const llm_arch arch;
};

class llm_graph_input_rs : public llm_graph_input_i {
public:
    llm_graph_input_rs(const llama_memory_recurrent_context * mctx) : mctx(mctx) {}
    virtual ~llm_graph_input_rs() = default;

    bool reads_token_ids() const override { return false; }

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * s_copy;  // I32 [n_rs]

    // views of s_copy, computed once per graph
    // and shared across layers which use build_rs
    ggml_tensor * s_copy_main;   // I32 [n_seqs]
    ggml_tensor * s_copy_extra;  // I32 [n_rs - n_seqs]

    const llama_memory_recurrent_context * mctx;

    // used in view offsets, need to match for valid graph reuse
    uint32_t head;
    int32_t rs_z;
};

class llm_graph_input_cross_embd : public llm_graph_input_i {
public:
    llm_graph_input_cross_embd(
            const llama_cross * cross) : cross(cross) {}
    virtual ~llm_graph_input_cross_embd() = default;

    bool reads_token_ids() const override { return false; }

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * cross_embd; // F32 [n_embd, n_outputs_enc]

    const llama_cross * cross;
};

class llm_graph_input_attn_no_cache : public llm_graph_input_i {
public:
    llm_graph_input_attn_no_cache(const llama_hparams & hparams, const llama_cparams & cparams) :
        hparams(hparams),
        cparams(cparams) {
    }
    ~llm_graph_input_attn_no_cache() = default;

    bool reads_token_ids() const override { return false; }

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * get_kq_mask()     const { return self_kq_mask_cnv; }
    ggml_tensor * get_kq_mask_swa() const { return self_kq_mask_swa_cnv; }

    // n_tokens == n_batch
    ggml_tensor * self_kq_mask         = nullptr; // F32/F16 [n_tokens, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_cnv     = nullptr; //         [n_tokens, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_swa     = nullptr; // F32/F16 [n_tokens, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_swa_cnv = nullptr; //         [n_tokens, n_batch/n_stream, 1, n_stream]

    const llama_hparams hparams;
    const llama_cparams cparams;
};

class llm_graph_input_attn_kv : public llm_graph_input_i {
public:
    llm_graph_input_attn_kv(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_kv() = default;

    bool reads_token_ids() const override { return false; }

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * get_k_idxs() const { return self_k_idxs; }
    ggml_tensor * get_v_idxs() const { return self_v_idxs; }

    ggml_tensor * get_kq_mask() const { return self_kq_mask_cnv; }

    ggml_tensor * self_k_idxs = nullptr; // I64 [n_batch]
    ggml_tensor * self_v_idxs = nullptr; // I64 [n_batch] or [n_batch*n_embd_v_gqa]

    ggml_tensor * self_kq_mask     = nullptr; // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_cnv = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]

    // note: assumes v_rot^2 == I
    ggml_tensor * self_k_rot = nullptr;
    ggml_tensor * self_v_rot = nullptr;

    // note: these have to be copies because in order to be able to reuse a graph, its inputs
    //       need to carry these parameters with them. otherwise, they can point to freed
    //       llm_graph_params from a previous batch, causing stack-use-after-return
    const llama_hparams hparams;
    const llama_cparams cparams;

    bool device = false; // a loop body selects its rows and mask on the device

    const llama_kv_cache_context * mctx;
};

// V-less input for the KV cache
// ref: https://github.com/ggml-org/llama.cpp/pull/19067
class llm_graph_input_attn_k : public llm_graph_input_i {
public:
    llm_graph_input_attn_k(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_k() = default;

    bool reads_token_ids() const override { return false; }

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    // like can_reuse, but does not re-bind mctx
    bool can_reuse_impl(const llm_graph_params & params);

    ggml_tensor * get_k_idxs() const { return self_k_idxs; }

    ggml_tensor * get_kq_mask() const { return self_kq_mask_cnv; }

    ggml_tensor * self_k_idxs = nullptr; // I64 [n_batch]

    ggml_tensor * self_kq_mask     = nullptr; // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_cnv = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]

    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_context * mctx;
};

class llm_graph_input_attn_k_dsa : public llm_graph_input_i {
public:
    llm_graph_input_attn_k_dsa(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_dsa_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_k_dsa() = default;

    bool reads_token_ids() const override { return false; }

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    // like can_reuse, but does not re-bind mctx
    bool can_reuse_impl(const llm_graph_params & params);

    ggml_tensor * get_k_idxs_mla() const { return self_k_idxs_mla; }
    ggml_tensor * get_k_idxs_lid() const { return self_k_idxs_lid; }

    ggml_tensor * get_kq_mask_mla() const { return self_kq_mask_mla_cnv; }
    ggml_tensor * get_kq_mask_lid() const { return self_kq_mask_lid; }

    ggml_tensor * self_k_idxs_mla = nullptr; // I64 [n_batch]
    ggml_tensor * self_k_idxs_lid = nullptr; // I64 [n_batch]

    ggml_tensor * self_kq_mask_mla     = nullptr; // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_mla_cnv = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_lid     = nullptr; // F32     [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_lid_cnv = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]

    ggml_tensor * self_k_rot_lid = nullptr;

    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_dsa_context * mctx;
};

// DSA input (full-attention layers + indexer) with K-only input for the SWA layers
class llm_graph_input_attn_k_dsa_iswa : public llm_graph_input_i {
public:
    llm_graph_input_attn_k_dsa_iswa(
            std::unique_ptr<llm_graph_input_attn_k_dsa> inp_dsa,
            std::unique_ptr<llm_graph_input_attn_k>     inp_swa,
            const llama_kv_cache_dsa_iswa_context *     mctx) :
        inp_dsa(std::move(inp_dsa)),
        inp_swa(std::move(inp_swa)),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_k_dsa_iswa() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    llm_graph_input_attn_k_dsa * get_dsa() const { return inp_dsa.get(); }
    llm_graph_input_attn_k     * get_swa() const { return inp_swa.get(); }

    std::unique_ptr<llm_graph_input_attn_k_dsa> inp_dsa;
    std::unique_ptr<llm_graph_input_attn_k>     inp_swa;

    const llama_kv_cache_dsa_iswa_context * mctx;
};

// standard K/V attention input against the base cache, plus destination indices for the indexer key cache
class llm_graph_input_attn_kv_msa : public llm_graph_input_attn_kv {
public:
    llm_graph_input_attn_kv_msa(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_msa_context * mctx);
    ~llm_graph_input_attn_kv_msa() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * get_k_idxs_idx() const { return self_k_idxs_idx; }

    ggml_tensor * self_k_idxs_idx = nullptr; // I64 [n_batch]

    const llama_kv_cache_msa_context * mctx_msa;
};

class llm_graph_input_attn_kv_iswa : public llm_graph_input_i {
public:
    llm_graph_input_attn_kv_iswa(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_iswa_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_kv_iswa() = default;

    bool reads_token_ids() const override { return false; }

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * get_k_idxs()     const { return self_k_idxs; }
    ggml_tensor * get_v_idxs()     const { return self_v_idxs; }
    ggml_tensor * get_k_idxs_swa() const { return self_k_idxs_swa; }
    ggml_tensor * get_v_idxs_swa() const { return self_v_idxs_swa; }

    ggml_tensor * get_kq_mask()     const { return self_kq_mask_cnv; }
    ggml_tensor * get_kq_mask_swa() const { return self_kq_mask_swa_cnv; }

    ggml_tensor * self_k_idxs     = nullptr; // I64 [n_batch]
    ggml_tensor * self_v_idxs     = nullptr; // I64 [n_batch] or [n_batch*n_embd_v_gqa]
    ggml_tensor * self_k_idxs_swa = nullptr; // I64 [n_batch]
    ggml_tensor * self_v_idxs_swa = nullptr; // I64 [n_batch] or [n_batch*n_embd_v_gqa]

    ggml_tensor * self_kq_mask         = nullptr; // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_cnv     = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_swa     = nullptr; // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_swa_cnv = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]

    ggml_tensor * self_k_rot = nullptr;
    ggml_tensor * self_v_rot = nullptr;

    ggml_tensor * self_k_rot_swa = nullptr;
    ggml_tensor * self_v_rot_swa = nullptr;

    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_iswa_context * mctx;

    // A grouped forward (a verify's rows and prompt chunks in one graph, llama_kv_cache_iswa_context::n_groups): the
    // parent holds no tensors and a child per group; a child's mctx is its group's view and its ubatch that view's.
    int group = -1;
    std::vector<llm_graph_input_attn_kv_iswa *> groups;

    // a group's streams, one per sequence, which the attention reads instead of the view's (I32 [n_seqs]); its masks
    // are then one per sequence too
    ggml_tensor * kv_strm = nullptr;

    // A grouped forward in vLLM's variable-length layout: one attention over every group's rows instead of one per
    // group. The indices and masks above are the groups' rows one after another, the masks as wide as the widest
    // group's view; the table is each sequence's first row, rows and stream (flash_attn_varlen_tile).
    ggml_tensor * varlen_tab = nullptr; // I32 [1 + 3*n_seq_max]
    int64_t n_kv_varlen     = 0;
    int64_t n_kv_varlen_swa = 0;

    void set_input_varlen(const llama_ubatch * ubatch);
};

class llm_graph_input_attn_k_iswa : public llm_graph_input_i {
public:
    llm_graph_input_attn_k_iswa(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_iswa_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_k_iswa() = default;

    bool reads_token_ids() const override { return false; }

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * get_k_idxs()     const { return self_k_idxs; }
    ggml_tensor * get_k_idxs_swa() const { return self_k_idxs_swa; }

    ggml_tensor * get_kq_mask()     const { return self_kq_mask_cnv; }
    ggml_tensor * get_kq_mask_swa() const { return self_kq_mask_swa_cnv; }

    ggml_tensor * self_k_idxs     = nullptr; // I64 [n_batch]
    ggml_tensor * self_k_idxs_swa = nullptr; // I64 [n_batch]

    ggml_tensor * self_kq_mask         = nullptr; // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_cnv     = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_swa     = nullptr; // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_swa_cnv = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]

    ggml_tensor * self_k_rot = nullptr;
    ggml_tensor * self_k_rot_swa = nullptr;

    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_iswa_context * mctx;
};

// What a round writes through, whichever cache flavour the architecture's MTP
// block attends against: the row indices of the K (and V) it writes and the
// mask it attends with. llm_graph_mtp_attn_of() reads it off an attention
// input, so a model does not care which flavour it has.
struct llm_graph_mtp_attn {
    const llama_kv_cache_context * mctx = nullptr;
    ggml_tensor * k_idxs  = nullptr;
    ggml_tensor * v_idxs  = nullptr; // null where the cache holds no separate V
    ggml_tensor * kq_mask = nullptr;
    ggml_tensor * k_rot   = nullptr; // the rotation a Hadamard-rotated cache applies
    ggml_tensor * kv_strm = nullptr; // each sequence's stream (llm_graph_input_attn_kv_iswa::kv_strm)
};

llm_graph_mtp_attn llm_graph_mtp_attn_of(llm_graph_input_attn_kv      * inp);
llm_graph_mtp_attn llm_graph_mtp_attn_of(llm_graph_input_attn_k       * inp);
llm_graph_mtp_attn llm_graph_mtp_attn_of(llm_graph_input_attn_kv_iswa * inp, bool swa);
llm_graph_mtp_attn llm_graph_mtp_attn_of(llm_graph_input_attn_k_iswa  * inp, bool swa);


// DSV4 raw graph inputs are SWA-only, but their mask may be stream-shaped
// so raw K can be concatenated with DSV4 compressed K in one attention op.
class llm_graph_input_dsv4_raw {
public:
    llm_graph_input_dsv4_raw(
            const llama_cparams & cparams,
            const llama_kv_cache_dsv4_raw_context * mctx) :
        cparams(cparams),
        mctx(mctx) {
    }

    void set_input(const llama_ubatch * ubatch);

    ggml_tensor * get_k_idxs() const { return self_k_idxs; }
    ggml_tensor * get_kq_mask() const { return self_kq_mask_cnv; }

    ggml_tensor * self_k_idxs = nullptr; // I64 [n_batch]

    ggml_tensor * self_kq_mask     = nullptr; // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_cnv = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]

    ggml_tensor * self_k_rot = nullptr;

    const llama_cparams cparams;

    const llama_kv_cache_dsv4_raw_context * mctx;
};

class llm_graph_input_dsv4 : public llm_graph_input_i {
public:
    struct comp_input {
        ggml_tensor * state_pos        = nullptr; // I32 [n_state]
        ggml_tensor * state_persist_src_idxs = nullptr; // I32 [n_state_persist]
        ggml_tensor * state_persist_dst_idxs = nullptr; // I32 [n_state_persist]
        ggml_tensor * state_restore_src_idxs = nullptr; // I32 [n_state_restore]
        ggml_tensor * state_restore_dst_idxs = nullptr; // I32 [n_state_restore]
        ggml_tensor * state_snapshot_src_idxs = nullptr; // I32 [n_state_snapshot]
        ggml_tensor * state_snapshot_dst_idxs = nullptr; // I32 [n_state_snapshot]
        ggml_tensor * state_read_idxs  = nullptr; // I32 [ratio*n_state_write]
        ggml_tensor * state_write_idxs = nullptr; // I64 [n_state_write]
        ggml_tensor * state_write_pos  = nullptr; // I32 [n_state_write]

        ggml_tensor * kq_mask    = nullptr; // F32 [n_kv, n_batch/n_stream, 1, n_stream]

        ggml_tensor * k_rot      = nullptr;
    };

    llm_graph_input_dsv4(
            const llama_cparams & cparams,
            std::unique_ptr<llm_graph_input_dsv4_raw> inp_raw,
            const llama_kv_cache_dsv4_context * mctx) :
        inp_raw(std::move(inp_raw)),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_dsv4() = default;

    bool reads_token_ids() const override { return false; }

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    llm_graph_input_dsv4_raw * get_raw() const { return inp_raw.get(); }
    const comp_input & get_csa() const { return inp_csa; }
    const comp_input & get_hca() const { return inp_hca; }
    const comp_input & get_lid() const { return inp_lid; }

    std::unique_ptr<llm_graph_input_dsv4_raw> inp_raw;

    comp_input inp_csa;
    comp_input inp_hca;
    comp_input inp_lid;

    const llama_cparams cparams;

    const llama_kv_cache_dsv4_context * mctx;
};

class llm_graph_input_attn_cross : public llm_graph_input_i {
public:
    llm_graph_input_attn_cross(const llama_cross * cross) : cross(cross) {}
    ~llm_graph_input_attn_cross() = default;

    bool reads_token_ids() const override { return false; }

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * get_kq_mask_cross() const { return cross_kq_mask_cnv; }

    ggml_tensor * cross_kq_mask     = nullptr; // F32/F16 [n_outputs_enc, n_batch, 1, 1]
    ggml_tensor * cross_kq_mask_cnv = nullptr; // F32/F16 [n_outputs_enc, n_batch, 1, 1]

    const llama_cross * cross = nullptr;
};

class llm_graph_input_mem_hybrid : public llm_graph_input_i {
public:
    llm_graph_input_mem_hybrid(
            const llama_cparams & cparams,
            std::unique_ptr<llm_graph_input_attn_kv> inp_attn,
            std::unique_ptr<llm_graph_input_rs>      inp_rs,
            const llama_memory_hybrid_context *      mctx) :
        inp_attn(std::move(inp_attn)),
        inp_rs(std::move(inp_rs)),
        cparams(cparams),
        mctx(mctx) { }
    virtual ~llm_graph_input_mem_hybrid() = default;

    bool reads_token_ids() const override { return false; }

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    std::unique_ptr<llm_graph_input_attn_kv> inp_attn;
    std::unique_ptr<llm_graph_input_rs>      inp_rs;

    llm_graph_input_attn_kv * get_attn() const { return inp_attn.get(); }
    llm_graph_input_rs      * get_recr() const { return inp_rs.get(); }

    const llama_cparams cparams;

    const llama_memory_hybrid_context * mctx;
};

class llm_graph_input_mem_hybrid_k : public llm_graph_input_i {
public:
    llm_graph_input_mem_hybrid_k(
            const llama_cparams & cparams,
            std::unique_ptr<llm_graph_input_attn_k> inp_attn,
            std::unique_ptr<llm_graph_input_rs>      inp_rs,
            const llama_memory_hybrid_context *      mctx) :
        inp_attn(std::move(inp_attn)),
        inp_rs(std::move(inp_rs)),
        cparams(cparams),
        mctx(mctx) { }
    virtual ~llm_graph_input_mem_hybrid_k() = default;

    bool reads_token_ids() const override { return false; }

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    std::unique_ptr<llm_graph_input_attn_k> inp_attn;
    std::unique_ptr<llm_graph_input_rs>      inp_rs;

    llm_graph_input_attn_k * get_attn() const { return inp_attn.get(); }
    llm_graph_input_rs      * get_recr() const { return inp_rs.get(); }

    const llama_cparams cparams;

    const llama_memory_hybrid_context * mctx;
};

class llm_graph_input_mem_hybrid_iswa : public llm_graph_input_i {
public:
    llm_graph_input_mem_hybrid_iswa(
            const llama_cparams & cparams,
            std::unique_ptr<llm_graph_input_attn_kv_iswa> inp_attn,
            std::unique_ptr<llm_graph_input_rs>          inp_rs,
            const llama_memory_hybrid_iswa_context *     mctx) :
        inp_attn(std::move(inp_attn)),
        inp_rs(std::move(inp_rs)),
        cparams(cparams),
        mctx(mctx) { }
    virtual ~llm_graph_input_mem_hybrid_iswa() = default;

    bool reads_token_ids() const override { return false; }

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    std::unique_ptr<llm_graph_input_attn_kv_iswa> inp_attn;
    std::unique_ptr<llm_graph_input_rs>          inp_rs;

    llm_graph_input_attn_kv_iswa * get_attn() const { return inp_attn.get(); }
    llm_graph_input_rs           * get_recr() const { return inp_rs.get(); }

    const llama_cparams cparams;

    const llama_memory_hybrid_iswa_context * mctx;
};

class llm_graph_input_sampling : public llm_graph_input_i {
public:
    llm_graph_input_sampling(std::map<llama_seq_id, llama_sampler *> samplers) :
        samplers(std::move(samplers)) { }
    virtual ~llm_graph_input_sampling() = default;

    bool reads_token_ids() const override { return false; }

    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    std::map<llama_seq_id, llama_sampler *> samplers;

    // built as one argmax over every output row: reusable by any ubatch whose sampling is the same
    bool greedy = false;

    // built in the batched form (llm_graph_params::samplers_batch): its inputs, filled from the step's samplers
    uint64_t      batch_key = 0;
    int32_t       batch_lw  = 0; // penalty window entries a row holds
    int32_t       batch_kp  = 0; // ids a row takes in before it within the batch
    int32_t       batch_nb  = 0; // logit bias entries a row holds
    ggml_tensor * inp_par   = nullptr; // [12, rows]: top-p, log min-p, temperature, uniform, 1/repeat - 1, repeat - 1, freq, presence, temperature 0, drafted, second uniform, draft distribution
    ggml_tensor * inp_hist  = nullptr; // [lw, rows]: the penalty window's ids (-1: none)
    ggml_tensor * inp_ppos  = nullptr; // [kp*rows]: the batch positions of the ids a row takes in before it
    ggml_tensor * inp_pmask = nullptr; // [kp, rows]: 1 where inp_ppos holds one
    ggml_tensor * inp_btok  = nullptr; // [nb, rows]: logit bias ids (-1: none)
    ggml_tensor * inp_bval  = nullptr; // [nb, rows]: their biases
    ggml_tensor * inp_dpos  = nullptr; // [rows]: the batch position of the draft a row checks (its own without one)
    ggml_tensor * inp_qstage = nullptr; // [2*LLAMA_MTP_DRAFT_TOP_K, rows]: the drafter's last round's distributions
    ggml_tensor * inp_qidx   = nullptr; // [rows]: the stage row of the draft a row checks

    llama_tokens_dev_state * tokens_dev = nullptr; // the drafter's round whose distributions fill inp_qstage
    ggml_backend_sched_t     sched      = nullptr;
};

// the batched form's shape key when every output row's sampler takes it, else 0
uint64_t llm_graph_samplers_batch_key(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers,
        int32_t * lw = nullptr, int32_t * kp = nullptr, int32_t * nb = nullptr, int32_t * k = nullptr, bool * rep = nullptr,
        bool * minp = nullptr, bool * t0 = nullptr, bool * sd = nullptr);

// the streams and sampling signature of the ubatch, see llm_graph_params
void llm_graph_params_fill(llm_graph_params & params);

// see llm_graph_input_attn_kv_iswa::kv_strm
bool llm_graph_attn_streams_in(const llama_cparams & cparams);

// see llm_graph_input_attn_kv_iswa::varlen_tab; the widest group's cache view sizes in n_kv and n_kv_swa
bool llm_graph_attn_varlen(const llama_cparams & cparams, const llama_hparams & hparams,
        const llama_kv_cache_iswa_context * mctx, int64_t * n_kv = nullptr, int64_t * n_kv_swa = nullptr);

// every output row has a sampler whose backend result is the argmax of its logits
bool llm_graph_samplers_greedy(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers);

// greedy rows whose chains all pick on the backend: the host reads none of their logits (llama_context::needs_raw_logits)
bool llm_graph_logits_internal(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers);

//
// llm_graph_result
//

// these objects deliver the result from the graph build process back to the llama_context
// note that the input tensors created for the graph are referenced here - the goal is to be able to populate their
//   specific data, by calling the set_inputs() method
// along with the input tensors, the object also provides commonly used outputs tensors, such as logits, embeddings, etc.
//   these are used by the llama_context to extact the relevant data, based on the compute parameters

// callback that allows us to apply custom logic to each tensor (e.g. ggml-alloc, offloading, etc.)
using llm_graph_cb = std::function<void(const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il)>;

class llm_graph_result;

struct llm_graph_params {
    llm_arch arch = LLM_ARCH_UNKNOWN;

    llama_hparams hparams;
    llama_cparams cparams;

    llama_ubatch ubatch; // note: intentionally make a copy

    llm_graph_type gtype;

    ggml_backend_sched_t sched;
    ggml_backend_t backend_cpu;

    const llama_adapter_cvec     * cvec;
    const llama_adapter_loras    * loras;
    const llama_memory_context_i * mctx;
    const llama_cross            * cross;

    llama_mtp_round_state  * mtp_round;
    llama_tokens_dev_state * tokens_dev;
    llama_embd_dev_state   * embd_dev;
    const llama_loop_state       * loop;
    const llama_decode_chain_state * chain;

    const llama_draft_head * draft_head; // the reduced head a drafter drafts through, see llama-draft-vocab.h

    std::map<llama_seq_id, llama_sampler *> samplers;

    static bool samplers_equal(
          const std::map<llama_seq_id, llama_sampler *> & lhs,
          const std::map<llama_seq_id, llama_sampler *> & rhs) {
        if (lhs.size() != rhs.size()) {
            return false;
        }
        for (const auto & [seq_id, sampler] : lhs) {
            auto it = rhs.find(seq_id);
            if (it == rhs.end() || it->second != sampler) {
                return false;
            }
        }
        return true;
    }

    uint32_t n_outputs;

    llm_graph_cb cb;

    llm_graph_result * res;

    // The cache streams the ubatch's views cover (first stream, count) and whether its sampling is one argmax over
    // every output row: two ubatches of one shape with these equal build the same graph, whatever their sequence
    // ids and sampler objects, so a step over other slots replays the graph instead of building it again.
    // Filled by llm_graph_params_fill; strm_n is 0 for memory without streams, which keeps the sequence id check, and
    // -1 where the attention reads its streams from an input (any sequences of the same count replay the graph).
    int32_t strm_s0 = -1;
    int32_t strm_n  = 0;
    bool    samplers_greedy = false;
    // greedy, and no row's logits go to the host (llm_graph_logits_internal): the logits are no graph output
    bool    logits_internal = false;
    // every row's sampler runs in the batched form (llm_graph_samplers_batch_key): the form's shape, which with
    // its parameters in inputs makes the graph the same whatever the sampler objects; 0 otherwise
    uint64_t samplers_batch = 0;

    // a grouped forward: each group's tokens, sequences and stream span (empty otherwise)
    std::vector<std::array<int32_t, 4>> groups_sig;

    // the sizes of the cache views the attention reads (each group's, for a grouped forward): a multi-sequence
    // graph's arena key (llama_context::get_gf_res_prev), so a round keeps one graph per size it has seen
    uint64_t n_kv_sig = 0;

    // return true if the "other" params would result in a graph with the same topology as with the current params
    //   having the same topology allows us to reuse the graph in some cases
    // why the last allow_reuse said no (LLAMA_ROUND_TIMING)
    static const char *& reuse_reason() { static const char * r = ""; return r; }

    bool allow_reuse(const llm_graph_params & other) const {
        reuse_reason() = "";

        if (groups_sig != other.groups_sig) {
            reuse_reason() = "groups";
            return false;
        }
        // first check the ubatch
        bool can_reuse_ubatch =
            ubatch.equal_seqs() == other.ubatch.equal_seqs() &&
            ubatch.n_tokens     == other.ubatch.n_tokens &&
            ubatch.n_seq_tokens == other.ubatch.n_seq_tokens &&
            ubatch.n_seqs       == other.ubatch.n_seqs &&
            ubatch.n_seqs_unq   == other.ubatch.n_seqs_unq &&
            (
                (!ubatch.token && !other.ubatch.token) ||
                (!ubatch.embd  && !other.ubatch.embd)  ||
                (ubatch.token && other.ubatch.token && ubatch.embd && other.ubatch.embd)
            );

        // when we split the batch using "equal_seqs" we have to verify that the participating sequences are the same
        //   the reason is because the set of attention streams would be different for different sequences
        if (can_reuse_ubatch && ubatch.equal_seqs()) {
            if (strm_n < 0 && other.strm_n < 0) {
                // the streams are inputs
            } else if (strm_n > 0 && other.strm_n > 0) {
                can_reuse_ubatch = strm_s0 == other.strm_s0 && strm_n == other.strm_n;
            } else if (!ubatch.data) {
                // if the old ubatch does not own it's data, then we cannot guarantee that it is still alive, and
                //   therefore we cannot perform the sequence id check. normally should never happen
                can_reuse_ubatch = false;
            } else {
                for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                    can_reuse_ubatch &= ubatch.seq_id_unq[s] == other.ubatch.seq_id_unq[s];
                }
            }
        }

        if (!can_reuse_ubatch) {
            static char buf[256];
            snprintf(buf, sizeof(buf), "ubatch: old T %u st %u seqs %u unq %u eq %d tok %d embd %d span %d+%d | new T %u st %u seqs %u unq %u eq %d tok %d embd %d span %d+%d",
                    ubatch.n_tokens, ubatch.n_seq_tokens, ubatch.n_seqs, ubatch.n_seqs_unq, (int) ubatch.equal_seqs(), ubatch.token != nullptr, ubatch.embd != nullptr, strm_s0, strm_n,
                    other.ubatch.n_tokens, other.ubatch.n_seq_tokens, other.ubatch.n_seqs, other.ubatch.n_seqs_unq, (int) other.ubatch.equal_seqs(), other.ubatch.token != nullptr, other.ubatch.embd != nullptr, other.strm_s0, other.strm_n);
            reuse_reason() = buf;
            return false;
        }

        if (n_outputs != other.n_outputs) {
            reuse_reason() = "n_outputs";
            return false;
        }

        // one argmax over the rows reads no sampler, so the samplers need not be the same objects; a graph
        // without output rows samples nothing at all
        const bool greedy = (samplers_greedy && other.samplers_greedy) || (n_outputs == 0 && other.n_outputs == 0) ||
                            (samplers_batch != 0 && samplers_batch == other.samplers_batch);

        if (!greedy && !samplers_equal(samplers, other.samplers)) {
            reuse_reason() = samplers_greedy == other.samplers_greedy ? "samplers differ (not greedy)" : "greedy flag differs";
            return false;
        }

        if (logits_internal != other.logits_internal) {
            reuse_reason() = "logits output flag differs";
            return false;
        }

        if (samplers.size() > 0 || other.samplers.size() > 0) {
            if (!ubatch.data || !other.ubatch.data) {
                reuse_reason() = "ubatch data not owned";
                return false;
            }

            // check that the outputs are the same for all samplers. The output rows' places are input data
            // (out_ids); one argmax over the gathered rows reads no sampler, so for it only their count matters
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                if (!greedy && ubatch.output[i] != other.ubatch.output[i]) {
                    reuse_reason() = "output rows differ";
                    return false;
                }
                if (!greedy && ubatch.seq_id[i][0] != other.ubatch.seq_id[i][0]) {
                    reuse_reason() = "row sequences differ (not greedy)";
                    return false;
                }
            }
        }

        // TODO: https://github.com/ggml-org/llama.cpp/pull/24340#discussion_r3448035248
        if (cparams.nextn_layer_offset != other.cparams.nextn_layer_offset) {
            return false;
        }

        if (cparams.mtp_chain != other.cparams.mtp_chain || draft_head != other.draft_head) {
            reuse_reason() = "mtp chain or draft head";
            return false;
        }

        if (cparams.embeddings              != other.cparams.embeddings)              { reuse_reason() = "embeddings";              return false; }
        if (cparams.embeddings_nextn        != other.cparams.embeddings_nextn)        { reuse_reason() = "embeddings_nextn";        return false; }
        if (cparams.embeddings_nextn_masked != other.cparams.embeddings_nextn_masked) { reuse_reason() = "embeddings_nextn_masked"; return false; }
        if (cparams.causal_attn             != other.cparams.causal_attn)             { reuse_reason() = "causal_attn";             return false; }
        if (arch  != other.arch)  { reuse_reason() = "arch";  return false; }
        if (gtype != other.gtype) { reuse_reason() = "gtype"; return false; }
        if (cvec  != other.cvec)  { reuse_reason() = "cvec";  return false; }
        if (loras != other.loras) { reuse_reason() = "loras"; return false; }
        if (cross != other.cross) { reuse_reason() = "cross"; return false; }

        return true;
    }
};

struct llm_graph_fused_node {
    llm_fused_op op;
    ggml_tensor * tensor;
    int il;
};

class llm_graph_result {
public:
    llm_graph_result(int64_t max_nodes);

    virtual ~llm_graph_result() = default;

    ggml_tensor * get_inp_tokens()  const { return t_inp_tokens; }
    ggml_tensor * get_logits()      const { return t_logits; }
    ggml_tensor * get_embd()        const { return t_embd; }
    ggml_tensor * get_embd_pooled() const { return t_embd_pooled; }
    ggml_tensor * get_h_nextn()     const { return t_h_nextn; }

    ggml_tensor * get_layer_inp(int il) const { return t_layer_inp[il]; }

    // the I32 id sampled on the device for output row i, or nullptr when the host finishes that row's sampling
    ggml_tensor * get_sampled_id(int64_t i) const;

    ggml_cgraph  * get_gf()  const { return gf; }
    ggml_context * get_ctx() const { return ctx_compute.get(); }

    int64_t get_max_nodes() const;

    void reset();

    void set_inputs(const llama_ubatch * ubatch);
    void set_outputs(const llm_graph_params & params);

    // try to update the existing graph result using the new graph parameters in order to reuse it
    // this can only be done if we determine that the resulting graph using the new graph parameters
    //   would be identical to the existing graph. in that case, we simply have to update the memory
    //   contexts of the input tensors of the graph and we can reuse it for another computation
    // return true if the graph was updated and can be reused
    bool can_reuse(const llm_graph_params & params);

    llm_graph_input_i * add_input(llm_graph_input_ptr input);

    void add_fused_node(llm_graph_fused_node result);

    const std::vector<llm_graph_fused_node> & get_fused_nodes() const { return fused_nodes; }

    void set_params(const llm_graph_params & params);

    // important graph nodes
    ggml_tensor * t_inp_tokens  = nullptr;
    ggml_tensor * t_inp_embd    = nullptr; // [n_embd_inp, n_tokens]
    ggml_tensor * t_logits      = nullptr;
    ggml_tensor * t_embd        = nullptr;
    ggml_tensor * t_embd_pooled = nullptr;
    ggml_tensor * t_h_nextn     = nullptr; // [n_embd, n_outputs] hidden state before final output norm

    // what stopped the last can_reuse, for LLAMA_ROUND_TIMING
    const char * reuse_blocker = nullptr;

    std::vector<ggml_tensor *> t_layer_inp;

    std::vector<ggml_tensor *> t_sampled;
    std::vector<ggml_tensor *> t_sampled_probs;
    std::vector<ggml_tensor *> t_sampled_logits;
    std::vector<ggml_tensor *> t_candidates;
    std::vector<ggml_tensor *> t_mtp_ids; // I32 [S], one per chained draft step, then the accepted counts
    ggml_tensor * t_mtp_ids_all = nullptr; // I32 [K, S], the same ids with a sequence's together
    ggml_tensor * t_mtp_q = nullptr;       // F32 [2*LLAMA_MTP_DRAFT_TOP_K, K, S]: each draft's candidates (ids) and probabilities
    ggml_tensor * t_mtp_hpend = nullptr;  // F32 [n_embd, S], the anchors' hidden rows for the next round

    // the sequence count the graph's matrix multiplications carry (ggml_mul_mat_set_n_seqs): 2 for several, else 0
    int32_t mm_seqs_tag = 0;

    // the token ids come in the chain form, every sampled id goes to tok_io, and no other input reads token ids
    bool chain_ok  = false;
    // a recurrent state this graph reads could not be kept for a rollback
    bool chain_bad = false;
    // the recurrent states gathered into their step rows
    int32_t n_step_states = 0;
    bool draft_head_unused = false;       // the MTP head did not multiply by the rows the draft head holds
    bool logits_internal   = false;       // t_logits is no graph output (llm_graph_logits_internal)

    std::vector<llm_graph_input_ptr> inputs;
    std::vector<llm_graph_fused_node> fused_nodes;

    ggml_context_ptr ctx_compute;

    // memory buffers used to evaluate the model
    std::vector<uint8_t> buf_compute_meta;

    ggml_cgraph * gf;

    int64_t max_nodes;

    void set_samplers_greedy(bool v) { params.samplers_greedy = v; }
    void set_samplers_batch(uint64_t v) { params.samplers_batch = v; }

    // the tensors the scheduler placed in its compute buffers, noted after the graph's allocation; the buffers may
    // be allocated again for other graphs, so the next allocation of this one starts from unplaced tensors
    std::vector<ggml_tensor *> sched_tensors;
    std::vector<ggml_tensor *> src_snapshot; // see snapshot_srcs

    void note_sched_tensors();

    // The scheduler's split rewrites a node's src to the copy it makes in its own context, which the next split
    // frees: a graph split again (one that gets its allocation back) first takes its original srcs back
    void snapshot_srcs();
    void restore_srcs();
    void clear_sched_tensors();

private:
    // keep a copy of the previous graph parameters
    // we will use this to determine whether the graph can be reused by comparing them with the new parameters
    // note: these are updated after constructing the new graph
    llm_graph_params params;

    // env: LLAMA_GRAPH_RESULT_DEBUG
    int debug = 0;
};

using llm_graph_result_ptr = std::unique_ptr<llm_graph_result>;

//
// llm_graph_context
//

// used in build_rs to properly order writes and avoid unnecessary copies
using llm_graph_get_rows_fn = std::function<ggml_tensor * (ggml_context *, ggml_tensor * states, ggml_tensor * ids)>;

// A tied token embedding is loaded twice: the table rows are gathered from (in mapped host memory when the input
// layer keeps it there) and the output head's copy in device memory. Gathers read the head's copy: the same bytes,
// without crossing PCIe. The table itself when there is no such copy.
static inline ggml_tensor * llm_gather_table(ggml_tensor * tok_embd, ggml_tensor * output) {
    if (!tok_embd || !output || output == tok_embd || !tok_embd->buffer || !output->buffer ||
            output->type != tok_embd->type || !ggml_are_same_shape(output, tok_embd) ||
            strcmp(ggml_get_name(output), ggml_get_name(tok_embd)) != 0 ||
            ggml_backend_buffer_is_host(output->buffer) || ggml_backend_buffer_is_host(tok_embd->buffer) ||
            ggml_backend_buffer_get_type(output->buffer) == ggml_backend_buffer_get_type(tok_embd->buffer)) {
        return tok_embd;
    }
    return output;
}

struct llm_graph_qkv {
    ggml_tensor * q; // [n_embd_head, n_head,    n_tokens]
    ggml_tensor * k; // [n_embd_head, n_head_kv, n_tokens]
    ggml_tensor * v; // [n_embd_head, n_head_kv, n_tokens]
};

struct llm_graph_context {
    const llm_arch arch;

    const llama_hparams & hparams;
    const llama_cparams & cparams;
    const llama_ubatch  & ubatch;

    const int64_t n_embd;
    const int64_t n_layer;
    const int64_t n_layer_nextn;
    const int64_t n_rot;
    const int64_t n_ctx;       // user-specified context size (can be different from n_ctx_train)
    const int64_t n_head;
    const int64_t n_head_kv;
    const int64_t n_embd_head_k;
    const int64_t n_embd_k_gqa;
    const int64_t n_embd_head_v;
    const int64_t n_embd_v_gqa;
    const int64_t n_expert;
    const int64_t n_expert_used;

    const float freq_base;
    const float freq_scale;
    const float ext_factor;
    const float attn_factor;
    const float beta_fast;
    const float beta_slow;
    const float norm_eps;
    const float norm_rms_eps;

    const int64_t n_tokens;
    const int64_t n_outputs;
    const int32_t n_ctx_orig; // yarn

    const enum llama_pooling_type pooling_type;
    const enum llama_rope_type    rope_type;

    ggml_backend_sched_t sched;

    ggml_backend_t backend_cpu; // TODO: needed by build_attn_mha, figure out a way to remove?

    const llama_adapter_cvec     * cvec;
    const llama_adapter_loras    * loras;
    const llama_memory_context_i * mctx;
    const llama_cross            * cross;

    llama_mtp_round_state  * mtp_round;
    llama_tokens_dev_state * tokens_dev;
    llama_embd_dev_state   * embd_dev;
    const llama_loop_state       * loop;
    const llama_decode_chain_state * chain;

    const llama_draft_head * draft_head;

    // while set, build_lora_mm multiplies by draft_head->w where it would multiply by this
    mutable const ggml_tensor * mm_subst = nullptr;

    // the stream maps of a ubatch that skips idle sequences, made by the first attention that needs them
    mutable llm_graph_input_strm * inp_strm = nullptr;

    // while a grouped forward's attention is built for one group: its ubatch and index (build_attn, build_attn_mha)
    mutable const llama_ubatch * ubatch_attn = nullptr;
    mutable int strm_group = -1;
    mutable const llama_memory_context_i * strm_gmctx = nullptr;
    mutable llm_graph_input_strm * inp_strm_g[16] = {};
    mutable ggml_tensor * fa_kv_strm = nullptr; // the next flash attention's K/V stream input (src[5])
    mutable llm_graph_input_mtp_rows * mtp_rows = nullptr; // a round over one stream per sequence: its K/V cells
    mutable ggml_tensor * shexp_ids = nullptr; // [1, n_tokens] of n_expert, the stacked shared expert's id (build_moe_ffn)
    llm_graph_input_strm * build_inp_strm(int64_t n_seqs, int64_t n_span) const;

    // a draft model's reduced head: logits for every token, -inf for those it leaves out
    void build_draft_logits();

    std::map<llama_seq_id, llama_sampler *> samplers;

    const llm_graph_cb & cb_func;

    llm_graph_result * res;

    ggml_context * ctx0 = nullptr;
    ggml_cgraph  * gf   = nullptr;

    llm_graph_context(const llm_graph_params & params);
    virtual ~llm_graph_context() = default;

    void cb(ggml_tensor * cur, const char * name, int il) const;

    //
    // common
    //

    ggml_tensor * build_cvec(
             ggml_tensor * cur,
                     int   il) const;

    // do mat_mul, while optionally apply lora and per-tensor scale
    ggml_tensor * build_lora_mm(
              ggml_tensor * w,
              ggml_tensor * cur,
              ggml_tensor * w_s = nullptr) const;

    // do mat_mul_id, while optionally apply lora and per-expert scale
    ggml_tensor * build_lora_mm_id(
              ggml_tensor * w,   // ggml_tensor * as
              ggml_tensor * cur, // ggml_tensor * b
              ggml_tensor * ids,
              ggml_tensor * w_s = nullptr) const;

    ggml_tensor * build_norm(
             ggml_tensor * cur,
             ggml_tensor * mw,
             ggml_tensor * mb,
           llm_norm_type   type,
                     int   il) const;


    // compute Q, K, V projections with optional bias and reshape
    // supports both fused wqkv and separate wq/wk/wv paths
    llm_graph_qkv build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head,
                  int64_t   n_head,
                  int64_t   n_head_kv,
                      int   il,
                  int64_t   n_rows = 0) const;

    // Set reshape to false to return contiguous projections before clamp/reshape.
    llm_graph_qkv build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head_q,
                  int64_t   n_head_q,
                  int64_t   n_embd_head_k,
                  int64_t   n_head_k,
                  int64_t   n_embd_head_v,
                  int64_t   n_head_v,
                      int   il,
                     bool   reshape = true,
                  int64_t   n_rows  = 0) const; // a round's step covers only its own rows; 0 = n_tokens

    ggml_tensor * build_ffn(
             ggml_tensor * cur,
             ggml_tensor * up,
             ggml_tensor * up_b,
             ggml_tensor * up_s,
             ggml_tensor * gate,
             ggml_tensor * gate_b,
             ggml_tensor * gate_s,
             ggml_tensor * down,
             ggml_tensor * down_b,
             ggml_tensor * down_s,
             ggml_tensor * act_scales,
         llm_ffn_op_type   type_op,
       llm_ffn_gate_type   type_gate,
                     int   il) const;

    // build MoE FFN without bias tensors
    ggml_tensor * build_moe_ffn(
             ggml_tensor * cur,
             ggml_tensor * gate_inp,
             ggml_tensor * up_exps,
             ggml_tensor * gate_exps,
             ggml_tensor * down_exps,
             ggml_tensor * exp_probs_b,
                 int64_t   n_expert,
                 int64_t   n_expert_used,
         llm_ffn_op_type   type_op,
                    bool   norm_w,
                   float   w_scale,
            llama_expert_gating_func_type gating_op,
                     int   il,
             ggml_tensor * probs_in = nullptr,
             ggml_tensor * gate_up_exps = nullptr,
             ggml_tensor * up_exps_s = nullptr,
             ggml_tensor * gate_exps_s = nullptr,
             ggml_tensor * down_exps_s = nullptr,
             ggml_tensor * selected_experts_in = nullptr,
             ggml_tensor * shexp_gate = nullptr) const;

    // shexp_gate: the sigmoid gate of a shared expert stacked as expert n_expert (llama_model_base::load_tensors);
    // every token is routed to it as well, weighted by the gate
    ggml_tensor * build_moe_ffn(
             ggml_tensor * cur,
             ggml_tensor * gate_inp,
             ggml_tensor * gate_inp_b,
             ggml_tensor * up_exps,
             ggml_tensor * up_exps_b,
             ggml_tensor * gate_exps,
             ggml_tensor * gate_exps_b,
             ggml_tensor * down_exps,
             ggml_tensor * down_exps_b,
             ggml_tensor * exp_probs_b,
                 int64_t   n_expert,
                 int64_t   n_expert_used,
         llm_ffn_op_type   type_op,
                    bool   norm_w,
                   float   w_scale,
            llama_expert_gating_func_type gating_op,
                     int   il,
             ggml_tensor * probs_in = nullptr,
             ggml_tensor * gate_up_exps = nullptr,
             ggml_tensor * gate_up_exps_b = nullptr,
             ggml_tensor * up_exps_s = nullptr,
             ggml_tensor * gate_exps_s = nullptr,
             ggml_tensor * down_exps_s = nullptr,
             ggml_tensor * selected_experts_in = nullptr,
             ggml_tensor * shexp_gate = nullptr) const;

    // whether a layer's shared expert runs as expert n_expert (see build_moe_ffn): stacked at load, and no adapter
    // changes the expert tensors
    bool shexp_stacked(const llama_layer & layer) const;

    //
    // inputs
    //

    ggml_tensor * build_inp_embd(ggml_tensor * tok_embd) const;

    // the token ids of this graph come through the decode chain's tok_io
    bool is_chain_form() const;
    ggml_tensor * build_inp_pos() const;

    // the loop body's step, as an index
    // the window rows of this step's tokens: counter + 0 .. n_tokens-1, I32 [n_tokens]
    ggml_tensor * loop_rows() const;

    // after sampling: the id joins the ring, feeds the next step, the step advances
    void build_loop_tail() const;

    // a whole speculative round in this graph, see llama_set_mtp_chain
    void build_mtp_round(
             llm_graph_mtp_i & mtp,
    const llm_graph_mtp_attn & attn,
                 ggml_tensor * tok_embd,
                 ggml_tensor * inp_tokens,
                 ggml_tensor * inp_pos,
                         int   n_chain,
                     int64_t   n_embd_h = 0);

    // The anchors of a draft round over the target's cache, taken on the device (llama_set_mtp_dev_anchors): the
    // ubatch holds n_chain + 1 rows per sequence, row j the anchor had the target accepted j drafts. The verify's
    // samples and hidden rows come over from the target's last graph; the accepted count is the run of drafts equal to
    // the target's samples, as vLLM's rejection sampler counts them on the device.
    struct dev_anchors {
        ggml_tensor * sel_flat  = nullptr; // I32 [S]: the anchor's row of the ubatch, s*(n_chain + 1) + n_acc[s]
        ggml_tensor * sel_local = nullptr; // I32 [1, S]: n_acc[s], its row within the sequence
        ggml_tensor * id        = nullptr; // I32 [S]: the anchor token, the target's sample after the accepted drafts
        ggml_tensor * h         = nullptr; // F32 [n_embd_h, S]: the target's hidden row of the last accepted token
        ggml_tensor * n_acc     = nullptr; // I32 [S]
    };
    dev_anchors build_mtp_dev_anchors(ggml_tensor * inp_tokens, int n_chain, int64_t n_embd_h, ggml_tensor * table) const;

    // the round over n_seq > 1 sequences
    void build_mtp_round_seqs(
             llm_graph_mtp_i & mtp,
    const llm_graph_mtp_attn & attn,
                 ggml_tensor * tok_embd,
                 ggml_tensor * inp_tokens,
                 ggml_tensor * inp_pos,
                         int   n_chain,
                     int64_t   n_seq,
                 ggml_tensor * c_ids,
                 ggml_tensor * h_tgt,
                 ggml_tensor * h_pend,
                 ggml_tensor * off);

    // one row group of a round: write its K/V rows, then attend against the
    // cache with the round's mask. row is the batch row the group starts at;
    // a replay (kv_only) refreshes the cache and returns nothing.
    ggml_tensor * build_attn_mtp(
    const llm_graph_mtp_attn & attn,
                 ggml_tensor * wo,
                 ggml_tensor * wo_b,
                 ggml_tensor * wo_s,
                 ggml_tensor * q,
                 ggml_tensor * k,
                 ggml_tensor * v,
                 ggml_tensor * sinks,
                 ggml_tensor * v_mla,
                       float   kq_scale,
                         int   il,
                         int   row,
                        bool   kv_only,
                 ggml_tensor * mask_in) const;
    ggml_tensor * build_inp_attn_scale() const;
    ggml_tensor * build_inp_out_ids() const;
    ggml_tensor * build_inp_mean() const;
    ggml_tensor * build_inp_cls() const;

    ggml_tensor * build_inp_cross_embd() const;
    ggml_tensor * build_inp_pos_bucket_enc() const;
    ggml_tensor * build_inp_pos_bucket_dec() const;
    ggml_tensor * build_pos_bias(ggml_tensor * pos_bucket, ggml_tensor * attn_rel_b) const;

    //
    // attention
    //

    ggml_tensor * build_attn_mha(
            ggml_tensor * q,       // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k,       // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v,       // [n_embd_head_v, n_head_v, n_tokens] (v_trans = false)
            ggml_tensor * kq_b,
            ggml_tensor * kq_mask,
            ggml_tensor * sinks,   // [n_head_q]
            ggml_tensor * v_mla,   // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                int64_t   n_kv_max,
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_no_cache * build_attn_inp_no_cache() const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_no_cache * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_kv * build_attn_inp_kv() const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_kv * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v] // TODO: remove
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_k  * build_attn_inp_k() const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_k * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_k_dsa * build_attn_inp_k_dsa() const;

    llm_graph_input_attn_k_dsa_iswa * build_attn_inp_k_dsa_iswa() const;

    llm_graph_input_attn_kv_msa * build_attn_inp_kv_msa(bool msa_enabled) const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_k_dsa * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
            ggml_tensor * top_k, // [n_indexer_top_k, n_tokens]
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_kv_iswa * build_attn_inp_kv_iswa() const;

    llm_graph_input_dsv4 * build_inp_dsv4() const;

    // note: if k_cur or v_cur are not provided, they will not be stored in the memory
    ggml_tensor * build_attn(
            llm_graph_input_attn_kv_iswa * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens] optional
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens] optional
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_k_iswa * build_attn_inp_k_iswa() const;

    // note: if k_cur is not provided, it will not be stored in the memory
    // note: the K cache is used as V (MLA-style attention)
    ggml_tensor * build_attn(
            llm_graph_input_attn_k_iswa * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens] optional
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens] optional
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_cross * build_attn_inp_cross() const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_cross * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    //
    // recurrent
    //

    // TODO: move this implementation to llama_memory_recurrent.
    //       this is analogous to llama_kv_cache::cpy_k / cpy_v
    //       when moving, avoid passing `ggml_cgraph` - only pass `ggml_context`. would likely need to split the
    //         implementation in 2 separate methods. the goal is to avoid calling `ggml_build_forward_expand` in
    //         `llama_memory_recurrent`
    ggml_tensor * build_rs(
            ggml_tensor * s,
            ggml_tensor * state_copy_main,
            ggml_tensor * state_copy_extra,
                int32_t   state_size,
                int32_t   n_seqs,
               uint32_t   n_rs,
               uint32_t   rs_head,
               uint32_t   rs_size,
                int32_t   rs_zero,
            const llm_graph_get_rows_fn & get_state_rows = ggml_get_rows,
            ggml_tensor * step = nullptr) const;

    llm_graph_input_rs * build_rs_inp() const;

    ggml_tensor * build_rs(
            llm_graph_input_rs * inp,
            ggml_tensor * s,
                int32_t   state_size,
                int32_t   n_seqs,
            const llm_graph_get_rows_fn & get_state_rows = ggml_get_rows) const;

    ggml_tensor * build_rwkv_token_shift_load(
        llm_graph_input_rs * inp,
        const llama_ubatch & ubatch,
                       int   il) const;

    ggml_tensor * build_rwkv_token_shift_store(
             ggml_tensor * token_shift,
      const llama_ubatch & ubatch,
                     int   il) const;
    //
    // hybrid
    //

    llm_graph_input_mem_hybrid * build_inp_mem_hybrid() const;
    llm_graph_input_mem_hybrid_k * build_inp_mem_hybrid_k() const;

    llm_graph_input_mem_hybrid_iswa * build_inp_mem_hybrid_iswa() const;

    //
    // pooling
    //

    void build_pooling(
            ggml_tensor * cls,
            ggml_tensor * cls_b,
            ggml_tensor * cls_out,
            ggml_tensor * cls_out_b,
            ggml_tensor * cls_norm) const;

    //
    // sampling (backend sampling)
    //

    void build_sampling() const;

    //
    // dense (out)
    //

    void build_dense_out(
            ggml_tensor * dense_2,
            ggml_tensor * dense_2_b,
            ggml_tensor * dense_3) const;
};

// TODO: better name
int32_t llama_relative_position_bucket(llama_pos x, llama_pos y, uint64_t n_buckets, bool bidirectional);
