#include "llama-graph.h"

#include <typeinfo>
#include <map>
#include <string>
#include "llama-context.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-sampler.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-dsa.h"
#include "llama-kv-cache-dsa-iswa.h"
#include "llama-kv-cache-msa.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"

#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_set>

// dedup helpers

// the sequences of a ubatch and the cache streams its attention covers: with a non-unified cache a stream per
// sequence, from the first sequence's to the last's, which is more than the sequences when idle ones lie between
struct ubatch_streams {
    int64_t n_seqs;
    int64_t n_span;
};

static ubatch_streams get_ubatch_streams(const llama_ubatch & ubatch, const llama_cparams & cparams, const llama_kv_cache_context * mctx) {
    if (cparams.kv_unified || ubatch.n_seqs_unq <= 1) {
        return { 1, 1 };
    }
    return { (int64_t) ubatch.n_seqs_unq, (int64_t) mctx->get_n_stream_span() };
}

bool llm_graph_samplers_greedy(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers) {
    if (samplers.empty() || llama_host_off(LLAMA_HOST_OFF_GREEDY_BATCH)) {
        return false;
    }
    bool any = false;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (!ubatch.output[i]) {
            continue;
        }
        const auto it = samplers.find(ubatch.seq_id[i][0]);
        if (it == samplers.end() || !llama_sampler_is_backend_greedy(it->second)) {
            return false;
        }
        any = true;
    }
    return any;
}

// a uniform in [0, 1) from a request's seed and a position (speculative sampling's draws besides the chain's own)
static float llm_spec_uniform(uint32_t seed, int32_t pos, uint32_t salt) {
    uint64_t x = ((uint64_t) seed << 32) ^ (uint32_t) pos ^ ((uint64_t) salt * 0x9E3779B97F4A7C15ull);
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27; x *= 0x94D049BB133111EBull;
    x ^= x >> 31;
    return (float) (x >> 40) * (1.0f / 16777216.0f);
}

// an output row the next row of the batch continues checks that row's token as a draft
static bool llm_spec_drafted(const llama_ubatch & ubatch, uint32_t i) {
    return i + 1 < ubatch.n_tokens && ubatch.output[i] && ubatch.output[i + 1] && ubatch.n_seq_id[i] == 1 &&
           ubatch.n_seq_id[i + 1] == 1 && ubatch.seq_id[i][0] == ubatch.seq_id[i + 1][0] && ubatch.pos[i + 1] == ubatch.pos[i] + 1;
}

uint64_t llm_graph_samplers_batch_key(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers,
        int32_t * lw_out, int32_t * kp_out, int32_t * nb_out, int32_t * k_out, bool * rep_out, bool * minp_out, bool * t0_out,
        bool * sd_out) {
    static const bool on = getenv("LLAMA_SAMPLING_BATCH") == nullptr || atoi(getenv("LLAMA_SAMPLING_BATCH")) != 0;
    if (!on || samplers.empty()) {
        return 0;
    }
    int32_t k = 0, lw = 0, nb = 0;
    bool rep = false, minp = false, t0 = false, any = false;
    std::map<llama_seq_id, int32_t> rows;
    std::map<llama_seq_id, llama_sampler_batch_params> seen;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (!ubatch.output[i]) {
            continue;
        }
        const llama_seq_id seq = ubatch.seq_id[i][0];
        const auto it = samplers.find(seq);
        if (it == samplers.end()) {
            return 0;
        }
        rows[seq]++;
        if (seen.find(seq) == seen.end()) {
            llama_sampler_batch_params bp;
            if (llama_sampler_is_backend_greedy(it->second)) {
                // a greedy row takes the batch's candidates at temperature 0
                bp.temp = 0.0f;
                t0 = true;
                seen[seq] = bp;
                any = true;
                continue;
            }
            if (!llama_sampler_batch_params_get(it->second, bp) || bp.penalty_last_n > 256 || bp.bias.size() > 64 ||
                    (k != 0 && bp.top_k != k)) {
                return 0;
            }
            k    = bp.top_k;
            lw   = std::max(lw, bp.penalty_last_n);
            nb   = std::max(nb, (int32_t) bp.bias.size());
            rep  = rep || bp.penalty_repeat != 1.0f;
            minp = minp || bp.min_p > 0.0f;
            t0   = t0 || bp.temp <= 0.0f;
            seen[seq] = bp;
        }
        any = true;
    }
    if (!any || k <= 0) {
        return 0;
    }
    // rows that check drafts take them by speculative sampling (build_sampling); LLAMA_NO_SPEC_SAMPLING=1 compares
    // samples instead
    static const bool spec_off = getenv("LLAMA_NO_SPEC_SAMPLING") != nullptr;
    bool sd = false;
    for (uint32_t i = 0; !spec_off && ubatch.pos && !sd && i < ubatch.n_tokens; ++i) {
        sd = llm_spec_drafted(ubatch, i);
    }
    int32_t kp = 0;
    if (lw > 0) {
        for (const auto & r : rows) {
            kp = std::max(kp, r.second - 1);
        }
    }
    if (lw_out)   *lw_out   = lw;
    if (kp_out)   *kp_out   = kp;
    if (nb_out)   *nb_out   = nb;
    if (k_out)    *k_out    = k;
    if (rep_out)  *rep_out  = rep;
    if (minp_out) *minp_out = minp;
    if (t0_out)   *t0_out   = t0;
    if (sd_out)   *sd_out   = sd;
    return 1ull | ((uint64_t) k << 1) | ((uint64_t) lw << 21) | ((uint64_t) kp << 31) | ((uint64_t) nb << 39) |
           ((uint64_t) rep << 47) | ((uint64_t) minp << 48) | ((uint64_t) t0 << 49) | ((uint64_t) sd << 50);
}

bool llm_graph_logits_internal(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers) {
    // greedy rows, or rows of the batched form (build_sampling reads the logits on the device either way)
    if (!llm_graph_samplers_greedy(ubatch, samplers) && llm_graph_samplers_batch_key(ubatch, samplers) == 0) {
        return false;
    }
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.output[i] && !llama_sampler_backend_selects(samplers.at(ubatch.seq_id[i][0]))) {
            return false;
        }
    }
    return true;
}

void llm_graph_params_fill(llm_graph_params & params) {
    params.strm_s0 = -1;
    params.strm_n  = 0;
    params.samplers_greedy = llm_graph_samplers_greedy(params.ubatch, params.samplers);
    params.logits_internal = llm_graph_logits_internal(params.ubatch, params.samplers);
    params.samplers_batch  = params.samplers_greedy ? 0 : llm_graph_samplers_batch_key(params.ubatch, params.samplers);

    const llama_kv_cache_context * kv = nullptr;
    if (const auto * c = dynamic_cast<const llama_kv_cache_context *>(params.mctx)) {
        kv = c;
    } else if (const auto * c = dynamic_cast<const llama_kv_cache_iswa_context *>(params.mctx)) {
        kv = c->get_base();
    } else if (const auto * c = dynamic_cast<const llama_memory_hybrid_context *>(params.mctx)) {
        kv = c->get_attn();
    }
    if (kv) {
        params.strm_s0 = params.cparams.kv_unified ? 0 : (int32_t) kv->get_stream_s0();
        params.strm_n  = params.cparams.kv_unified ? 1 : (int32_t) kv->get_n_stream_span();
    }
    if (dynamic_cast<const llama_kv_cache_iswa_context *>(params.mctx) && llm_graph_attn_streams_in(params.cparams)) {
        params.strm_s0 = -1;
        params.strm_n  = -1;
    }

    params.groups_sig.clear();
    if (const auto * c = dynamic_cast<const llama_kv_cache_iswa_context *>(params.mctx)) {
        for (uint32_t g = 0; g < c->n_groups(); ++g) {
            const auto * v = c->get_group(g);
            if (!v) {
                break;
            }
            const llama_ubatch & u = v->get_ubatch();
            // a group reads its streams from an input (llm_graph_input_attn_kv_iswa::kv_strm): which streams, and
            // the span between them, are not part of the topology
            const bool strm_in = llm_graph_attn_streams_in(params.cparams);
            params.groups_sig.push_back({ (int32_t) u.n_tokens, (int32_t) u.n_seqs,
                    strm_in ? -1 : (int32_t) v->get_base()->get_stream_s0(),
                    strm_in ? -1 : (int32_t) v->get_base()->get_n_stream_span() });
        }
        if (!params.groups_sig.empty()) {
            params.strm_s0 = -1;
            params.strm_n  = 0;
            // one attention over every group's rows: the groups are not part of the topology
            if (llm_graph_attn_varlen(params.cparams, params.hparams, c)) {
                params.groups_sig.assign(1, { -1, -1, -1, -1 });
            }
        }
    }

    params.n_kv_sig = 0;
    auto mix = [&](uint64_t v) { params.n_kv_sig = (params.n_kv_sig ^ v) * 0x100000001b3ULL; };
    int64_t n_kv_varlen = 0, n_kv_varlen_swa = 0;
    if (const auto * c = dynamic_cast<const llama_kv_cache_iswa_context *>(params.mctx)) {
        if (llm_graph_attn_varlen(params.cparams, params.hparams, c, &n_kv_varlen, &n_kv_varlen_swa)) {
            mix(n_kv_varlen);
            mix(n_kv_varlen_swa);
        } else if (c->n_groups() > 0) {
            for (uint32_t g = 0; g < c->n_groups(); ++g) {
                if (const auto * v = c->get_group(g)) {
                    mix(v->get_base()->get_n_kv());
                    mix(v->get_swa()->get_n_kv());
                }
            }
        } else {
            mix(c->get_base()->get_n_kv());
            mix(c->get_swa()->get_n_kv());
        }
    } else if (kv) {
        mix(kv->get_n_kv());
    }
}

// per_seq: a mask per sequence rather than per stream of the span (a group whose attention reads its streams from an
// input, llm_graph_input_attn_kv_iswa::kv_strm)
// the attention of an ISWA cache with a stream per sequence reads each sequence's stream from an input
// (llm_graph_input_attn_kv_iswa::kv_strm) and covers only the ubatch's sequences, not the streams between them;
// LLAMA_ATTN_STREAMS_IN=0 off
bool llm_graph_attn_streams_in(const llama_cparams & cparams) {
    static const bool off = getenv("LLAMA_ATTN_STREAMS_IN") && atoi(getenv("LLAMA_ATTN_STREAMS_IN")) == 0;
    return !off && cparams.flash_attn && !cparams.kv_unified && cparams.n_seq_max > 1;
}

// The CUDA kernel has the variable-length layout for head sizes 256 and 512 with 4 query heads a K/V head (Gemma 4),
// over cache views a multiple of 256 cells. LLAMA_ATTN_VARLEN=0 off
bool llm_graph_attn_varlen(const llama_cparams & cparams, const llama_hparams & hparams,
        const llama_kv_cache_iswa_context * mctx, int64_t * n_kv, int64_t * n_kv_swa) {
    static const bool off = getenv("LLAMA_ATTN_VARLEN") && atoi(getenv("LLAMA_ATTN_VARLEN")) == 0;
    if (off || !mctx || mctx->n_groups() < 2 || !llm_graph_attn_streams_in(cparams)) {
        return false;
    }
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        const uint32_t d = hparams.n_embd_head_k(il);
        if ((d != 256 && d != 512) || hparams.n_embd_head_v(il) != d || hparams.n_head(il) != 4*hparams.n_head_kv(il)) {
            return false;
        }
    }
    int64_t nb = 0;
    int64_t ns = 0;
    for (uint32_t g = 0; g < mctx->n_groups(); ++g) {
        const auto * v = mctx->get_group(g);
        if (!v) {
            return false;
        }
        nb = std::max<int64_t>(nb, v->get_base()->get_n_kv());
        ns = std::max<int64_t>(ns, v->get_swa()->get_n_kv());
    }
    if (nb % 256 != 0 || ns % 256 != 0) {
        return false;
    }
    if (n_kv) {
        *n_kv = nb;
    }
    if (n_kv_swa) {
        *n_kv_swa = ns;
    }
    return true;
}

static ggml_tensor * build_attn_inp_kq_mask(
        ggml_context * ctx,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams,
        bool per_seq = false) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;
    const auto strm     = get_ubatch_streams(ubatch, cparams, mctx);

    // flash attention requires an f16 mask
    const auto type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    ggml_tensor * res = ggml_new_tensor_4d(ctx, type, n_kv, n_tokens/strm.n_seqs, 1, per_seq ? strm.n_seqs : strm.n_span);
    ggml_set_input(res);
    ggml_set_name(res, "attn_inp_kq_mask");

    return res;
}

static bool can_reuse_kq_mask(
        ggml_tensor * kq_mask,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams,
        bool per_seq = false) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;
    const auto strm     = get_ubatch_streams(ubatch, cparams, mctx);

    bool res = true;

    res &= (kq_mask->ne[0] == n_kv);
    res &= (kq_mask->ne[1] == n_tokens/strm.n_seqs);
    res &= (kq_mask->ne[2] == 1);
    res &= (kq_mask->ne[3] == (per_seq ? strm.n_seqs : strm.n_span));

    return res;
}

// impl

static void mtp_round_copy(ggml_backend_sched_t sched_src, ggml_backend_sched_t sched_dst, ggml_tensor * src, ggml_tensor * dst, size_t nbytes, size_t dst_offset, size_t src_offset = 0);
static void mtp_round_order(ggml_backend_sched_t sched_dst, ggml_backend_t b_src);

// the drafted step and element a device-filled row takes, or -1 for a row the host fills
static int32_t tokens_dev_row(const llama_tokens_dev_state & td, const llama_ubatch & ubatch, int32_t r, int32_t & col) {
    for (const auto & sp : td.spans) {
        if (ubatch.n_seq_id[r] != 1 || ubatch.seq_id[r][0] != sp.seq) {
            continue;
        }
        const int32_t j = ubatch.pos[r] - sp.pos0 - 1;
        if (j >= 0 && j < sp.n) {
            col = sp.col;
            return j;
        }
    }
    return -1;
}

void llm_graph_input_embd::set_input(const llama_ubatch * ubatch) {
    if (device) {
        return;
    }
    if (ubatch->token && tok_in) {
        const int64_t n_tokens = ubatch->n_tokens;

        std::vector<int32_t> buf(2*n_tokens);
        for (int64_t i = 0; i < n_tokens; ++i) {
            const llama_seq_id s = ubatch->seq_id[i][0];
            const int32_t      r = s >= 0 && s < (llama_seq_id) chain->seq_row.size() ? chain->seq_row[s] : -1;

            buf[i]            = ubatch->token[i];
            buf[n_tokens + i] = r >= 0 ? r : chain->n_cm + (int32_t) i;
        }
        ggml_backend_tensor_set(tok_in, buf.data(), 0, buf.size()*sizeof(int32_t));
    } else if (ubatch->token) {
        const int64_t n_tokens = ubatch->n_tokens;
        const size_t  es       = ggml_element_size(tokens);

        llama_tokens_dev_state * td = tokens_dev;
        const bool armed = td && td->src && !td->spans.empty();
        const std::vector<ggml_tensor *> * ids = armed ? &td->src->get_mtp_ids_pending() : nullptr;

        // tok_dev is read on the target's stream: a write from another stream could land while the previous
        // decode's graph still reads it
        ggml_backend_t b_tok = nullptr;
        if (dev_tokens && sched) {
            b_tok = ggml_backend_sched_get_tensor_backend(sched, tokens);
            if (!b_tok && tokens->view_src) {
                b_tok = ggml_backend_sched_get_tensor_backend(sched, tokens->view_src);
            }
        }
        auto set_rows = [&](const void * data, size_t offset, size_t size) {
            if (b_tok) {
                ggml_backend_tensor_set_async(b_tok, tokens, data, offset, size);
            } else {
                ggml_backend_tensor_set(tokens, data, offset, size);
            }
        };
        if (armed && dev_tokens && !ids->empty()) {
            ggml_backend_sched_t sched_src = td->src->get_sched();

            // the padded verify: every span is one sequence's K rows after its anchor, in column order, so the
            // anchors go up as one strided upload and the drafted ids come over as one strided copy
            ggml_tensor * all = td->src->get_mtp_ids_all_pending();
            const int32_t K   = td->src->get_mtp_ids_n_draft();
            // a grouped forward's prompt rows follow the spans: they go up from the host behind the strided copies
            const int64_t n_reg = (int64_t) td->spans.size()*(K + 1);
            bool regular = all != nullptr && b_tok != nullptr && K > 0 && all->ne[0] == K &&
                    n_reg <= n_tokens && (int64_t) td->spans.size() <= all->ne[1];
            for (int64_t r = n_reg; regular && r < n_tokens; ++r) {
                int32_t col = 0;
                regular = tokens_dev_row(*td, *ubatch, (int32_t) r, col) < 0;
            }
            for (size_t c = 0; regular && c < td->spans.size(); ++c) {
                const auto & sp = td->spans[c];
                regular = sp.col == (int32_t) c && sp.n == K;
                for (int32_t j = 0; regular && j <= K; ++j) {
                    const int64_t r = (int64_t) c*(K + 1) + j;
                    regular = ubatch->n_seq_id[r] == 1 && ubatch->seq_id[r][0] == sp.seq && ubatch->pos[r] == sp.pos0 + j;
                }
            }
            if (regular) {
                typedef void (*copy_2d_fn_t)(ggml_backend_t, void *, size_t, const void *, size_t, size_t, size_t);
                auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(b_tok));
                auto copy_2d = (copy_2d_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_copy_2d_async");
                ggml_backend_t b_all = ggml_backend_sched_get_tensor_backend(sched_src, all);
                regular = copy_2d != nullptr && b_all != nullptr;
                if (regular) {
                    const size_t S = td->spans.size();
                    char * base = (char *) tokens->data;
                    copy_2d(b_tok, base, (K + 1)*es, ubatch->token, (K + 1)*es, es, S);
                    copy_2d(b_all, base + es, (K + 1)*es, all->data, (size_t) K*es, (size_t) K*es, S);
                    mtp_round_order(td->sched_dst, b_all);
                    if (n_tokens > n_reg) {
                        set_rows(ubatch->token + n_reg, (size_t) n_reg*es, (size_t) (n_tokens - n_reg)*es);
                    }
                }
            }
            static int said_regular = 0;
            if (said_regular < 3 && getenv("LLAMA_ROUND_TIMING") != nullptr) {
                said_regular++;
                fprintf(stderr, "[round] token handoff: %s (n_tokens %d, spans %d, K %d, ids_all %p)\n", regular ? "one strided copy" : "row by row", (int) n_tokens, (int) td->spans.size(), (int) K, (void *) all);
            }
            if (regular) {
                return;
            }

            // the host's rows around the drafter's, which it never writes: a later write of its placeholders would
            // race the copies below. The rows sit on the device, so neither the writes nor the copies wait for the
            // draft graph; the verify's graph does, on the device
            int32_t r0 = 0;
            for (int32_t r = 0; r <= n_tokens; ++r) {
                int32_t col = 0;
                const int32_t j = r < n_tokens ? tokens_dev_row(*td, *ubatch, r, col) : -1;
                if (r == n_tokens || j >= 0) {
                    if (r > r0) {
                        set_rows(ubatch->token + r0, r0*es, (r - r0)*es);
                    }
                    r0 = r + 1;
                }
                if (j >= 0) {
                    // a replaying round's last pending tensor holds the accepted count, not a drafted id
                    GGML_ASSERT(j < td->src->get_mtp_ids_n_draft() && (size_t) (col + 1)*es <= ggml_nbytes((*ids)[j]));
                    mtp_round_copy(sched_src, td->sched_dst, (*ids)[j], tokens, es, (size_t) r*es, (size_t) col*es);
                }
            }
            mtp_round_order(td->sched_dst, ggml_backend_sched_get_tensor_backend(sched_src, (*ids)[0]));
        } else if (armed) {
            // a batch that did not get the device rows: read the drafted ids back here, and wait for them
            int32_t n_ids = 0;
            const llama_token * host_ids = td->src->get_mtp_chain_ids(n_ids);
            const int32_t n_step = td->src->get_mtp_chain_n_seq() > 0 ? n_ids / td->src->get_mtp_chain_n_seq() : 0;
            std::vector<llama_token> rows(ubatch->token, ubatch->token + n_tokens);
            for (int32_t r = 0; r < n_tokens; ++r) {
                int32_t col = 0;
                const int32_t j = tokens_dev_row(*td, *ubatch, r, col);
                if (j >= 0) {
                    GGML_ASSERT(j < td->src->get_mtp_ids_n_draft() && j < n_step && col < td->src->get_mtp_chain_n_seq());
                    rows[r] = host_ids[j*td->src->get_mtp_chain_n_seq() + col];
                }
            }
            set_rows(rows.data(), 0, n_tokens*es);
        } else {
            set_rows(ubatch->token, 0, n_tokens*es);
        }
    }

    if (ubatch->embd) {
        GGML_ASSERT(n_embd == embd->ne[0]);

        const int64_t n_tokens  = ubatch->n_tokens;
        const size_t  row_bytes = (size_t) n_embd*ggml_element_size(embd);

        if (idx) {
            // a chained draft: the source's hidden rows come over as one block into the stage, the host's anchors go
            // behind it, and the graph gathers each sequence's anchor by index; embd is not read
            const int32_t S = (int32_t) idx->ne[0];
            const int32_t K = chain_k;
            GGML_ASSERT(embd_dev && embd_dev->stage && K > 0 && (int64_t) S*K == n_tokens);
            ggml_tensor * stage = embd_dev->stage;
            GGML_ASSERT(stage->ne[0] == n_embd);
            int64_t n_src = 0;
            std::vector<int32_t> idx_h(S, -1);
            if (embd_dev->src && !embd_dev->rows.empty()) {
                const llm_graph_result * res_src = embd_dev->src->get_gf_res_last();
                ggml_tensor * src_h = res_src ? res_src->get_h_nextn() : nullptr;
                GGML_ASSERT(src_h && src_h->ne[0] == n_embd && ggml_is_contiguous(src_h) && sched && "device embedding rows: no hidden rows on the source");
                n_src = src_h->ne[1];
                GGML_ASSERT(n_src + S <= stage->ne[1] && "device embedding rows: the stage is too small");
                ggml_backend_sched_t sched_src = embd_dev->src->get_sched();
                mtp_round_copy(sched_src, sched, src_h, stage, (size_t) n_src*row_bytes, 0, 0);
                mtp_round_order(sched, ggml_backend_sched_get_tensor_backend(sched_src, src_h));
                for (const auto & r : embd_dev->rows) {
                    GGML_ASSERT(r[1] >= 0 && r[1] < n_src);
                    for (int32_t s = 0; s < S; ++s) {
                        if (ubatch->n_seq_id[s*K] == 1 && ubatch->seq_id[s*K][0] == r[0]) {
                            idx_h[s] = r[1];
                        }
                    }
                }
            }
            for (int32_t s = 0; s < S; ++s) {
                if (idx_h[s] < 0) {
                    idx_h[s] = (int32_t) n_src + s;
                    ggml_backend_tensor_set(stage, ubatch->embd + (size_t) s*K*n_embd, (size_t) (n_src + s)*row_bytes, row_bytes);
                }
            }
            ggml_backend_tensor_set(idx, idx_h.data(), 0, (size_t) S*sizeof(int32_t));
            static int said_gather = 0;
            if (said_gather < 3 && getenv("LLAMA_ROUND_TIMING") != nullptr) {
                said_gather++;
                int n_host = 0;
                for (int32_t s = 0; s < S; ++s) {
                    n_host += idx_h[s] >= n_src;
                }
                fprintf(stderr, "[round] anchors: gathered from the stage (S %d, source rows %d, host rows %d)\n", (int) S, (int) n_src, n_host);
            }
            embd_dev->rows.clear();
            embd_dev->src = nullptr;
        } else if (!(embd_dev && embd_dev->src && !embd_dev->rows.empty())) {
            ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*row_bytes);
        } else {
            // rows another context's last graph holds (a drafter's anchors in the target's verify) go over on the
            // device, into the first row of their sequence; that sequence's other rows stay as they are, a chained
            // draft never reads them (gemma4-assistant.cpp), and nothing of it comes up from the host
            const llm_graph_result * res_src = embd_dev->src->get_gf_res_last();
            ggml_tensor * src_h = res_src ? res_src->get_h_nextn() : nullptr;
            GGML_ASSERT(src_h && src_h->ne[0] == n_embd && ggml_is_contiguous(src_h) && sched && "device embedding rows: no hidden rows on the source");
            ggml_backend_sched_t sched_src = embd_dev->src->get_sched();
            std::vector<uint8_t> dev(n_tokens, 0);
            for (const auto & r : embd_dev->rows) {
                int64_t r0 = -1;
                for (int64_t i = 0; i < n_tokens; ++i) {
                    if (ubatch->n_seq_id[i] == 1 && ubatch->seq_id[i][0] == r[0]) {
                        r0 = r0 < 0 ? i : r0;
                        dev[i] = 1;
                    }
                }
                GGML_ASSERT(r0 >= 0 && r[1] >= 0 && r[1] < src_h->ne[1] && "device embedding rows: the sequence has no row in the batch");
                mtp_round_copy(sched_src, sched, src_h, embd, row_bytes, (size_t) r0*row_bytes, (size_t) r[1]*row_bytes);
            }
            mtp_round_order(sched, ggml_backend_sched_get_tensor_backend(sched_src, src_h));
            for (int64_t i0 = 0; i0 < n_tokens; ) {
                if (dev[i0]) {
                    i0++;
                    continue;
                }
                int64_t i1 = i0;
                while (i1 < n_tokens && !dev[i1]) {
                    i1++;
                }
                ggml_backend_tensor_set(embd, ubatch->embd + (size_t) i0*n_embd, (size_t) i0*row_bytes, (size_t) (i1 - i0)*row_bytes);
                i0 = i1;
            }
            embd_dev->rows.clear();
            embd_dev->src = nullptr;
        }
    }
}

bool llm_graph_input_embd::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= (!params.ubatch.token) || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (embd   &&   embd->ne[1] == params.ubatch.n_tokens);

    // the device rows are a different graph from a host input
    const bool want_dev = !device && params.tokens_dev && params.tokens_dev->use_dev(params.ubatch);
    res &= want_dev == dev_tokens;

    // a chained draft gathering its anchors from the stage is a different graph from one reading embd
    const bool want_idx = params.embd_dev && params.embd_dev->stage && params.cparams.mtp_chain > 1 && params.ubatch.embd;
    res &= want_idx == (idx != nullptr);

    const bool want_chain = !device && !want_dev && params.chain && params.chain->n_cm > 0 && params.ubatch.token &&
                            params.ubatch.n_tokens <= (uint32_t) params.chain->n_cm;
    res &= want_chain == (tok_in != nullptr);

    return res;
}

void llm_graph_input_embd_h::set_input(const llama_ubatch * ubatch) {
    const int64_t n_tokens = ubatch->n_tokens;

    if (ubatch->token) {
        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));
    } else {
        // note: mtmd embedding input goes through here
        GGML_ASSERT(ubatch->embd);
        GGML_ASSERT(n_embd == embd->ne[0]);

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(h));
    }

    // TODO: extend llama_ubatch to differentiate between token embeddings and hidden states
    //       for now, we assume that the hidden state is always provided as an embedding
    //       ref: https://github.com/ggml-org/llama.cpp/pull/23643
    if (ubatch->embd && h) {
        GGML_ASSERT(n_embd == h->ne[0]);

        ggml_backend_tensor_set(h, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(h));
    }
}

bool llm_graph_input_embd_h::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= (!params.ubatch.token) || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (embd   && embd->ne[1]   == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || !h || (h->ne[1] == params.ubatch.n_tokens);

    return res;
}

// the round's inputs for n_seq sequences, on the device, for as long as the context lives
static const llama_mtp_round_stage & mtp_round_ensure_dev(llama_mtp_round_state * st, ggml_backend_buffer_type_t buft, int64_t n_embd, int64_t n_verify, int64_t n_seq) {
    if (st->n_embd_dev != n_embd || st->n_verify_dev != n_verify) {
        st->free_stages();
        st->n_embd_dev   = n_embd;
        st->n_verify_dev = n_verify;
    }
    if ((int64_t) st->stages.size() < n_seq) {
        st->stages.resize(n_seq);
    }
    auto & sg = st->stages[n_seq - 1];
    if (sg.ctx_dev) {
        return sg;
    }

    ggml_init_params ip = { 4*ggml_tensor_overhead(), nullptr, true };
    sg.ctx_dev = ggml_init(ip);
    GGML_ASSERT(sg.ctx_dev);

    sg.c_ids_dev  = ggml_new_tensor_1d(sg.ctx_dev, GGML_TYPE_I32, n_verify*n_seq);
    sg.h_tgt_dev  = ggml_new_tensor_2d(sg.ctx_dev, GGML_TYPE_F32, n_embd, n_verify*n_seq);
    sg.h_pend_dev = ggml_new_tensor_2d(sg.ctx_dev, GGML_TYPE_F32, n_embd, n_seq);
    ggml_set_name(sg.c_ids_dev,  "mtp_round_c");
    ggml_set_name(sg.h_tgt_dev,  "mtp_round_h");
    ggml_set_name(sg.h_pend_dev, "mtp_round_hpend");

    sg.buf_dev = ggml_backend_alloc_ctx_tensors_from_buft(sg.ctx_dev, buft);
    GGML_ASSERT(sg.buf_dev && "failed to allocate the MTP round inputs");
    return sg;
}

// copies nbytes from src at src_offset into dst at dst_offset, on the device, ordered
// after the graph that wrote src
static void mtp_round_copy(ggml_backend_sched_t sched_src, ggml_backend_sched_t sched_dst, ggml_tensor * src, ggml_tensor * dst, size_t nbytes, size_t dst_offset, size_t src_offset) {
    GGML_ASSERT(src_offset + nbytes <= ggml_nbytes(src) && dst_offset + nbytes <= ggml_nbytes(dst));
    ggml_tensor s = *src;
    ggml_tensor d = *dst;
    s.ne[0] = (int64_t) (nbytes / ggml_type_size(src->type)); s.ne[1] = 1; s.ne[2] = 1; s.ne[3] = 1;
    s.nb[1] = nbytes; s.nb[2] = nbytes; s.nb[3] = nbytes;
    d.ne[0] = s.ne[0]; d.ne[1] = 1; d.ne[2] = 1; d.ne[3] = 1;
    d.nb[1] = nbytes; d.nb[2] = nbytes; d.nb[3] = nbytes;
    d.data = (char *) dst->data + dst_offset;
    s.data = (char *) src->data + src_offset;
    d.view_src = nullptr;
    s.view_src = nullptr;
    ggml_backend_t b_src = ggml_backend_sched_get_tensor_backend(sched_src, src);
    if (!b_src && src->view_src) {
        b_src = ggml_backend_sched_get_tensor_backend(sched_src, src->view_src);
    }
    ggml_backend_t b_dst = ggml_backend_sched_get_tensor_backend(sched_dst, dst);
    // a persistent tensor outside any graph shares its peer's backend; a graph's tensor always has its own
    GGML_ASSERT((b_src || sched_src == sched_dst) && "MTP round: no backend for a tensor of another context's graph");
    if (!b_src) b_src = b_dst;
    if (!b_dst) b_dst = b_src;
    GGML_ASSERT(b_src && b_dst);
    bool same = s.type == d.type;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        same = same && s.ne[i] == d.ne[i] && s.nb[i] == d.nb[i];
    }
    if (!same) {
        LLAMA_LOG_ERROR("%s: copy (same %d) src %s type %d ne %lld %lld %lld %lld nb %zu %zu %zu %zu | dst %s type %d ne %lld %lld %lld %lld nb %zu %zu %zu %zu\n", __func__, (int) same,
                src->name, (int) s.type, (long long) s.ne[0], (long long) s.ne[1], (long long) s.ne[2], (long long) s.ne[3], s.nb[0], s.nb[1], s.nb[2], s.nb[3],
                dst->name, (int) d.type, (long long) d.ne[0], (long long) d.ne[1], (long long) d.ne[2], (long long) d.ne[3], d.nb[0], d.nb[1], d.nb[2], d.nb[3]);
    }
    // say once what these are: ggml_backend_tensor_copy_async falls back to
    // synchronizing both backends when either side is not a CUDA buffer, and
    // that fallback waits for the target's whole forward pass
    static int said = 0;
    if (said < 4 && getenv("LLAMA_ROUND_TIMING") != nullptr) {
        said++;
        fprintf(stderr, "[round] round copy: src %s buffer '%s' backend '%s' | dst %s buffer '%s' backend '%s'\n",
                src->name,
                src->buffer ? ggml_backend_buffer_name(src->buffer) : "(none)",
                b_src ? ggml_backend_name(b_src) : "(none)",
                dst->name,
                dst->buffer ? ggml_backend_buffer_name(dst->buffer) : "(none)",
                b_dst ? ggml_backend_name(b_dst) : "(none)");
    }
    // both sides on one stream: the source's. A copy issued across streams
    // would make the destination's stream wait here, and the scheduler
    // synchronizes that stream before every graph, so the host would wait too.
    ggml_backend_tensor_copy_async(b_src, b_src, &s, &d);
}

// the destination's next graph waits for the source's work, on the device
static void mtp_round_order(ggml_backend_sched_t sched_dst, ggml_backend_t b_src) {
    ggml_backend_t b_dst = nullptr;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched_dst); ++i) {
        ggml_backend_t b = ggml_backend_sched_get_backend(sched_dst, i);
        if (ggml_backend_dev_type(ggml_backend_get_device(b)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            b_dst = b;
            break;
        }
    }
    if (!b_dst || !b_src || b_dst == b_src) {
        return;
    }
    typedef void (*order_fn_t)(ggml_backend_t, ggml_backend_t);
    auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(b_dst));
    auto order = (order_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_order_after");
    if (order) {
        order(b_dst, b_src);
    }
}

struct mtp_round_input_timing {
    bool on = getenv("LLAMA_ROUND_TIMING") != nullptr;
    double total = 0, copies = 0; int n = 0, host = 0;
    void add(double t, double c, bool is_host) {
        total += t; copies += c; n++; host += is_host ? 1 : 0;
        if (n % 100 == 0) {
            fprintf(stderr, "[round] mtp inputs calls %d-%d  set_input %.3f  of which device copies %.3f ms per call, host route %d/100\n",
                    n - 100, n, total / 100, copies / 100, host);
            total = copies = 0; host = 0;
        }
    }
};
static mtp_round_input_timing & mtp_round_input_timer() { static mtp_round_input_timing t; return t; }

void llm_graph_input_mtp_sample::set_input(const llama_ubatch * ubatch) {
    const int64_t n_s = par->ne[1];
    std::vector<float> p((size_t) 2*n_s), uu((size_t) n_s*K);
    for (int64_t s = 0; s < n_s; ++s) {
        // a greedy sequence's drafts: the most likely
        float    inv_t = 1e4f;
        float    top_p = 1.0f;
        uint32_t seed  = 0;
        const llama_seq_id seq = s < (int64_t) state->seqs.size() ? state->seqs[s].seq_id : -1;
        llama_sampler * chain = state->ctx_tgt && seq >= 0 ? state->ctx_tgt->get_sampler(seq) : nullptr;
        llama_sampler_batch_params bp;
        if (chain && llama_sampler_batch_params_get(chain, bp) && bp.temp > 0.0f) {
            inv_t = 1.0f/bp.temp;
            top_p = bp.top_p;
            seed  = llama_sampler_batch_seed(chain);
        }
        p[2*s + 0] = inv_t;
        p[2*s + 1] = top_p;
        const int32_t pos0 = ubatch->pos ? ubatch->pos[(size_t) s*V] : 0;
        for (int j = 0; j < K; ++j) {
            uu[(size_t) j*n_s + s] = llm_spec_uniform(seed, pos0 + j, 0x64726166u);
        }
    }
    ggml_backend_tensor_set(par, p.data(),  0, ggml_nbytes(par));
    ggml_backend_tensor_set(u,   uu.data(), 0, ggml_nbytes(u));
}

void llm_graph_input_mtp_round::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);
    const auto * st = state;
    GGML_ASSERT((int64_t) st->seqs.size() == n_seq);
    const size_t row_bytes = (size_t) n_embd * sizeof(float);
    const size_t es        = sizeof(int32_t);
    auto & rit = mtp_round_input_timer();
    const int64_t rit0 = rit.on ? ggml_time_us() : 0;
    double rit_copies = 0;
    bool   any_host   = false;

    llm_graph_result   * res_tgt   = nullptr;
    ggml_backend_sched_t sched_tgt = nullptr;
    ggml_tensor        * src_h     = nullptr;

    std::vector<float> offs(n_seq);

    // A round taking only its anchors from a verify whose sequences sit back to back on the target (every round of a
    // full batch): the hidden rows go over as one copy and the samples, rows of one argmax, as another. One copy per
    // sequence and kind costs the copy engine a few us each, before the drafter can start.
    if (st->anchors_only && st->ctx_tgt && !llama_host_off(LLAMA_HOST_OFF_GREEDY_BATCH)) {
        bool regular = true;
        const int32_t row0 = st->seqs[0].tgt_row;
        const int32_t out0 = st->seqs[0].tgt_out;
        for (int64_t s = 0; regular && s < n_seq; ++s) {
            const auto & q = st->seqs[s];
            regular = !q.c_ids && q.tgt_row == row0 + s*n_verify && q.tgt_out == out0 + s*n_verify;
        }
        llm_graph_result * rt = regular ? st->ctx_tgt->get_gf_res_last() : nullptr;
        ggml_tensor      * sh = rt ? rt->get_h_nextn() : nullptr;
        ggml_tensor      * c0 = rt && out0 >= 0 && (int64_t) rt->t_sampled.size() >= out0 + n_seq*n_verify ? rt->get_sampled_id(out0) : nullptr;
        regular = sh && c0 && c0->view_src && sh->ne[0] == n_embd && ggml_is_contiguous(sh) && row0 >= 0 &&
                  row0 + n_seq*n_verify <= sh->ne[1] &&
                  (!st->ctx_tgt->get_cparams().embeddings_nextn_masked || row0 == out0);
        for (int64_t i = 1; regular && i < n_seq*n_verify; ++i) {
            const ggml_tensor * ci = rt->get_sampled_id(out0 + i);
            regular = ci && ci->view_src == c0->view_src && ci->view_offs == c0->view_offs + i*es;
        }
        if (regular) {
            ggml_backend_sched_t sched_t = st->ctx_tgt->get_sched();
            const int64_t rc0 = rit.on ? ggml_time_us() : 0;
            mtp_round_copy(sched_t, sched, sh, h_tgt, row_bytes*n_verify*n_seq, 0, (size_t) row0*row_bytes);
            mtp_round_copy(sched_t, sched, c0->view_src, c_ids, es*n_verify*n_seq, 0, c0->view_offs);
            mtp_round_order(sched, ggml_backend_sched_get_tensor_backend(sched_t, sh));
            if (rit.on) { rit_copies += (ggml_time_us() - rc0) / 1000.0; }
            for (int64_t s = 0; s < n_seq; ++s) {
                offs[s] = (float) st->seqs[s].off;
            }
            if (off && off->buffer) {
                ggml_backend_tensor_set(off, offs.data(), 0, n_seq*sizeof(float));
            }
            if (rit.on) { rit.add((ggml_time_us() - rit0) / 1000.0, rit_copies, false); }
            return;
        }
    }

    for (int64_t s = 0; s < n_seq; ++s) {
        const auto & q = st->seqs[s];
        offs[s] = (float) q.off;
        const size_t h_off = (size_t) s*n_verify*row_bytes;
        const size_t c_off = (size_t) s*n_verify*es;
        if (q.c_ids) {
            GGML_ASSERT(q.h_tgt);
            any_host = true;
            ggml_backend_tensor_set(h_tgt, q.h_tgt, h_off, row_bytes * n_verify);
            if (q.h_pend) {
                ggml_backend_tensor_set(h_pend, q.h_pend, s*row_bytes, row_bytes);
            }
            ggml_backend_tensor_set(c_ids, q.c_ids, c_off, es * n_verify);
        } else {
            if (!res_tgt) {
                GGML_ASSERT(st->ctx_tgt);
                res_tgt   = st->ctx_tgt->get_gf_res_last();
                sched_tgt = st->ctx_tgt->get_sched();
                src_h     = res_tgt->get_h_nextn();
                GGML_ASSERT(src_h && src_h->ne[0] == n_embd && ggml_is_contiguous(src_h));
            }
            GGML_ASSERT(q.tgt_row >= 0 && q.tgt_row + n_verify <= src_h->ne[1]);
            // a masked h_nextn holds output rows only; the device route's verify rows are all outputs
            GGML_ASSERT(!st->ctx_tgt->get_cparams().embeddings_nextn_masked || q.tgt_row == q.tgt_out);
            GGML_ASSERT(q.tgt_out >= 0 && (int64_t) res_tgt->t_sampled.size() >= q.tgt_out + n_verify);
            const int64_t rit_c0 = rit.on ? ggml_time_us() : 0;
            mtp_round_copy(sched_tgt, sched, src_h, h_tgt, row_bytes * n_verify, h_off, (size_t) q.tgt_row*row_bytes);

            auto sampled_id = [&](int64_t i) {
                ggml_tensor * src_c = res_tgt->get_sampled_id(q.tgt_out + i);
                GGML_ASSERT(src_c && "MTP round: the target must sample on the backend");
                return src_c;
            };
            for (int64_t i = 0; i < n_verify; ) {
                ggml_tensor * src_c = sampled_id(i);
                // ids that sit back to back in one tensor (the batched greedy rows) go over as one copy
                int64_t n_run = 1;
                while (src_c->view_src != nullptr && i + n_run < n_verify && !llama_host_off(LLAMA_HOST_OFF_GREEDY_BATCH)) {
                    const ggml_tensor * next = sampled_id(i + n_run);
                    if (next->view_src != src_c->view_src || next->view_offs != src_c->view_offs + n_run*sizeof(int32_t)) {
                        break;
                    }
                    n_run++;
                }
                if (n_run > 1) {
                    mtp_round_copy(sched_tgt, sched, src_c->view_src, c_ids, n_run*es, c_off + i*es, src_c->view_offs);
                } else {
                    mtp_round_copy(sched_tgt, sched, src_c, c_ids, es, c_off + i*es);
                }
                i += n_run;
            }
            if (rit.on) { rit_copies += (ggml_time_us() - rit_c0) / 1000.0; }
        }
        if (!q.h_pend && !st->anchors_only) {
            GGML_ASSERT(st->hpend_dev && q.seq_id >= 0 && q.seq_id < st->hpend_dev->ne[1]);
            mtp_round_copy(sched, sched, st->hpend_dev, h_pend, row_bytes, s*row_bytes, (size_t) q.seq_id*row_bytes);
        }
    }
    if (src_h) {
        mtp_round_order(sched, ggml_backend_sched_get_tensor_backend(sched_tgt, src_h));
    }

    // a round taking only its anchors does not read off
    if (off && off->buffer) {
        ggml_backend_tensor_set(off, offs.data(), 0, n_seq*sizeof(float));
    }
    if (rit.on) { rit.add((ggml_time_us() - rit0) / 1000.0, rit_copies, any_host); }
}

bool llm_graph_input_mtp_round::can_reuse(const llm_graph_params & params) {
    bool res = true;
    res &= params.mtp_round == state;
    res &= (int64_t) state->stages.size() >= n_seq && state->stages[n_seq - 1].c_ids_dev == c_ids;
    res &= c_ids && c_ids->ne[0] == n_verify*n_seq;
    return res;
}

void llm_graph_input_pos::set_input(const llama_ubatch * ubatch) {
    if (device) {
        return;
    }
    if (ubatch->pos && pos) {
        const int64_t n_tokens = ubatch->n_tokens;

        if (ubatch->token && n_pos_per_embd == 4) {
            // in case we're using M-RoPE with text tokens, convert the 1D positions to 4D
            // the 3 first dims are the same, and 4th dim is all 0
            std::vector<llama_pos> pos_data(n_tokens*n_pos_per_embd);
            // copy the first dimension
            for (int i = 0; i < n_tokens; ++i) {
                pos_data[               i] = ubatch->pos[i];
                pos_data[    n_tokens + i] = ubatch->pos[i];
                pos_data[2 * n_tokens + i] = ubatch->pos[i];
                pos_data[3 * n_tokens + i] = 0; // 4th dim is 0
            }
            ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size()*ggml_element_size(pos));
        } else {
            ggml_backend_tensor_set(pos, ubatch->pos, 0, n_tokens*n_pos_per_embd*ggml_element_size(pos));
        }
    }
}

bool llm_graph_input_pos::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= pos->ne[0] == params.ubatch.n_tokens*n_pos_per_embd;

    return res;
}

void llm_graph_input_attn_temp::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && attn_scale) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(f_attn_temp_scale != 0.0f);
        GGML_ASSERT(n_attn_temp_floor_scale != 0);

        std::vector<float> attn_scale_data(n_tokens, 0.0f);
        for (int i = 0; i < n_tokens; ++i) {
            const float pos = ubatch->pos[i];
            attn_scale_data[i] = std::log(
                std::floor((pos + f_attn_temp_offset) / n_attn_temp_floor_scale) + 1.0
            ) * f_attn_temp_scale + 1.0;
        }

        ggml_backend_tensor_set(attn_scale, attn_scale_data.data(), 0, n_tokens*ggml_element_size(attn_scale));
    }
}

void llm_graph_input_pos_bucket::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(ggml_backend_buffer_is_host(pos_bucket->buffer));
        GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

        int32_t * data = (int32_t *) pos_bucket->data;

        for (int j = 0; j < n_tokens; ++j) {
            for (int i = 0; i < n_tokens; ++i) {
                data[j*n_tokens + i] = llama_relative_position_bucket(ubatch->pos[i], ubatch->pos[j], hparams.n_rel_attn_bkts, true);
            }
        }
    }
}

void llm_graph_input_pos_bucket_kv::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        mctx->set_input_pos_bucket(pos_bucket, ubatch);
    }
}

void llm_graph_input_out_ids::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(out_ids);

    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(out_ids->buffer));
    int32_t * data = (int32_t *) out_ids->data;

    if (n_outputs == n_tokens) {
        for (int i = 0; i < n_tokens; ++i) {
            data[i] = i;
        }

        return;
    }

    GGML_ASSERT(ubatch->output);

    int n_outputs = 0;

    for (int i = 0; i < n_tokens; ++i) {
        if (ubatch->output[i]) {
            data[n_outputs++] = i;
        }
    }
}

bool llm_graph_input_out_ids::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= n_outputs == params.n_outputs;

    return res;
}

void llm_graph_input_strm::set_input(const llama_ubatch * ubatch) {
    if (group >= 0 && gmctx) {
        ubatch = &gmctx->get_ubatch();
    }
    GGML_ASSERT((int64_t) ubatch->n_seqs_unq == n_seqs);

    const llama_seq_id s0 = n_span == n_stream ? 0 : ubatch->seq_id_unq[0];

    std::vector<int32_t> q(n_span, 0);
    std::vector<int32_t> o(n_seqs);
    for (int64_t s = 0; s < n_seqs; ++s) {
        const int64_t j = ubatch->seq_id_unq[s] - s0;
        GGML_ASSERT(j >= 0 && j < n_span);
        q[j] = (int32_t) s;
        o[s] = (int32_t) j;
    }

    ggml_backend_tensor_set(strm_q, q.data(), 0, q.size()*sizeof(int32_t));
    ggml_backend_tensor_set(strm_o, o.data(), 0, o.size()*sizeof(int32_t));
}

bool llm_graph_input_strm::can_reuse(const llm_graph_params & params) {
    if (group >= 0) {
        const auto * pm = dynamic_cast<const llama_kv_cache_iswa_context *>(params.mctx);
        if (!pm || pm->n_groups() <= (uint32_t) group) {
            return false;
        }
        gmctx = pm->get_group(group);
        return !params.cparams.kv_unified && (int64_t) gmctx->get_ubatch().n_seqs_unq == n_seqs;
    }
    // the span is the masks' to check
    return !params.cparams.kv_unified && (int64_t) params.ubatch.n_seqs_unq == n_seqs;
}

void llm_graph_input_mean::set_input(const llama_ubatch * ubatch) {
    if (cparams.embeddings   &&
       (cparams.pooling_type == LLAMA_POOLING_TYPE_MEAN ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_RANK )) {

        const int64_t n_tokens     = ubatch->n_tokens;
        const int64_t n_seq_tokens = ubatch->n_seq_tokens;
        const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

        GGML_ASSERT(mean);
        GGML_ASSERT(ggml_backend_buffer_is_host(mean->buffer));

        float * data = (float *) mean->data;
        memset(mean->data, 0, n_tokens*n_seqs_unq*ggml_element_size(mean));

        std::vector<uint64_t> sums(n_seqs_unq, 0);
        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                sums[seq_idx] += ubatch->n_seq_tokens;
            }
        }

        std::vector<float> div(n_seqs_unq, 0.0f);
        for (int s = 0; s < n_seqs_unq; ++s) {
            const uint64_t sum = sums[s];
            if (sum > 0) {
                div[s] = 1.0f/float(sum);
            }
        }

        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                for (int j = 0; j < n_seq_tokens; ++j) {
                    data[seq_idx*n_tokens + i + j] = div[seq_idx];
                }
            }
        }
    }
}

void llm_graph_input_cls::set_input(const llama_ubatch * ubatch) {
    const int64_t n_tokens     = ubatch->n_tokens;
    const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

    if (cparams.embeddings && (
        cparams.pooling_type == LLAMA_POOLING_TYPE_CLS  ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_RANK ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_LAST
    )) {
        GGML_ASSERT(cls);
        GGML_ASSERT(ggml_backend_buffer_is_host(cls->buffer));

        uint32_t * data = (uint32_t *) cls->data;
        memset(cls->data, 0, n_seqs_unq*ggml_element_size(cls));

        std::vector<int> target_pos(n_seqs_unq, -1);
        std::vector<int> target_row(n_seqs_unq, -1);

        const bool last = (
             cparams.pooling_type == LLAMA_POOLING_TYPE_LAST ||
            (cparams.pooling_type == LLAMA_POOLING_TYPE_RANK && (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_QWEN3VL)) // qwen3 reranking & embedding models use last token
        );

        for (int i = 0; i < n_tokens; ++i) {
            const llama_pos pos = ubatch->pos[i];

            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                if (
                    (target_pos[seq_idx] == -1) ||
                    ( last && pos >= target_pos[seq_idx]) ||
                    (!last && pos <  target_pos[seq_idx])
                ) {
                    target_pos[seq_idx] = pos;
                    target_row[seq_idx] = i;
                }
            }
        }

        for (int s = 0; s < n_seqs_unq; ++s) {
            if (target_row[s] >= 0) {
                data[s] = target_row[s];
            }
        }
    }
}

void llm_graph_input_rs::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    const int64_t n_rs = mctx->get_n_rs();

    if (s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(s_copy->buffer));
        int32_t * data = (int32_t *) s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->s_copy(i);
        }
    }
}

bool llm_graph_input_rs::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_recurrent_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= s_copy->ne[0] == mctx->get_n_rs();

    res &= s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= s_copy_extra->ne[0] == mctx->get_n_rs() - params.ubatch.n_seqs;

    res &= head == mctx->get_head();
    res &= rs_z == mctx->get_rs_z();

    return res;
}

void llm_graph_input_cross_embd::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (cross_embd && !cross->v_embd.empty()) {
        assert(cross_embd->type == GGML_TYPE_F32);

        ggml_backend_tensor_set(cross_embd, cross->v_embd.data(), 0, ggml_nbytes(cross_embd));
    }
}

template <typename T>
static void print_mask(const T * data, int64_t n_tokens, int64_t n_kv, int64_t n_swa, llama_swa_type swa_type) {
    LLAMA_LOG_DEBUG("%s: === Attention mask ===\n", __func__);
    const char * swa_type_str = "unknown";

    switch (swa_type) {
        case LLAMA_SWA_TYPE_NONE:      swa_type_str = "LLAMA_SWA_TYPE_NONE"; break;
        case LLAMA_SWA_TYPE_STANDARD:  swa_type_str = "LLAMA_SWA_TYPE_STANDARD"; break;
        case LLAMA_SWA_TYPE_CHUNKED:   swa_type_str = "LLAMA_SWA_TYPE_CHUNKED"; break;
        case LLAMA_SWA_TYPE_SYMMETRIC: swa_type_str = "LLAMA_SWA_TYPE_SYMMETRIC"; break;
    };

    LLAMA_LOG_DEBUG("%s: n_swa : %d, n_kv: %d, swa_type: %s\n", __func__, (int)n_swa, (int)n_kv, swa_type_str);
    LLAMA_LOG_DEBUG("%s: '0' = can attend, '∞' = masked\n", __func__);
    LLAMA_LOG_DEBUG("%s: Rows = query tokens, Columns = key/value tokens\n\n", __func__);

    LLAMA_LOG_DEBUG("    ");
    for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
        LLAMA_LOG_DEBUG("%2d", j);
    }
    LLAMA_LOG_DEBUG("\n");

    for (int i = 0; i < std::min((int64_t)20, n_tokens); ++i) {
        LLAMA_LOG_DEBUG(" %2d ", i);
        for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
            float val = llama_cast<float>(data[i * n_kv + j]);
            if (val == -INFINITY) {
                LLAMA_LOG_DEBUG(" ∞");
            } else {
                LLAMA_LOG_DEBUG(" 0");
            }
        }
        LLAMA_LOG_DEBUG("\n");
    }
}

void llm_graph_input_attn_no_cache::set_input(const llama_ubatch * ubatch) {
    const int64_t n_kv     = ubatch->n_tokens;
    const int64_t n_tokens = ubatch->n_tokens;

    const auto fill_mask = [&](auto * data, int64_t ne, int n_swa, llama_swa_type swa_type) {
        using T = std::remove_reference_t<decltype(*data)>;
        std::fill(data, data + ne, llama_cast<T>(-INFINITY));

        for (int i1 = 0; i1 < n_tokens; ++i1) {
            const llama_seq_id s1 = ubatch->seq_id[i1][0];
            const llama_pos    p1 = ubatch->pos[i1];

            const uint64_t idst = i1*n_kv;

            for (int i0 = 0; i0 < n_tokens; ++i0) {
                const llama_seq_id s0 = ubatch->seq_id[i0][0];
                const llama_pos p0    = ubatch->pos[i0];

                // mask different sequences
                if (s0 != s1) {
                    continue;
                }

                // mask future tokens
                if (cparams.causal_attn && p0 > p1) {
                    continue;
                }

                // apply SWA if any
                if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                    continue;
                }

                data[idst + i0] = llama_cast<T>(hparams.use_alibi ? -std::abs(p0 - p1) : 0.0f);
            }
        }

        if (debug) {
            print_mask(data, n_tokens, n_kv, n_swa, swa_type);
        }
    };

    GGML_ASSERT(self_kq_mask);
    GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask->buffer));
    if (self_kq_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) self_kq_mask->data, ggml_nelements(self_kq_mask), 0, LLAMA_SWA_TYPE_NONE);
    } else {
        fill_mask((float       *) self_kq_mask->data, ggml_nelements(self_kq_mask), 0, LLAMA_SWA_TYPE_NONE);
    }

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        GGML_ASSERT(self_kq_mask_swa);
        GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask_swa->buffer));
        if (self_kq_mask_swa->type == GGML_TYPE_F16) {
            fill_mask((ggml_fp16_t *) self_kq_mask_swa->data, ggml_nelements(self_kq_mask_swa), hparams.n_swa, hparams.swa_type);
        } else {
            fill_mask((float       *) self_kq_mask_swa->data, ggml_nelements(self_kq_mask_swa), hparams.n_swa, hparams.swa_type);
        }
    }
}

void llm_graph_input_attn_kv::set_input(const llama_ubatch * ubatch) {
    if (device) {
        return;
    }
    mctx->set_input_k_idxs(self_k_idxs, ubatch);
    mctx->set_input_v_idxs(self_v_idxs, ubatch);

    // the mask is left unallocated when the graph only stores K/V without attending
    // (e.g. DFlash's KV-injection pass)
    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }

    if (self_k_rot && self_k_rot->buffer) {
        mctx->set_input_k_rot(self_k_rot);
    }

    if (self_v_rot && self_v_rot->buffer) {
        mctx->set_input_v_rot(self_v_rot);
    }
}

bool llm_graph_input_attn_kv::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= can_reuse_kq_mask(self_kq_mask, mctx, params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_k::set_input(const llama_ubatch * ubatch) {
    mctx->set_input_k_idxs(self_k_idxs, ubatch);

    mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
}

bool llm_graph_input_attn_k::can_reuse(const llm_graph_params & params) {
    mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    return can_reuse_impl(params);
}

bool llm_graph_input_attn_k::can_reuse_impl(const llm_graph_params & params) {
    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(self_kq_mask, mctx, params.ubatch, params.cparams);

    return res;
}

llm_graph_input_attn_kv_msa::llm_graph_input_attn_kv_msa(
        const llama_hparams & hparams,
        const llama_cparams & cparams,
        const llama_kv_cache_msa_context * mctx) :
    llm_graph_input_attn_kv(hparams, cparams, mctx->get_base()),
    mctx_msa(mctx) {
}

void llm_graph_input_attn_kv_msa::set_input(const llama_ubatch * ubatch) {
    llm_graph_input_attn_kv::set_input(ubatch);

    if (self_k_idxs_idx) {
        mctx_msa->get_idx()->set_input_k_idxs(self_k_idxs_idx, ubatch);
    }
}

bool llm_graph_input_attn_kv_msa::can_reuse(const llm_graph_params & params) {
    mctx_msa = static_cast<const llama_kv_cache_msa_context *>(params.mctx);

    // the parent class operates on the base cache context
    this->mctx = mctx_msa->get_base();

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
    if (self_k_idxs_idx) {
        res &= self_k_idxs_idx->ne[0] == params.ubatch.n_tokens;
    }

    res &= can_reuse_kq_mask(self_kq_mask, this->mctx, params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_k_dsa::set_input(const llama_ubatch * ubatch) {
    mctx->get_mla()->set_input_k_idxs(self_k_idxs_mla, ubatch);

    mctx->get_mla()->set_input_kq_mask(self_kq_mask_mla, ubatch, cparams.causal_attn);

    mctx->get_lid()->set_input_k_idxs(self_k_idxs_lid, ubatch);

    mctx->get_lid()->set_input_kq_mask(self_kq_mask_lid, ubatch, cparams.causal_attn);

    // left unallocated when the indexer does not use the rotation
    if (self_k_rot_lid && self_k_rot_lid->buffer) {
        mctx->get_lid()->set_input_k_rot(self_k_rot_lid);
    }
}

bool llm_graph_input_attn_k_dsa::can_reuse(const llm_graph_params & params) {
    mctx = static_cast<const llama_kv_cache_dsa_context *>(params.mctx);

    return can_reuse_impl(params);
}

bool llm_graph_input_attn_k_dsa::can_reuse_impl(const llm_graph_params & params) {
    bool res = true;

    res &= self_k_idxs_mla->ne[0] == params.ubatch.n_tokens;
    res &= self_k_idxs_lid->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(self_kq_mask_mla, mctx->get_mla(), params.ubatch, params.cparams);
    res &= can_reuse_kq_mask(self_kq_mask_lid, mctx->get_lid(), params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_k_dsa_iswa::set_input(const llama_ubatch * ubatch) {
    inp_dsa->set_input(ubatch);
    inp_swa->set_input(ubatch);
}

bool llm_graph_input_attn_k_dsa_iswa::can_reuse(const llm_graph_params & params) {
    mctx = static_cast<const llama_kv_cache_dsa_iswa_context *>(params.mctx);

    inp_dsa->mctx = mctx->get_dsa();
    inp_swa->mctx = mctx->get_swa();

    bool res = true;

    res &= inp_dsa->can_reuse_impl(params);
    res &= inp_swa->can_reuse_impl(params);

    return res;
}

// rows [row0, row0 + n) of a 1D input, as an input of its own for a cache's setter
static ggml_tensor llm_graph_rows_1d(const ggml_tensor * t, int64_t row0, int64_t n) {
    ggml_tensor r = *t;
    r.ne[0] = n;
    r.nb[1] = r.nb[2] = r.nb[3] = n*t->nb[0];
    r.data  = (char *) t->data + row0*t->nb[0];
    return r;
}

// rows [row0, row0 + n_seqs*n_tps) of a mask, as the mask of n_seqs sequences of n_tps rows each
static ggml_tensor llm_graph_rows_mask(const ggml_tensor * t, int64_t row0, int64_t n_tps, int64_t n_seqs) {
    ggml_tensor r = *t;
    r.ne[1] = n_tps;
    r.ne[2] = 1;
    r.ne[3] = n_seqs;
    r.nb[2] = n_tps*t->nb[1];
    r.nb[3] = r.nb[2];
    r.data  = (char *) t->data + row0*t->nb[1];
    return r;
}

void llm_graph_input_attn_kv_iswa::set_input_varlen(const llama_ubatch * ubatch) {
    const int64_t n_tab = (varlen_tab->ne[0] - 1)/3;
    std::vector<int32_t> tab(varlen_tab->ne[0], 0);
    tab[0] = -(int32_t) n_tab;

    // each group's cache writes its rows' indices and mask rows in place; the table gets its sequences
    int64_t row0 = 0;
    int64_t s    = 0;
    for (uint32_t g = 0; g < mctx->n_groups(); ++g) {
        const auto * gm = mctx->get_group(g);
        const llama_ubatch & gu = gm->get_ubatch();
        const auto & strm = gm->get_base()->get_streams();
        const int64_t n      = gu.n_tokens;
        const int64_t n_seqs = (int64_t) strm.size();
        const int64_t n_tps  = n/n_seqs;
        GGML_ASSERT(n_tps*n_seqs == n && s + n_seqs <= n_tab);

        if (self_k_idxs && self_k_idxs->buffer) {
            ggml_tensor t = llm_graph_rows_1d(self_k_idxs, row0, n);
            gm->get_base()->set_input_k_idxs(&t, &gu);
            if (self_v_idxs) {
                t = llm_graph_rows_1d(self_v_idxs, row0, n);
                gm->get_base()->set_input_v_idxs(&t, &gu);
            }
        }
        if (self_kq_mask && self_kq_mask->buffer) {
            ggml_tensor t = llm_graph_rows_mask(self_kq_mask, row0, n_tps, n_seqs);
            gm->get_base()->set_input_kq_mask(&t, &gu, cparams.causal_attn, cparams.kq_mask_strict);
        }
        if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
            ggml_tensor t = llm_graph_rows_1d(self_k_idxs_swa, row0, n);
            gm->get_swa()->set_input_k_idxs(&t, &gu);
            if (self_v_idxs_swa) {
                t = llm_graph_rows_1d(self_v_idxs_swa, row0, n);
                gm->get_swa()->set_input_v_idxs(&t, &gu);
            }
        }
        if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
            ggml_tensor t = llm_graph_rows_mask(self_kq_mask_swa, row0, n_tps, n_seqs);
            gm->get_swa()->set_input_kq_mask(&t, &gu, cparams.causal_attn, cparams.kq_mask_strict);
        }

        for (int64_t k = 0; k < n_seqs; ++k, ++s) {
            tab[1 + 3*s + 0] = (int32_t) (row0 + k*n_tps);
            tab[1 + 3*s + 1] = (int32_t) n_tps;
            tab[1 + 3*s + 2] = (int32_t) strm[k];
        }
        row0 += n;
    }
    GGML_ASSERT(row0 == (int64_t) ubatch->n_tokens);

    if (varlen_tab->buffer) {
        ggml_backend_tensor_set(varlen_tab, tab.data(), 0, tab.size()*sizeof(int32_t));
    }

    if (self_k_rot && self_k_rot->buffer) {
        mctx->get_base()->set_input_k_rot(self_k_rot);
    }
    if (self_v_rot && self_v_rot->buffer) {
        mctx->get_base()->set_input_v_rot(self_v_rot);
    }
    if (self_k_rot_swa && self_k_rot_swa->buffer) {
        mctx->get_swa()->set_input_k_rot(self_k_rot_swa);
    }
    if (self_v_rot_swa && self_v_rot_swa->buffer) {
        mctx->get_swa()->set_input_v_rot(self_v_rot_swa);
    }
}

void llm_graph_input_attn_kv_iswa::set_input(const llama_ubatch * ubatch) {
    if (varlen_tab) {
        set_input_varlen(ubatch);
        return;
    }
    if (group >= 0) {
        ubatch = &mctx->get_ubatch();
    }
    if (kv_strm && kv_strm->buffer) {
        const auto & strm = mctx->get_base()->get_streams();
        GGML_ASSERT((int64_t) strm.size() == kv_strm->ne[0]);
        static_assert(sizeof(strm[0]) == sizeof(int32_t), "streams are I32");
        ggml_backend_tensor_set(kv_strm, strm.data(), 0, strm.size()*sizeof(int32_t));
    }

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->get_base()->set_input_k_idxs(self_k_idxs, ubatch);
        if (self_v_idxs) {
            mctx->get_base()->set_input_v_idxs(self_v_idxs, ubatch);
        }
    }

    // the kq mask guards on its own buffer: shared cells leave idxs unbacked while the mask stays live
    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->get_base()->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn, cparams.kq_mask_strict);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        mctx->get_swa()->set_input_k_idxs(self_k_idxs_swa, ubatch);
        if (self_v_idxs_swa) {
            mctx->get_swa()->set_input_v_idxs(self_v_idxs_swa, ubatch);
        }
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        mctx->get_swa()->set_input_kq_mask(self_kq_mask_swa, ubatch, cparams.causal_attn, cparams.kq_mask_strict);
    }

    if (self_k_rot && self_k_rot->buffer) {
        mctx->get_base()->set_input_k_rot(self_k_rot);
    }

    if (self_v_rot && self_v_rot->buffer) {
        mctx->get_base()->set_input_v_rot(self_v_rot);
    }

    if (self_k_rot_swa && self_k_rot_swa->buffer) {
        mctx->get_swa()->set_input_k_rot(self_k_rot_swa);
    }

    if (self_v_rot_swa && self_v_rot_swa->buffer) {
        mctx->get_swa()->set_input_v_rot(self_v_rot_swa);
    }
}

bool llm_graph_input_attn_kv_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_iswa_context *>(params.mctx);

    // the variable-length layout: the rows, the widest views and room in the table for the sequences
    if (varlen_tab) {
        this->mctx = mctx;
        int64_t n_kv     = 0;
        int64_t n_kv_swa = 0;
        if (!llm_graph_attn_varlen(params.cparams, params.hparams, mctx, &n_kv, &n_kv_swa)) {
            return false;
        }
        int64_t n_seqs = 0;
        for (uint32_t g = 0; g < mctx->n_groups(); ++g) {
            n_seqs += (int64_t) mctx->get_group(g)->get_base()->get_streams().size();
        }
        bool res = n_kv == n_kv_varlen && n_kv_swa == n_kv_varlen_swa && 1 + 3*n_seqs <= varlen_tab->ne[0];
        res &= !self_k_idxs  || self_k_idxs->ne[0]  == params.ubatch.n_tokens;
        res &= !self_kq_mask || self_kq_mask->ne[1] == params.ubatch.n_tokens;
        return res;
    }

    // a grouped forward: the parent has the groups' count to match, each child its group's view and shapes
    if (!groups.empty()) {
        this->mctx = mctx;
        return mctx->n_groups() == groups.size();
    }
    const llama_ubatch * ub = &params.ubatch;
    if (group >= 0) {
        if (mctx->n_groups() <= (uint32_t) group) {
            return false;
        }
        mctx = mctx->get_group(group);
        ub   = &mctx->get_ubatch();
    } else if (mctx->n_groups() > 0) {
        return false;
    }
    const llama_ubatch & ubatch_cur = *ub;

    this->mctx = mctx;

    bool res = true;

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        res &= self_k_idxs->ne[0] == ubatch_cur.n_tokens;
      //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    const bool per_seq = kv_strm != nullptr;
    if (per_seq) {
        res &= kv_strm->ne[0] == (int64_t) mctx->get_base()->get_streams().size();
    }

    if (self_kq_mask && self_kq_mask->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask, mctx->get_base(), ubatch_cur, params.cparams, per_seq);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        res &= self_k_idxs_swa->ne[0] == ubatch_cur.n_tokens;
      //res &= self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask_swa, mctx->get_swa(), ubatch_cur, params.cparams, per_seq);
    }

    return res;
}

void llm_graph_input_attn_k_iswa::set_input(const llama_ubatch * ubatch) {
    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->get_base()->set_input_k_idxs(self_k_idxs, ubatch);
    }

    // the kq mask guards on its own buffer: shared cells leave idxs unbacked while the mask stays live
    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->get_base()->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn, cparams.kq_mask_strict);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        mctx->get_swa()->set_input_k_idxs(self_k_idxs_swa, ubatch);
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        mctx->get_swa()->set_input_kq_mask(self_kq_mask_swa, ubatch, cparams.causal_attn, cparams.kq_mask_strict);
    }

    if (self_k_rot && self_k_rot->buffer) {
        mctx->get_base()->set_input_k_rot(self_k_rot);
    }

    if (self_k_rot_swa && self_k_rot_swa->buffer) {
        mctx->get_swa()->set_input_k_rot(self_k_rot_swa);
    }
}

bool llm_graph_input_attn_k_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
    }

    if (self_kq_mask && self_kq_mask->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask, mctx->get_base(), params.ubatch, params.cparams);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        res &= self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask_swa, mctx->get_swa(), params.ubatch, params.cparams);
    }

    return res;
}

static void dsv4_set_i64(ggml_tensor * dst, const std::vector<int64_t> & src) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->ne[0] == (int64_t) src.size());
    ggml_backend_tensor_set(dst, src.data(), 0, src.size()*ggml_element_size(dst));
}

static void dsv4_set_i32(ggml_tensor * dst, const std::vector<int32_t> & src) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->ne[0] == (int64_t) src.size());
    ggml_backend_tensor_set(dst, src.data(), 0, src.size()*ggml_element_size(dst));
}

static void dsv4_set_kq_mask(
        ggml_tensor * dst,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(n_stream > 0);
    GGML_ASSERT(n_tokens%n_stream == 0);
    GGML_ASSERT(dst->ne[0] == plan.n_kv);
    GGML_ASSERT(dst->ne[1] == (int64_t) n_tokens/n_stream);
    GGML_ASSERT(dst->ne[2] == 1);
    GGML_ASSERT(dst->ne[3] == n_stream);
    GGML_ASSERT((int64_t) plan.n_visible.size() == (int64_t) n_tokens);
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    if (dst->type == GGML_TYPE_F32) {
        float * data = (float *) dst->data;

        for (int64_t i = 0; i < (int64_t) n_tokens; ++i) {
            const int32_t n_visible = plan.n_visible[i];

            for (int64_t j = 0; j < dst->ne[0]; ++j) {
                data[i*dst->ne[0] + j] = j < n_visible ? 0.0f : -INFINITY;
            }
        }
    } else if (dst->type == GGML_TYPE_F16) {
        ggml_fp16_t * data = (ggml_fp16_t *) dst->data;
        const ggml_fp16_t fp16_ninf = llama_cast<ggml_fp16_t>(-INFINITY);
        const ggml_fp16_t fp16_zero = llama_cast<ggml_fp16_t>(0.0f);

        for (int64_t i = 0; i < (int64_t) n_tokens; ++i) {
            const int32_t n_visible = plan.n_visible[i];

            for (int64_t j = 0; j < dst->ne[0]; ++j) {
                data[i*dst->ne[0] + j] = j < n_visible ? fp16_zero : fp16_ninf;
            }
        }
    }
}

static ggml_tensor * dsv4_build_raw_kq_mask(
        ggml_context * ctx,
        const llama_kv_cache_dsv4_raw_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams,
        int64_t n_stream) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;

    GGML_ASSERT(n_stream > 0);
    GGML_ASSERT(n_tokens%n_stream == 0);

    const auto type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    ggml_tensor * res = ggml_new_tensor_4d(ctx, type, n_kv, n_tokens/n_stream, 1, n_stream);
    ggml_set_input(res);
    ggml_set_name(res, "attn_inp_kq_mask");

    return res;
}

static bool dsv4_can_reuse_raw_kq_mask(
        ggml_tensor * kq_mask,
        const llama_kv_cache_dsv4_raw_context * mctx,
        const llama_ubatch & ubatch,
        int64_t n_stream) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;

    GGML_ASSERT(n_stream > 0);

    bool res = true;

    res &= (kq_mask->ne[0] == n_kv);
    res &= (kq_mask->ne[1] == n_tokens/n_stream);
    res &= (kq_mask->ne[2] == 1);
    res &= (kq_mask->ne[3] == n_stream);

    return res;
}

static std::string dsv4_plan_positions(const std::vector<int32_t> & values) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            ss << ", ";
        }
        ss << values[i];
    }
    ss << "]";
    return ss.str();
}

static bool dsv4_compress_debug() {
    static const bool debug = []() {
        const char * env = getenv("LLAMA_DSV4_COMPRESS_DEBUG");
        return env && atoi(env) > 0;
    }();

    return debug;
}

static void dsv4_set_comp_inputs(
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        const char * name,
        bool debug,
        uint32_t n_tokens,
        int64_t n_stream) {
    dsv4_set_i32(inp.state_pos, plan.state_pos);
    dsv4_set_i32(inp.state_persist_src_idxs, plan.state_persist_src_idxs);
    dsv4_set_i32(inp.state_persist_dst_idxs, plan.state_persist_dst_idxs);
    dsv4_set_i32(inp.state_restore_src_idxs, plan.state_restore_src_idxs);
    dsv4_set_i32(inp.state_restore_dst_idxs, plan.state_restore_dst_idxs);
    dsv4_set_i32(inp.state_snapshot_src_idxs, plan.state_snapshot_src_idxs);
    dsv4_set_i32(inp.state_snapshot_dst_idxs, plan.state_snapshot_dst_idxs);
    dsv4_set_i32(inp.state_read_idxs, plan.state_read_idxs);
    dsv4_set_i64(inp.state_write_idxs, plan.state_write_idxs);
    dsv4_set_i32(inp.state_write_pos, plan.state_write_pos);
    dsv4_set_kq_mask(inp.kq_mask, plan, n_tokens, n_stream);

    if (debug || dsv4_compress_debug()) {
        LLAMA_LOG_INFO("%s: %s n_tokens=%u, n_stream=%d, state_persist_dst=%s, state_write_pos=%s\n",
                __func__, name, n_tokens, (int) n_stream,
                dsv4_plan_positions(plan.state_persist_dst_idxs).c_str(),
                dsv4_plan_positions(plan.state_write_pos).c_str());
    }
}

static bool dsv4_can_reuse_tensor_1d(ggml_tensor * t, int64_t ne0) {
    return (t == nullptr && ne0 == 0) || (t != nullptr && t->ne[0] == ne0);
}

static bool dsv4_can_reuse_kq_mask(
        ggml_tensor * t,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    if (plan.n_kv == 0) {
        return t == nullptr;
    }

    GGML_ASSERT(n_stream > 0);

    return t != nullptr &&
           t->ne[0] == plan.n_kv &&
           t->ne[1] == (int64_t) n_tokens/n_stream &&
           t->ne[2] == 1 &&
           t->ne[3] == n_stream;
}

static bool dsv4_can_reuse_comp_input(
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    bool res = true;
    res &= dsv4_can_reuse_tensor_1d(inp.state_pos, plan.state_pos.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_persist_src_idxs, plan.state_persist_src_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_persist_dst_idxs, plan.state_persist_dst_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_restore_src_idxs, plan.state_restore_src_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_restore_dst_idxs, plan.state_restore_dst_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_snapshot_src_idxs, plan.state_snapshot_src_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_snapshot_dst_idxs, plan.state_snapshot_dst_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_read_idxs, plan.state_read_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_write_idxs, plan.state_write_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_write_pos, plan.state_write_pos.size());
    res &= dsv4_can_reuse_kq_mask(inp.kq_mask, plan, n_tokens, n_stream);

    return res;
}

static ggml_tensor * dsv4_build_input_1d(
        ggml_context * ctx,
        ggml_type type,
        int64_t ne0,
        const std::string & name) {
    if (ne0 == 0) {
        return nullptr;
    }

    ggml_tensor * res = ggml_new_tensor_1d(ctx, type, ne0);
    ggml_set_input(res);
    ggml_set_name(res, name.c_str());

    return res;
}

static void dsv4_build_comp_inputs(
        ggml_context * ctx,
        llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        const char * name,
        const llama_cparams & cparams,
        int64_t n_stream) {
    inp.state_pos = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_pos.size(), std::string("dsv4_") + name + "_state_pos");
    inp.state_persist_src_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_persist_src_idxs.size(), std::string("dsv4_") + name + "_state_persist_src_idxs");
    inp.state_persist_dst_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_persist_dst_idxs.size(), std::string("dsv4_") + name + "_state_persist_dst_idxs");
    inp.state_restore_src_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_restore_src_idxs.size(), std::string("dsv4_") + name + "_state_restore_src_idxs");
    inp.state_restore_dst_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_restore_dst_idxs.size(), std::string("dsv4_") + name + "_state_restore_dst_idxs");
    inp.state_snapshot_src_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_snapshot_src_idxs.size(), std::string("dsv4_") + name + "_state_snapshot_src_idxs");
    inp.state_snapshot_dst_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_snapshot_dst_idxs.size(), std::string("dsv4_") + name + "_state_snapshot_dst_idxs");
    inp.state_read_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_read_idxs.size(), std::string("dsv4_") + name + "_state_read_idxs");
    inp.state_write_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I64, plan.state_write_idxs.size(), std::string("dsv4_") + name + "_state_write_idxs");
    inp.state_write_pos = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_write_pos.size(), std::string("dsv4_") + name + "_state_write_pos");

    if (plan.n_kv > 0) {
        const int64_t n_tokens = (int64_t) plan.n_visible.size();

        GGML_ASSERT(n_stream > 0);
        GGML_ASSERT(n_tokens%n_stream == 0);

        inp.kq_mask = ggml_new_tensor_4d(ctx, (strcmp(name, "lid") != 0 && cparams.flash_attn) || (strcmp(name, "lid") == 0 && cparams.fused_lid) ? GGML_TYPE_F16 : GGML_TYPE_F32, plan.n_kv, n_tokens/n_stream, 1, n_stream);
        ggml_set_input(inp.kq_mask);
        ggml_set_name(inp.kq_mask, (std::string("dsv4_") + name + "_kq_mask").c_str());
    }
}

void llm_graph_input_dsv4_raw::set_input(const llama_ubatch * ubatch) {
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->set_input_k_idxs(self_k_idxs);
    }

    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }

    if (self_k_rot) {
        mctx->set_input_k_rot(self_k_rot);
    }
}

void llm_graph_input_dsv4::set_input(const llama_ubatch * ubatch) {
    const auto & plan_csa = mctx->get_csa_plan(*ubatch);
    const auto & plan_hca = mctx->get_hca_plan(*ubatch);
    const auto & plan_lid = mctx->get_lid_plan(*ubatch);
    const int64_t n_stream = plan_csa.n_stream;

    inp_raw->mctx = mctx->get_raw();
    inp_raw->set_input(ubatch);

    dsv4_set_comp_inputs(inp_csa, plan_csa, "csa", debug > 0, ubatch->n_tokens, n_stream);
    dsv4_set_comp_inputs(inp_hca, plan_hca, "hca", debug > 0, ubatch->n_tokens, n_stream);
    dsv4_set_comp_inputs(inp_lid, plan_lid, "lid", debug > 0, ubatch->n_tokens, n_stream);

    if (inp_csa.k_rot && inp_csa.k_rot->buffer) {
        mctx->get_csa()->set_input_k_rot(inp_csa.k_rot);
    }

    if (inp_hca.k_rot && inp_hca.k_rot->buffer) {
        mctx->get_hca()->set_input_k_rot(inp_hca.k_rot);
    }

    if (inp_lid.k_rot && inp_lid.k_rot->buffer) {
        mctx->get_lid()->set_input_k_rot(inp_lid.k_rot);
    }
}

bool llm_graph_input_dsv4::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_dsv4_context *>(params.mctx);

    this->mctx = mctx;
    inp_raw->mctx = mctx->get_raw();

    bool res = true;

    const auto & plan_csa = mctx->get_csa_plan(params.ubatch);
    const auto & plan_hca = mctx->get_hca_plan(params.ubatch);
    const auto & plan_lid = mctx->get_lid_plan(params.ubatch);
    const int64_t n_stream = plan_csa.n_stream;

    const auto * raw_ctx = mctx->get_raw();
    inp_raw->mctx = raw_ctx;

    if (inp_raw->self_k_idxs && inp_raw->self_k_idxs->buffer) {
        res &= inp_raw->self_k_idxs->ne[0] == raw_ctx->get_n_write();
    }
    if (inp_raw->self_kq_mask && inp_raw->self_kq_mask->buffer) {
        res &= dsv4_can_reuse_raw_kq_mask(inp_raw->self_kq_mask, raw_ctx, params.ubatch, n_stream);
    }

    res &= dsv4_can_reuse_comp_input(inp_csa, plan_csa, params.ubatch.n_tokens, n_stream);
    res &= dsv4_can_reuse_comp_input(inp_hca, plan_hca, params.ubatch.n_tokens, n_stream);
    res &= dsv4_can_reuse_comp_input(inp_lid, plan_lid, params.ubatch.n_tokens, n_stream);

    return res;
}

void llm_graph_input_attn_cross::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(cross_kq_mask);

    const int64_t n_enc    = cross_kq_mask->ne[0];
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(cross_kq_mask->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    const auto fill_mask = [&](auto * data) {
        using T = std::remove_reference_t<decltype(*data)>;
        for (int i = 0; i < n_tokens; ++i) {
            GGML_ASSERT(!cross->seq_ids_enc.empty() && "llama_encode must be called first");
            for (int j = 0; j < n_enc; ++j) {
                float f = -INFINITY;

                for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                    const llama_seq_id seq_id = ubatch->seq_id[i][s];

                    if (cross->seq_ids_enc[j].find(seq_id) != cross->seq_ids_enc[j].end()) {
                        f = 0.0f;
                    }
                }

                data[i*n_enc + j] = llama_cast<T>(f);
            }
        }
    };

    if (cross_kq_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) cross_kq_mask->data);
    } else {
        fill_mask((float *) cross_kq_mask->data);
    }
}

void llm_graph_input_mem_hybrid::set_input(const llama_ubatch * ubatch) {
    // a loop body selects its rows and mask on the device
    if (!inp_attn->device) {
        mctx->get_attn()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);
        mctx->get_attn()->set_input_v_idxs(inp_attn->self_v_idxs, ubatch);

        mctx->get_attn()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);
    }

    if (inp_attn->self_k_rot) {
        mctx->get_attn()->set_input_k_rot(inp_attn->self_k_rot);
    }

    if (inp_attn->self_v_rot) {
        mctx->get_attn()->set_input_v_rot(inp_attn->self_v_rot);
    }

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, mctx->get_attn(), params.ubatch, params.cparams);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

// TODO: Hybrid input classes are a bit redundant.
// Instead of creating a hybrid input, the graph can simply create 2 separate inputs.
// Refactoring is required in the future.
void llm_graph_input_mem_hybrid_k::set_input(const llama_ubatch * ubatch) {
    mctx->get_attn()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);

    mctx->get_attn()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid_k::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, mctx->get_attn(), params.ubatch, params.cparams);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

void llm_graph_input_mem_hybrid_iswa::set_input(const llama_ubatch * ubatch) {
    const auto * attn_ctx = mctx->get_attn();

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        attn_ctx->get_base()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);
        attn_ctx->get_base()->set_input_v_idxs(inp_attn->self_v_idxs, ubatch);
    }

    if (inp_attn->self_kq_mask && inp_attn->self_kq_mask->buffer) {
        attn_ctx->get_base()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (inp_attn->self_k_idxs_swa && inp_attn->self_k_idxs_swa->buffer) {
        attn_ctx->get_swa()->set_input_k_idxs(inp_attn->self_k_idxs_swa, ubatch);
        attn_ctx->get_swa()->set_input_v_idxs(inp_attn->self_v_idxs_swa, ubatch);
    }

    if (inp_attn->self_kq_mask_swa && inp_attn->self_kq_mask_swa->buffer) {
        attn_ctx->get_swa()->set_input_kq_mask(inp_attn->self_kq_mask_swa, ubatch, cparams.causal_attn);
    }

    if (inp_attn->self_k_rot) {
        attn_ctx->get_base()->set_input_k_rot(inp_attn->self_k_rot);
    }

    if (inp_attn->self_v_rot) {
        attn_ctx->get_base()->set_input_v_rot(inp_attn->self_v_rot);
    }

    if (inp_attn->self_k_rot_swa) {
        attn_ctx->get_swa()->set_input_k_rot(inp_attn->self_k_rot_swa);
    }

    if (inp_attn->self_v_rot_swa) {
        attn_ctx->get_swa()->set_input_v_rot(inp_attn->self_v_rot_swa);
    }

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    const auto * attn_ctx = mctx->get_attn();

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
      //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, attn_ctx->get_base(), params.ubatch, params.cparams);

    // swa tensors may not be allocated if there are no SWA attention layers
    if (inp_attn->self_k_idxs_swa && inp_attn->self_k_idxs_swa->buffer) {
        res &= inp_attn->self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
      //res &= inp_attn->self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask_swa, attn_ctx->get_swa(), params.ubatch, params.cparams);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

void llm_graph_input_sampling::set_input(const llama_ubatch * ubatch) {
    if (batch_key != 0) {
        // the batched form: every row's parameters, draw, penalty window and the batch positions of the ids it takes
        // in before it, each input in one copy
        const int64_t n_rows = inp_par->ne[1];
        constexpr int NP = 12;
        std::vector<float>   par((size_t) NP*n_rows, 0.0f);
        std::vector<int32_t> dpos(inp_dpos ? (size_t) n_rows : 0, 0);
        std::vector<float>   hist(inp_hist  ? (size_t) batch_lw*n_rows : 0, -1.0f);
        std::vector<int32_t> ppos(inp_ppos  ? (size_t) batch_kp*n_rows : 0, 0);
        std::vector<float>   pmask(inp_pmask ? (size_t) batch_kp*n_rows : 0, 0.0f);
        std::vector<float>   btok(inp_btok  ? (size_t) batch_nb*n_rows : 0, -1.0f);
        std::vector<float>   bval(inp_bval  ? (size_t) batch_nb*n_rows : 0, 0.0f);
        std::map<llama_seq_id, std::vector<std::pair<int64_t, uint32_t>>> rows; // seq -> (row, batch position)
        int64_t r = 0;
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            if (ubatch->output[i]) {
                rows[ubatch->seq_id[i][0]].push_back({ r++, i });
            }
        }
        GGML_ASSERT(r == n_rows);

        // the drafter's last round's distributions ([2*KQ, K, S], llm_graph_result::t_mtp_q) go over as one copy, and a
        // row that checks a draft the device drew gathers its own (build_sampling); any other draft is its own
        int32_t        q_K  = 0;
        int64_t        q_S  = 0;
        bool           q_ok = false;
        std::vector<int32_t> qidx(inp_qidx ? (size_t) n_rows : 0, 0);
        const llama_tokens_dev_state * td = tokens_dev;
        if (inp_qstage) {
            ggml_tensor * qs = td && td->src && !td->spans.empty() ? td->src->get_mtp_q_pending() : nullptr;
            if (qs && qs->ne[0] == inp_qstage->ne[0] && qs->ne[1]*qs->ne[2] <= inp_qstage->ne[1] && sched) {
                typedef void (*copy_2d_fn_t)(ggml_backend_t, void *, size_t, const void *, size_t, size_t, size_t);
                ggml_backend_t b_q   = ggml_backend_sched_get_tensor_backend(sched, inp_qstage);
                ggml_backend_t b_src = ggml_backend_sched_get_tensor_backend(td->src->get_sched(), qs);
                copy_2d_fn_t copy_2d = b_q ? (copy_2d_fn_t) ggml_backend_reg_get_proc_address(
                        ggml_backend_dev_backend_reg(ggml_backend_get_device(b_q)), "ggml_backend_cuda_copy_2d_async") : nullptr;
                if (copy_2d && b_src) {
                    // on the drafter's stream, as its drafted ids: ahead of its next round, and the target waits for it
                    const size_t n_bytes = ggml_nbytes(qs);
                    copy_2d(b_src, inp_qstage->data, n_bytes, qs->data, n_bytes, n_bytes, 1);
                    mtp_round_order(td->sched_dst, b_src);
                    q_K  = (int32_t) qs->ne[1];
                    q_S  = qs->ne[2];
                    q_ok = true;
                }
            }
            if (!q_ok) {
                std::vector<float> zeros((size_t) ggml_nelements(inp_qstage), 0.0f);
                ggml_backend_tensor_set(inp_qstage, zeros.data(), 0, ggml_nbytes(inp_qstage));
            }
        }

        std::vector<llama_token> window;
        for (const auto & [seq, rs] : rows) {
            llama_sampler * chain = samplers.at(seq);
            llama_sampler_batch_params bp;
            const bool greedy = llama_sampler_is_backend_greedy(chain);
            if (greedy) {
                bp.temp = 0.0f;
                window.clear();
            } else {
                GGML_ASSERT(llama_sampler_batch_params_get(chain, bp));
                llama_sampler_batch_window(chain, window);
            }
            const int64_t n_win = (int64_t) window.size();
            for (size_t j = 0; j < rs.size(); ++j) {
                const int64_t row = rs[j].first;
                float * pr = par.data() + NP*row;
                pr[0] = bp.top_p;
                pr[1] = bp.min_p > 0.0f ? logf(bp.min_p) : -1e30f;
                // temperature 0 takes the top candidate: a temperature of 1 for the ops, and the draw's flag
                pr[2] = bp.temp > 0.0f ? bp.temp : 1.0f;
                pr[3] = greedy ? 0.0f : llama_sampler_batch_uniform(chain);
                pr[4] = 1.0f/bp.penalty_repeat - 1.0f;
                pr[5] = bp.penalty_repeat - 1.0f;
                pr[6] = bp.penalty_freq;
                pr[7] = bp.penalty_present;
                pr[8] = bp.temp > 0.0f ? 0.0f : 1.0f;
                if (inp_dpos) {
                    const uint32_t i = rs[j].second;
                    const bool drafted = llm_spec_drafted(*ubatch, i);
                    dpos[row] = (int32_t) (drafted ? i + 1 : i);
                    pr[9]  = drafted ? 1.0f : 0.0f;
                    pr[10] = llm_spec_uniform(greedy ? 0u : llama_sampler_batch_seed(chain), ubatch->pos[i], 0x72657364u);
                    // the device drew the draft at pos + 1: a span of this sequence holds it (tokens_dev_row)
                    int32_t qi = -1;
                    if (q_ok && drafted) {
                        for (const auto & sp : td->spans) {
                            const int32_t jd = ubatch->pos[i] - sp.pos0;
                            if (sp.seq == seq && jd >= 0 && jd < sp.n && jd < q_K && sp.col >= 0 && sp.col < q_S) {
                                qi = sp.col*q_K + jd;
                                break;
                            }
                        }
                    }
                    qidx[row] = qi >= 0 ? qi : 0;
                    pr[11]    = qi >= 0 ? 1.0f : 0.0f;
                }
                if (bp.penalty_last_n > 0 && inp_hist) {
                    // row j takes in j ids before it, which push the window's oldest out
                    const int64_t n_drop = std::clamp<int64_t>(n_win + (int64_t) j - bp.penalty_last_n, 0, std::min<int64_t>((int64_t) j, n_win));
                    int64_t h = 0;
                    for (int64_t t = n_drop; t < n_win && h < batch_lw; ++t) {
                        hist[(size_t) row*batch_lw + h++] = (float) window[t];
                    }
                    for (size_t t = 1; t <= j && (int64_t) t <= batch_kp; ++t) {
                        ppos [(size_t) row*batch_kp + t - 1] = (int32_t) rs[t].second;
                        pmask[(size_t) row*batch_kp + t - 1] = 1.0f;
                    }
                }
                for (size_t b = 0; b < bp.bias.size() && (int64_t) b < batch_nb; ++b) {
                    btok[(size_t) row*batch_nb + b] = (float) bp.bias[b].token;
                    bval[(size_t) row*batch_nb + b] = std::isinf(bp.bias[b].bias) ? -1e30f : bp.bias[b].bias;
                }
            }
        }
        ggml_backend_tensor_set(inp_par, par.data(), 0, ggml_nbytes(inp_par));
        if (inp_dpos)  ggml_backend_tensor_set(inp_dpos, dpos.data(), 0, ggml_nbytes(inp_dpos));
        if (inp_qidx)  ggml_backend_tensor_set(inp_qidx, qidx.data(), 0, ggml_nbytes(inp_qidx));
        if (inp_hist)  ggml_backend_tensor_set(inp_hist,  hist.data(),  0, ggml_nbytes(inp_hist));
        if (inp_ppos)  ggml_backend_tensor_set(inp_ppos,  ppos.data(),  0, ggml_nbytes(inp_ppos));
        if (inp_pmask) ggml_backend_tensor_set(inp_pmask, pmask.data(), 0, ggml_nbytes(inp_pmask));
        if (inp_btok)  ggml_backend_tensor_set(inp_btok,  btok.data(),  0, ggml_nbytes(inp_btok));
        if (inp_bval)  ggml_backend_tensor_set(inp_bval,  bval.data(),  0, ggml_nbytes(inp_bval));
        return;
    }

    // set the inputs only for the active samplers in the current ubatch
    std::unordered_set<llama_seq_id> active_samplers;
    for (uint32_t i = 0; i < ubatch->n_tokens; i++) {
        if (ubatch->output[i]) {
            llama_seq_id seq_id = ubatch->seq_id[i][0];
            active_samplers.insert(seq_id);
        }
    }

    for (auto seq_id : active_samplers) {
        if (samplers.find(seq_id) == samplers.end()) {
            continue;
        }

        auto & sampler = samplers[seq_id];

        if (sampler->iface->backend_set_input) {
            sampler->iface->backend_set_input(sampler);
        }
    }
}

bool llm_graph_input_sampling::can_reuse(const llm_graph_params & params) {
    if (batch_key != 0 && params.samplers_batch == batch_key) {
        // the parameters are inputs: the step's samplers fill them
        samplers = params.samplers;
        return true;
    }
    if (greedy && params.samplers_greedy) {
        // the step's samplers, for set_input: the graph reads none of them
        samplers = params.samplers;
        return true;
    }

    if (samplers.size() != params.samplers.size()) {
        return false;
    }

    for (const auto & [seq_id, sampler] : params.samplers) {
        if (samplers[seq_id] != sampler) {
            return false;
        }
    }

    return true;
}

//
// llm_graph_result
//

llm_graph_result::llm_graph_result(int64_t max_nodes) : max_nodes(max_nodes) {
    reset();

    const char * LLAMA_GRAPH_RESULT_DEBUG = getenv("LLAMA_GRAPH_RESULT_DEBUG");
    debug = LLAMA_GRAPH_RESULT_DEBUG ? atoi(LLAMA_GRAPH_RESULT_DEBUG) : 0;
}

int64_t llm_graph_result::get_max_nodes() const {
    return max_nodes;
}

void llm_graph_result::note_sched_tensors() {
    sched_tensors.clear();
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_compute.get()); t; t = ggml_get_next_tensor(ctx_compute.get(), t)) {
        if (t->buffer && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
            sched_tensors.push_back(t);
        }
    }
}

void llm_graph_result::snapshot_srcs() {
    const int n = ggml_graph_n_nodes(gf);
    src_snapshot.resize((size_t) n*GGML_MAX_SRC);
    for (int i = 0; i < n; ++i) {
        const ggml_tensor * node = ggml_graph_node(gf, i);
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            src_snapshot[(size_t) i*GGML_MAX_SRC + j] = node->src[j];
        }
    }
}

void llm_graph_result::restore_srcs() {
    const int n = ggml_graph_n_nodes(gf);
    if (src_snapshot.size() != (size_t) n*GGML_MAX_SRC) {
        return;
    }
    for (int i = 0; i < n; ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            node->src[j] = src_snapshot[(size_t) i*GGML_MAX_SRC + j];
        }
    }
}

void llm_graph_result::clear_sched_tensors() {
    for (ggml_tensor * t : sched_tensors) {
        t->data   = nullptr;
        t->buffer = nullptr;
    }
}

void llm_graph_result::reset() {
    sched_tensors.clear();
    mm_seqs_tag   = 0;
    t_inp_tokens  = nullptr;
    t_inp_embd    = nullptr;
    t_logits      = nullptr;
    t_embd        = nullptr;
    t_embd_pooled = nullptr;
    t_h_nextn     = nullptr;

    t_layer_inp.resize(LLAMA_MAX_LAYERS + 1);
    std::fill(t_layer_inp.begin(), t_layer_inp.end(), nullptr);

    t_sampled.clear();
    t_sampled_probs.clear();
    t_sampled_logits.clear();
    t_candidates.clear();
    t_mtp_ids.clear();
    t_mtp_ids_all = nullptr;
    t_mtp_q       = nullptr;
    t_mtp_hpend = nullptr;
    draft_head_unused = false;
    logits_internal   = false;

    chain_ok  = false;
    chain_bad = false;

    n_step_states = 0;

    params = {};

    inputs.clear();
    fused_nodes.clear();

    buf_compute_meta.resize(ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false));

    ggml_init_params params = {
        /*.mem_size   =*/ buf_compute_meta.size(),
        /*.mem_buffer =*/ buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    ctx_compute.reset(ggml_init(params));

    gf = ggml_new_graph_custom(ctx_compute.get(), max_nodes, false);
}

ggml_tensor * llm_graph_result::get_sampled_id(int64_t i) const {
    ggml_tensor * id = i < (int64_t) t_sampled.size() ? t_sampled[i] : nullptr;
    if (id == nullptr && i < (int64_t) t_candidates.size() && t_candidates[i] && ggml_nelements(t_candidates[i]) == 1) {
        id = t_candidates[i]; // the greedy sampler leaves one candidate
    }
    return id && id->type == GGML_TYPE_I32 ? id : nullptr;
}

void llm_graph_result::set_inputs(const llama_ubatch * ubatch) {
    // LLAMA_ROUND_TIMING=1 times each input class, printed every 200 calls per token count
    static const bool timing = getenv("LLAMA_ROUND_TIMING") != nullptr;
    if (!timing || ubatch->n_tokens <= 8) {
        for (auto & input : inputs) {
            input->set_input(ubatch);
        }
        return;
    }
    static std::map<std::pair<int, std::string>, std::pair<double, int>> acc;
    static std::map<int, int> calls;
    for (auto & input : inputs) {
        const int64_t t0 = ggml_time_us();
        input->set_input(ubatch);
        auto & a = acc[{(int) ubatch->n_tokens, typeid(*input).name()}];
        a.first += (ggml_time_us() - t0) / 1000.0;
        a.second++;
    }
    if (++calls[(int) ubatch->n_tokens] % 200 == 0) {
        for (auto & kv : acc) {
            if (kv.first.first == (int) ubatch->n_tokens && kv.second.second > 0) {
                fprintf(stderr, "[round] set_input T=%d %-40s %.3f ms per call (%d calls)\n", kv.first.first, kv.first.second.c_str(), kv.second.first / kv.second.second, kv.second.second);
                kv.second = {0.0, 0};
            }
        }
    }
}

void llm_graph_result::set_outputs(const llm_graph_params & params) {
    if (t_logits != nullptr && !logits_internal) {
        ggml_set_output(t_logits);
    }
    if (t_embd != nullptr) {
        ggml_set_output(t_embd);
    }
    if (t_embd_pooled != nullptr) {
        ggml_set_output(t_embd_pooled);
    }
    if (t_h_nextn != nullptr) {
        ggml_set_output(t_h_nextn);
    }
    {
        const auto & embeddings_layer_inp = params.cparams.embeddings_layer_inp;
        for (size_t il = 0; il < embeddings_layer_inp.size(); ++il) {
            if (embeddings_layer_inp[il]) {
                GGML_ASSERT(t_layer_inp[il] != nullptr && "layer input tensor is null");
                ggml_set_output(t_layer_inp[il]);
            }
        }
    }
    for (auto * tensor : t_sampled) {
        if (tensor != nullptr) {
            ggml_set_output(tensor);
        }
    }
    for (auto * tensor : t_sampled_probs) {
        if (tensor != nullptr) {
            ggml_set_output(tensor);
        }
    }
    for (auto * tensor : t_sampled_logits) {
        if (tensor != nullptr) {
            ggml_set_output(tensor);
        }
    }
    for (auto * tensor : t_candidates) {
        if (tensor != nullptr) {
            ggml_set_output(tensor);
        }
    }
}

bool llm_graph_result::can_reuse(const llm_graph_params & params) {
    reuse_blocker = nullptr;
    if (!this->params.allow_reuse(params)) {
        reuse_blocker = "graph parameters";
        if (debug > 1) {
            LLAMA_LOG_DEBUG("%s: cannot reuse graph due to incompatible graph parameters\n", __func__);
        }

        return false;
    }

    if (debug > 1) {
        LLAMA_LOG_DEBUG("%s: checking compatibility of %d inputs:\n", __func__, (int) inputs.size());
    }

    bool res = true;

    for (auto & input : inputs) {
        const bool cur = input->can_reuse(params);

        if (!cur) {
            reuse_blocker = typeid(*input).name();
        }

        if (debug > 1) {
            LLAMA_LOG_DEBUG("%s: can_reuse = %d\n", typeid(*input).name(), cur);
        }

        res = res && cur;
    }

    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s: can reuse graph = %d\n", __func__, res);
    }

    return res;
}

llm_graph_input_i * llm_graph_result::add_input(llm_graph_input_ptr input) {
    inputs.emplace_back(std::move(input));
    return inputs.back().get();
}

void llm_graph_result::add_fused_node(llm_graph_fused_node result) {
    fused_nodes.push_back(result);
}

void llm_graph_result::set_params(const llm_graph_params & params) {
    this->params = params;
}

//
// llm_graph_context
//

llm_graph_context::llm_graph_context(const llm_graph_params & params) :
    arch             (params.arch),
    hparams          (params.hparams),
    cparams          (params.cparams),
    ubatch           (params.ubatch),
    n_embd           (hparams.n_embd),
    n_layer          (hparams.n_layer()),
    n_layer_nextn    (hparams.n_layer_nextn),
    n_rot            (hparams.n_rot()),
    n_ctx            (cparams.n_ctx),
    n_head           (hparams.n_head()),
    n_head_kv        (hparams.n_head_kv()),
    n_embd_head_k    (hparams.n_embd_head_k()),
    n_embd_k_gqa     (hparams.n_embd_k_gqa()),
    n_embd_head_v    (hparams.n_embd_head_v()),
    n_embd_v_gqa     (hparams.n_embd_v_gqa()),
    n_expert         (hparams.n_expert),
    n_expert_used    (cparams.warmup ? hparams.n_expert : hparams.n_expert_used()),
    freq_base        (cparams.rope_freq_base),
    freq_scale       (cparams.rope_freq_scale),
    ext_factor       (cparams.yarn_ext_factor),
    attn_factor      (cparams.yarn_attn_factor),
    beta_fast        (cparams.yarn_beta_fast),
    beta_slow        (cparams.yarn_beta_slow),
    norm_eps         (hparams.f_norm_eps),
    norm_rms_eps     (hparams.f_norm_rms_eps),
    n_tokens         (ubatch.n_tokens),
    n_outputs        (params.n_outputs),
    n_ctx_orig       (cparams.n_ctx_orig_yarn),
    pooling_type     (cparams.pooling_type),
    rope_type        (hparams.rope_type),
    sched            (params.sched),
    backend_cpu      (params.backend_cpu),
    cvec             (params.cvec),
    loras            (params.loras),
    mctx             (params.mctx),
    cross            (params.cross),
    mtp_round        (params.mtp_round),
    tokens_dev       (params.tokens_dev),
    embd_dev         (params.embd_dev),
    loop             (params.loop),
    chain            (params.chain),
    draft_head       (params.draft_head),
    samplers         (params.samplers),
    cb_func          (params.cb),
    res              (params.res),
    ctx0             (res->get_ctx()),
    gf               (res->get_gf()) {
        res->set_params(params);
        if (draft_head && draft_head->whole) {
            mm_subst = draft_head->src;
            for (const auto & lora : *loras) {
                mm_subst = lora.first->get_weight(draft_head->src) ? nullptr : mm_subst;
            }
        }
    }

void llm_graph_context::cb(ggml_tensor * cur, const char * name, int il) const {
    if (cb_func) {
        cb_func(ubatch, cur, name, il);
    }
}



ggml_tensor * llm_graph_context::build_cvec(
         ggml_tensor * cur,
                 int   il) const {
    return cvec->apply_to(ctx0, cur, il);
}

ggml_tensor * llm_graph_context::build_lora_mm(
          ggml_tensor * w,
          ggml_tensor * cur,
          ggml_tensor * w_s) const {
    if (mm_subst && w == mm_subst && (!w_s || ggml_nelements(w_s) == 1)) {
        ggml_tensor * res = ggml_mul_mat(ctx0, draft_head->w, cur);
        return w_s ? ggml_mul(ctx0, res, w_s) : res;
    }

    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);

    if (w_s) {
        res = ggml_mul(ctx0, res, w_s);
    }

    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float adapter_scale = lora.second;
        const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

        ggml_tensor * ab_cur = ggml_mul_mat(
                ctx0, lw->b,
                ggml_mul_mat(ctx0, lw->a, cur)
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_lora_mm_id(
          ggml_tensor * w,   // ggml_tensor * as
          ggml_tensor * cur, // ggml_tensor * b
          ggml_tensor * ids,
          ggml_tensor * w_s) const {
    ggml_tensor * res = ggml_mul_mat_id(ctx0, w, cur, ids);

    if (w_s) {
        const int64_t n_expert = w_s->ne[0];
        const int64_t n_tokens = cur->ne[2];
        ggml_tensor * s = ggml_reshape_3d(ctx0, w_s, 1, n_expert, 1);
        s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
        s = ggml_get_rows(ctx0, s, ids);
        res = ggml_mul(ctx0, res, s);
    }
    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float alpha = lora.first->alpha;
        const float rank  = (float) lw->b->ne[0];
        const float scale = alpha ? lora.second * alpha / rank : lora.second;

        ggml_tensor * ab_cur = ggml_mul_mat_id(
                ctx0, lw->b,
                ggml_mul_mat_id(ctx0, lw->a, cur, ids),
                ids
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_norm(
         ggml_tensor * cur,
         ggml_tensor * mw,
         ggml_tensor * mb,
       llm_norm_type   type,
                 int   il) const {
    switch (type) {
        case LLM_NORM:       cur = ggml_norm    (ctx0, cur, hparams.f_norm_eps);     break;
        case LLM_NORM_RMS:   cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps); break;
        case LLM_NORM_GROUP:
            {
                cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);
                cur = ggml_group_norm(ctx0, cur, hparams.n_norm_groups, hparams.f_norm_group_eps);
                cur = ggml_reshape_2d(ctx0, cur, cur->ne[0],    cur->ne[2]);
            } break;
    }

    if (mw || mb) {
        cb(cur, "norm", il);
    }

    if (mw) {
        cur = ggml_mul(ctx0, cur, mw);
        if (mb) {
            cb(cur, "norm_w", il);
        }
    }

    if (mb) {
        cur = ggml_add(ctx0, cur, mb);
    }

    return cur;
}


llm_graph_qkv llm_graph_context::build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head,
                  int64_t   n_head,
                  int64_t   n_head_kv,
                      int   il,
                  int64_t   n_rows) const {
    return build_qkv(layer, cur,
            n_embd_head, n_head,
            n_embd_head, n_head_kv,
            n_embd_head, n_head_kv,
            il, true, n_rows);
}

llm_graph_qkv llm_graph_context::build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head_q,
                  int64_t   n_head_q,
                  int64_t   n_embd_head_k,
                  int64_t   n_head_k,
                  int64_t   n_embd_head_v,
                  int64_t   n_head_v,
                      int   il,
                     bool   reshape,
                  int64_t   n_rows) const {
    const int64_t n_embd_q = n_embd_head_q * n_head_q;
    const int64_t n_embd_k = n_embd_head_k * n_head_k;

    // a round's step covers only its own rows
    if (n_rows == 0) {
        n_rows = n_tokens;
    }

    ggml_tensor * Qcur, * Kcur, * Vcur;

    if (layer.wqkv) {
        // fused QKV path
        ggml_tensor * qkv = build_lora_mm(layer.wqkv, cur, layer.wqkv_s);
        cb(qkv, "wqkv", il);
        if (layer.wqkv_b) {
            qkv = ggml_add(ctx0, qkv, layer.wqkv_b);
            cb(qkv, "wqkv_b", il);
        } else if (layer.wq_b && layer.wk_b && layer.wv_b) {
            // Fused weights may coexist with separate Q/K/V biases in legacy or custom GGUFs.
            ggml_tensor * qkv_b = ggml_concat(ctx0, ggml_concat(ctx0, layer.wq_b, layer.wk_b, 0), layer.wv_b, 0);
            qkv = ggml_add(ctx0, qkv, qkv_b);
            cb(qkv, "wqkv_b", il);
        }
        if (reshape && hparams.f_clamp_kqv > 0.0f) {
            qkv = ggml_clamp(ctx0, qkv, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(qkv, "wqkv_clamped", il);
        }
        if (reshape) {
            Qcur = ggml_view_3d(ctx0, qkv, n_embd_head_q, n_head_q, n_rows,
                ggml_row_size(qkv->type, n_embd_head_q), qkv->nb[1], 0);
            Kcur = ggml_view_3d(ctx0, qkv, n_embd_head_k, n_head_k, n_rows,
                ggml_row_size(qkv->type, n_embd_head_k), qkv->nb[1],
                ggml_row_size(qkv->type, n_embd_q));
            Vcur = ggml_view_3d(ctx0, qkv, n_embd_head_v, n_head_v, n_rows,
                ggml_row_size(qkv->type, n_embd_head_v), qkv->nb[1],
                ggml_row_size(qkv->type, n_embd_q + n_embd_k));
        } else {
            Qcur = ggml_view_2d(ctx0, qkv, n_embd_q, n_rows, qkv->nb[1], 0);
            Kcur = ggml_view_2d(ctx0, qkv, n_embd_k, n_rows, qkv->nb[1],
                ggml_row_size(qkv->type, n_embd_q));
            Vcur = ggml_view_2d(ctx0, qkv, n_embd_head_v * n_head_v, n_rows, qkv->nb[1],
                ggml_row_size(qkv->type, n_embd_q + n_embd_k));
        }
        if (!reshape) {
            Qcur = ggml_cont(ctx0, Qcur);
            Kcur = ggml_cont(ctx0, Kcur);
            Vcur = ggml_cont(ctx0, Vcur);
        }
    } else {
        // separate Q/K/V path
        Qcur = build_lora_mm(layer.wq, cur, layer.wq_s);
        if (reshape) {
            cb(Qcur, "Qcur", il);
        }
        if (layer.wq_b) {
            Qcur = ggml_add(ctx0, Qcur, layer.wq_b);
            if (reshape) {
                cb(Qcur, "Qcur", il);
            }
        }
        if (reshape && hparams.f_clamp_kqv > 0.0f) {
            Qcur = ggml_clamp(ctx0, Qcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Qcur, "Qcur_clamped", il);
        }
        Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
        if (reshape) {
            cb(Kcur, "Kcur", il);
        }
        if (layer.wk_b) {
            Kcur = ggml_add(ctx0, Kcur, layer.wk_b);
            if (reshape) {
                cb(Kcur, "Kcur", il);
            }
        }
        if (reshape && hparams.f_clamp_kqv > 0.0f) {
            Kcur = ggml_clamp(ctx0, Kcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Kcur, "Kcur_clamped", il);
        }
        Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
        if (reshape) {
            cb(Vcur, "Vcur", il);
        }
        if (layer.wv_b) {
            Vcur = ggml_add(ctx0, Vcur, layer.wv_b);
            if (reshape) {
                cb(Vcur, "Vcur", il);
            }
        }
        if (reshape && hparams.f_clamp_kqv > 0.0f) {
            Vcur = ggml_clamp(ctx0, Vcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Vcur, "Vcur_clamped", il);
        }
        if (reshape) {
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_q, n_head_q, n_rows);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, n_head_k, n_rows);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head_v, n_head_v, n_rows);
        }
    }

    if (reshape) {
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);
    }

    return { Qcur, Kcur, Vcur };
}


ggml_tensor * llm_graph_context::build_ffn(
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
                 int   il) const {
    // NVFP4 support is currently restricted to
    // 1) LORA absence (*_s would be applied after LORA residual, which is incorrect)
    // 2) bias absense (*_s would be applied after bias addition, which is incorrect)
    // TODO: disambiguate LLM-architectural scales (which use *_s) from NVFP4 scale_2 (which also uses *_s currently)
    auto has_lora = [this](ggml_tensor * w) {
        if (!w) {
            return false;
        }
        for (const auto & lora : *loras) {
            if (lora.first->get_weight(w) != nullptr) {
                return true;
            }
        }
        return false;
    };

    GGML_ASSERT(!up_s   || !up_b   || !up   || up->type   != GGML_TYPE_NVFP4);
    GGML_ASSERT(!gate_s || !gate_b || !gate || gate->type != GGML_TYPE_NVFP4);
    GGML_ASSERT(!down_s || !down_b || !down || down->type != GGML_TYPE_NVFP4);
    GGML_ASSERT(!up_s   || !up   || up->type   != GGML_TYPE_NVFP4 || !has_lora(up));
    GGML_ASSERT(!gate_s || !gate || gate->type != GGML_TYPE_NVFP4 || !has_lora(gate));
    GGML_ASSERT(!down_s || !down || down->type != GGML_TYPE_NVFP4 || !has_lora(down));

    ggml_tensor * tmp = up ? build_lora_mm(up, cur) : cur;
    cb(tmp, "ffn_up", il);

    if (up_b) {
        tmp = ggml_add(ctx0, tmp, up_b);
        cb(tmp, "ffn_up_b", il);
    }

    if (up_s) {
        tmp = ggml_mul(ctx0, tmp, up_s);
        cb(tmp, "ffn_up_s", il);
    }

    if (gate) {
        switch (type_gate) {
            case LLM_FFN_SEQ:
                {
                    cur = build_lora_mm(gate, tmp);
                    cb(cur, "ffn_gate", il);
                } break;
            case LLM_FFN_PAR:
                {
                    cur = build_lora_mm(gate, cur);
                    cb(cur, "ffn_gate", il);
                } break;
        }

        if (gate_b) {
            cur = ggml_add(ctx0, cur, gate_b);
            cb(cur, "ffn_gate_b", il);
        }

        if (gate_s) {
            cur = ggml_mul(ctx0, cur, gate_s);
            cb(cur, "ffn_gate_s", il);
        }

    } else {
        cur = tmp;
    }

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate && type_gate == LLM_FFN_PAR) {
                if (il >= 0) {
                    const float limit = hparams.swiglu_clamp_shexp[il];
                    constexpr float eps = 1e-6f;
                    if (limit > eps) {
                        if (arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_GLM5_NEXT || (arch == LLM_ARCH_DFLASH && hparams.dsv4_hc_mult > 0)) {
                            cur = ggml_swiglu_clamp(ctx0, cur, tmp, limit);
                        } else {
                            tmp = ggml_clamp(ctx0, tmp, -limit, limit);
                            cb(tmp, "ffn_up_clamped", il);
                            ggml_tensor * gate_act = ggml_silu(ctx0, cur);
                            cb(gate_act, "ffn_silu", il);
                            gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
                            cb(gate_act, "ffn_silu_clamped", il);
                            cur = ggml_mul(ctx0, gate_act, tmp);
                        }
                        cb(cur, "ffn_swiglu_limited", il);
                        type_gate = LLM_FFN_SEQ;
                        break;
                    }
                }

                cur = ggml_swiglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_swiglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_silu", il);
            } break;
        case LLM_FFN_GELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_geglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_gelu", il);
                if (act_scales != NULL) {
                    cur = ggml_div(ctx0, cur, act_scales);
                    cb(cur, "ffn_act", il);
                }
            } break;
        case LLM_FFN_RELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_reglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_reglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);

                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_sqr(relu)", il);
            } break;
        case LLM_FFN_SWIGLU:
            {
                cur = ggml_swiglu(ctx0, cur);
                cb(cur, "ffn_swiglu", il);
            } break;
        case LLM_FFN_SWIGLU_OAI_MOE:
            if (gate && type_gate == LLM_FFN_PAR) {
                // same alpha/limit constants as gpt-oss
                const float alpha = 1.702f;
                const float limit = 7.0f;
                cur = ggml_swiglu_oai(ctx0, cur, tmp, alpha, limit);
                cb(cur, "ffn_swiglu_oai", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                GGML_ABORT("LLM_FFN_SWIGLU_OAI_MOE requires a parallel gate");
            } break;
        case LLM_FFN_GEGLU:
            {
                cur = ggml_geglu(ctx0, cur);
                cb(cur, "ffn_geglu", il);
            } break;
        case LLM_FFN_REGLU:
            {
                cur = ggml_reglu(ctx0, cur);
                cb(cur, "ffn_reglu", il);
            } break;
        case LLM_FFN_SITU:
            GGML_ABORT("not yet supported");
        default:
            GGML_ABORT("fatal error");
    }

    if (gate && type_gate == LLM_FFN_PAR) {
        cur = ggml_mul(ctx0, cur, tmp);
        cb(cur, "ffn_gate_par", il);
    }

    if (down) {
        cur = build_lora_mm(down, cur);
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_JAIS2) {
            // GLM4, GLM4_MOE, and JAIS2 seem to have numerical issues with half-precision accumulators
            ggml_prec_set_acc(cur, GGML_PREC_F32);
        }
    }

    if (down_b) {
        cb(cur, "ffn_down", il);
    }

    if (down_b) {
        cur = ggml_add(ctx0, cur, down_b);
    }

    if (down_s) {
        cur = ggml_mul(ctx0, cur, down_s);
        cb(cur, "ffn_down_s", il);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_moe_ffn(
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
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s,
         ggml_tensor * selected_experts_in) const {
    return build_moe_ffn(
        cur,
        gate_inp,  /* gate_inp_b  */ nullptr,
        up_exps,   /* up_exps_b   */ nullptr,
        gate_exps, /* gate_exps_b */ nullptr,
        down_exps, /* down_exps_b */ nullptr,
        exp_probs_b,
        n_expert,
        n_expert_used,
        type_op,
        norm_w,
        w_scale,
        gating_op,
        il,
        probs_in,
        gate_up_exps,
        /* gate_up_exps_b */ nullptr,
        up_exps_s,
        gate_exps_s,
        down_exps_s,
        selected_experts_in
    );
}

ggml_tensor * llm_graph_context::build_moe_ffn(
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
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * gate_up_exps_b,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s,
         ggml_tensor * selected_experts_in) const {
    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const bool weight_before_ffn = arch == LLM_ARCH_LLAMA4; // for llama4, we apply the sigmoid-ed weights before the FFN

    ggml_tensor * logits = nullptr;

    if (probs_in == nullptr) {
        logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]
        if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
            ggml_prec_set_acc(logits, GGML_PREC_F32);
        }
        cb(logits, "ffn_moe_logits", il);
    } else {
        logits = probs_in;
    }

    if (gate_inp_b) {
        logits = ggml_add(ctx0, logits, gate_inp_b);
        cb(logits, "ffn_moe_logits_biased", il);
    }

    ggml_tensor * probs = nullptr;
    switch (gating_op) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
            {
                probs = ggml_soft_max(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
            {
                probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT:
            {
                probs = logits; // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS:
            {
                probs = ggml_sqrt(ctx0, ggml_softplus(ctx0, logits)); // [n_expert, n_tokens]
            } break;
        default:
            GGML_ABORT("fatal error");
    }
    cb(probs, "ffn_moe_probs", il);

    // add experts selection bias - introduced in DeepSeek V3
    // leave probs unbiased as it's later used to get expert weights
    ggml_tensor * selection_probs = probs;
    if (exp_probs_b != nullptr) {
        selection_probs = ggml_add(ctx0, probs, exp_probs_b);
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // llama4 doesn't have exp_probs_b, and sigmoid is only used after top_k
    // see: https://github.com/meta-llama/llama-models/blob/699a02993512fb36936b1b0741e13c06790bcf98/models/llama4/moe.py#L183-L198
    if (arch == LLM_ARCH_LLAMA4) {
        selection_probs = logits;
    }

    if (arch == LLM_ARCH_GROVEMOE) {
        selection_probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // select top n_group_used expert groups
    // https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/modeling_deepseek.py#L440-L457
    if (hparams.n_expert_groups > 1 && n_tokens > 0) {
        const int64_t n_exp_per_group = n_expert / hparams.n_expert_groups;

        // organize experts into n_expert_groups
        ggml_tensor * selection_groups = ggml_reshape_3d(ctx0, selection_probs, n_exp_per_group, hparams.n_expert_groups, n_tokens); // [n_exp_per_group, n_expert_groups, n_tokens]

        ggml_tensor * group_scores = ggml_argsort_top_k(ctx0, selection_groups, 2); // [2, n_expert_groups, n_tokens]
        group_scores = ggml_get_rows(ctx0, ggml_reshape_4d(ctx0, selection_groups, 1, selection_groups->ne[0], selection_groups->ne[1], selection_groups->ne[2]), group_scores); // [1, 2, n_expert_groups, n_tokens]

        // get top n_group_used expert groups
        group_scores = ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2], group_scores->ne[3])); // [1, n_expert_groups, n_tokens]
        group_scores = ggml_reshape_2d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2]); // [n_expert_groups, n_tokens]

        ggml_tensor * expert_groups = ggml_argsort_top_k(ctx0, group_scores, hparams.n_group_used); // [n_group_used, n_tokens]
        cb(expert_groups, "ffn_moe_group_topk", il);

        // mask out the other groups
        selection_probs = ggml_get_rows(ctx0, selection_groups, expert_groups); // [n_exp_per_group, n_group_used, n_tokens]
        selection_probs = ggml_set_rows(ctx0, ggml_fill(ctx0, selection_groups, -INFINITY), selection_probs, expert_groups); // [n_exp_per_group, n_expert_groups, n_tokens]
        selection_probs = ggml_reshape_2d(ctx0, selection_probs, n_expert, n_tokens); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_masked", il);
    }

    // select experts
    ggml_tensor * selected_experts = selected_experts_in;
    if (selected_experts == nullptr) {
        selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
        cb(selected_experts->src[0], "ffn_moe_argsort", il);
    }
    cb(selected_experts, "ffn_moe_topk", il);

    if (arch == LLM_ARCH_GROVEMOE && n_expert != hparams.n_expert) {
        // TODO: Use scalar div instead when/if implemented
        ggml_tensor * f_sel = ggml_cast(ctx0, selected_experts, GGML_TYPE_F32);
        selected_experts = ggml_cast(ctx0, ggml_scale(ctx0, f_sel, 1.0f / float(hparams.n_group_experts)), GGML_TYPE_I32);
        probs = ggml_reshape_3d(ctx0, probs, 1, hparams.n_expert, n_tokens);
    } else {
        probs = ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens);
    }

    ggml_tensor * weights = ggml_get_rows(ctx0, probs, selected_experts); // [1, n_expert_used, n_tokens]
    cb(weights, "ffn_moe_weights", il);


    if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
        weights = ggml_soft_max(ctx0, weights); // [n_expert_used, n_tokens]
        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
        cb(weights, "ffn_moe_weights_softmax", il);
    }

    if (norm_w) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);

        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
        cb(weights_sum, "ffn_moe_weights_sum", il);

        // Avoid division by zero, clamp to smallest number representable by F16
        weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
        cb(weights_sum, "ffn_moe_weights_sum_clamped", il);

        weights = ggml_div(ctx0, weights, weights_sum); // [n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights_norm", il);

        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
    }
    if (w_scale != 0.0f && w_scale != 1.0f) {
        weights = ggml_scale(ctx0, weights, w_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }

    //call early so that topk-moe can be used
    ggml_build_forward_expand(gf, weights);

    cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);

    if (weight_before_ffn) {
        // repeat cur to [n_embd, n_expert_used, n_tokens]
        ggml_tensor * repeated = ggml_repeat_4d(ctx0, cur, n_embd, n_expert_used, n_tokens, 1);
        cur = ggml_mul(ctx0, repeated, weights);
        cb(cur, "ffn_moe_weighted", il);
    }

    ggml_tensor * up = nullptr;
    ggml_tensor * experts = nullptr;

    if (gate_up_exps) {
        // merged gate_up path: one mul_mat_id, then split into gate and up views
        ggml_tensor * gate_up = build_lora_mm_id(gate_up_exps, cur, selected_experts, up_exps_s); // [n_ff*2, n_expert_used, n_tokens]
        cb(gate_up, "ffn_moe_gate_up", il);

        if (up_exps_s) {
            cb(gate_up, "ffn_moe_gate_up_scaled", il);
        }

        if (gate_up_exps_b) {
            gate_up = ggml_add_id(ctx0, gate_up, gate_up_exps_b, selected_experts);
            cb(gate_up, "ffn_moe_gate_up_biased", il);
        }

        const int64_t n_ff = gate_up->ne[0] / 2;
        cur = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
        cb(cur, "ffn_moe_gate", il);
        up  = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], n_ff * gate_up->nb[0]);
        cb(up, "ffn_moe_up", il);
    } else {
        // separate gate and up path
        up = build_lora_mm_id(up_exps, cur, selected_experts, up_exps_s); // [n_ff, n_expert_used, n_tokens]
        cb(up, "ffn_moe_up", il);

        if (up_exps_s) {
            cb(up, "ffn_moe_up_scaled", il);
        }

        if (up_exps_b) {
            up = ggml_add_id(ctx0, up, up_exps_b, selected_experts);
            cb(up, "ffn_moe_up_biased", il);
        }

        if (gate_exps) {
            cur = build_lora_mm_id(gate_exps, cur, selected_experts, gate_exps_s); // [n_ff, n_expert_used, n_tokens]
            cb(cur, "ffn_moe_gate", il);
        } else {
            cur = up;
        }

        if (gate_exps_s) {
            cb(cur, "ffn_moe_gate_scaled", il);
        }

        if (gate_exps_b) {
            cur = ggml_add_id(ctx0, cur, gate_exps_b, selected_experts);
            cb(cur, "ffn_moe_gate_biased", il);
        }
    }

    const bool has_gate = gate_exps || gate_up_exps;

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate_exps) {
                if (il >= 0) {
                    const float limit = hparams.swiglu_clamp_exp[il];
                    constexpr float eps = 1e-6f;
                    if (limit > eps) {
                        if (arch == LLM_ARCH_MAPLE || arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_GLM5_NEXT || (arch == LLM_ARCH_DFLASH && hparams.dsv4_hc_mult > 0) || arch == LLM_ARCH_HY_V4) {
                            cur = ggml_swiglu_clamp(ctx0, cur, up, limit);
                        } else {
                            up = ggml_clamp(ctx0, up, -limit, limit);
                            cb(up, "ffn_moe_up_clamped", il);
                            ggml_tensor * gate_act = ggml_silu(ctx0, cur);
                            cb(gate_act, "ffn_moe_silu", il);
                            gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
                            cb(gate_act, "ffn_moe_silu_clamped", il);
                            cur = ggml_mul(ctx0, gate_act, up);
                        }
                        cb(cur, "ffn_moe_swiglu_limited", il);
                        break;
                    }
                }
            }

            if (has_gate) {
                cur = ggml_swiglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_swiglu", il);
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_moe_silu", il);
            } break;
        case LLM_FFN_SITU:
            {
                // situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb)
                GGML_ASSERT(has_gate);
                const float beta = hparams.situ_beta;
                const float lb   = hparams.situ_linear_beta;

                ggml_tensor * act = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/beta)), beta);
                act = ggml_mul(ctx0, act, ggml_sigmoid(ctx0, cur));
                if (lb > 0.0f) {
                    up = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, up, 1.0f/lb)), lb);
                }
                cur = ggml_mul(ctx0, act, up);
                cb(cur, "ffn_moe_situ", il);
            } break;
        case LLM_FFN_GELU:
            if (has_gate) {
                cur = ggml_geglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_geglu", il);
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_moe_gelu", il);
            } break;
        case LLM_FFN_SWIGLU_OAI_MOE:
            {
                // TODO: move to hparams?
                constexpr float alpha = 1.702f;
                constexpr float limit = 7.0f;
                cur = ggml_swiglu_oai(ctx0, cur, up, alpha, limit);
                cb(cur, "ffn_moe_swiglu_oai", il);
            } break;
        case LLM_FFN_RELU:
            if (has_gate) {
                cur = ggml_reglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_reglu", il);
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_moe_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            if (has_gate) {
                // TODO: add support for gated squared relu
                GGML_ABORT("fatal error: gated squared relu not implemented");
            } else {
                cur = ggml_relu(ctx0, cur);
                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_moe_relu_sqr", il);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    experts = build_lora_mm_id(down_exps, cur, selected_experts, down_exps_s); // [n_embd, n_expert_used, n_tokens]
    cb(experts, "ffn_moe_down", il);

    if (down_exps_s) {
        cb(experts, "ffn_moe_down_scaled", il);
    }

    if (down_exps_b) {
        experts = ggml_add_id(ctx0, experts, down_exps_b, selected_experts);
        cb(experts, "ffn_moe_down_biased", il);
    }

    if (!weight_before_ffn) {
        experts = ggml_mul(ctx0, experts, weights);
        cb(experts, "ffn_moe_weighted", il);
    }

    ggml_build_forward_expand(gf, experts);

    ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };

    assert(n_expert_used > 0);

    // order the views before the adds
    // Use per-layer n_expert_used to bound the graph even during warmup (avoids
    // the large-add-nodes issue for uniform arches; for Puzzle the per-layer
    // value is correct). ref: https://github.com/ggml-org/llama.cpp/pull/14753
    const uint32_t n_expert_used_il = hparams.n_expert_used(il);
    for (uint32_t i = 0; i < n_expert_used_il; ++i) {
        cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);

        ggml_build_forward_expand(gf, cur_experts[i]);
    }

    // aggregate experts
    ggml_tensor * moe_out = cur_experts[0];

    for (uint32_t i = 1; i < n_expert_used_il; ++i) {
        moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);

        ggml_build_forward_expand(gf, moe_out);
    }

    if (n_expert_used_il == 1) {
        // avoid returning a non-contiguous tensor
        moe_out = ggml_cont(ctx0, moe_out);
    }

    cb(moe_out, "ffn_moe_out", il);

    return moe_out;
}

// input embeddings with optional lora
ggml_tensor * llm_graph_context::build_inp_embd(ggml_tensor * tok_embd) const {
    const int64_t n_embd_inp = hparams.n_embd_inp();
    const int64_t n_embd     = hparams.n_embd;

    assert(n_embd_inp >= n_embd);

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

    if (loop) {
        // the loop body reads the ids its previous step left
        GGML_ASSERT(ubatch.token && ubatch.n_tokens <= loop->n_tok && "loop body: more tokens than the loop holds");
        inp->device = true;
        inp->tokens = ggml_view_1d(ctx0, loop->tok, ubatch.n_tokens, 0);
    } else if (tokens_dev && tokens_dev->use_dev(ubatch)) {
        if (!tokens_dev->tok_dev) {
            ggml_init_params ip = { ggml_tensor_overhead(), nullptr, true };
            tokens_dev->ctx_dev = ggml_init(ip);
            GGML_ASSERT(tokens_dev->ctx_dev);
            tokens_dev->tok_dev = ggml_new_tensor_1d(tokens_dev->ctx_dev, GGML_TYPE_I32, tokens_dev->n_max);
            ggml_set_name(tokens_dev->tok_dev, "inp_tokens_dev");
            // the rows live in the memory of the device that gathers from the table: a table in mapped host memory
            // still has its ids written and read on the device, never through the host
            ggml_backend_buffer_type_t rows_buft = ggml_backend_buffer_get_type(tok_embd->buffer);
            if (ggml_backend_dev_t rows_dev = ggml_backend_buft_get_device(rows_buft)) {
                rows_buft = ggml_backend_dev_buffer_type(rows_dev);
            }
            tokens_dev->buf_dev = ggml_backend_alloc_ctx_tensors_from_buft(tokens_dev->ctx_dev, rows_buft);
            GGML_ASSERT(tokens_dev->buf_dev && "failed to allocate the device token rows");
        }
        inp->tokens     = ggml_view_1d(ctx0, tokens_dev->tok_dev, ubatch.n_tokens, 0);
        inp->tokens_dev = tokens_dev;
        inp->dev_tokens = true;
        inp->sched      = sched;
    } else if (is_chain_form()) {
        const int64_t n  = ubatch.n_tokens;
        ggml_tensor * io = chain->tok_io;

        inp->chain  = chain;
        inp->tok_in = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 2*n);
        cb(inp->tok_in, "inp_tok_in", -1);
        ggml_set_input(inp->tok_in);

        // the host ids land in tok_io first, so one gather serves host and device rows alike
        ggml_tensor * table = ggml_set_1d_inplace(ctx0, io, ggml_view_1d(ctx0, inp->tok_in, n, 0),
                chain->n_cm*ggml_element_size(io));

        ggml_tensor * ids = ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, table, 1, 2*chain->n_cm),
                ggml_view_1d(ctx0, inp->tok_in, n, n*ggml_element_size(inp->tok_in)));
        inp->tokens = ggml_reshape_1d(ctx0, ids, n);
        cb(inp->tokens, "inp_tokens", -1);

        res->chain_ok = true;
    } else {
        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        cb(inp->tokens, "inp_tokens", -1);
        ggml_set_input(inp->tokens);
        // an armed decode split into several ubatches uploads the drafted ids over the batch's placeholders
        inp->tokens_dev = tokens_dev;
    }
    res->t_inp_tokens = inp->tokens;

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, ubatch.n_tokens);
    cb(inp->embd, "inp_embd", -1);
    ggml_set_input(inp->embd);

    // select one of the 2 inputs, based on the batch contents
    // ref: https://github.com/ggml-org/llama.cpp/pull/18550
    std::array<ggml_tensor *, 2> inps;

    // token embeddings path (ubatch.token != nullptr)
    {
        auto & cur = inps[0];

        cur = ggml_get_rows(ctx0, tok_embd, inp->tokens);

        // apply lora for embedding tokens if needed
        for (const auto & lora : *loras) {
            llama_adapter_lora_weight * lw = lora.first->get_weight(tok_embd);
            if (lw == nullptr) {
                continue;
            }

            const float adapter_scale = lora.second;
            const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

            ggml_tensor * inpL_delta = ggml_scale(ctx0, ggml_mul_mat(
                        ctx0, lw->b, // non-transposed lora_b
                        ggml_get_rows(ctx0, lw->a, inp->tokens)
                        ), scale);

            cur = ggml_add(ctx0, cur, inpL_delta);
        }

        if (n_embd_inp != n_embd) {
            cur = ggml_pad(ctx0, cur, hparams.n_embd_inp() - n_embd, 0, 0, 0);
        }
    }

    // vector embeddings path (ubatch.embd != nullptr)
    {
        auto & cur = inps[1];

        cur = inp->embd;
    }

    assert(ggml_are_same_shape (inps[0], inps[1]));
    assert(ggml_are_same_stride(inps[0], inps[1]));

    ggml_tensor * cur = ggml_build_forward_select(gf, inps.data(), inps.size(), ubatch.token ? 0 : 1);

    if (n_embd_inp != n_embd) {
        cur = ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 0);
    }

    res->t_inp_embd = cur;

    // For Granite architecture
    // NOTE: For deepstack models, only apply scale to token inputs (ie text-only input).
    //  Raw embeddings are assumed to be multimodal inputs that should not be scaled.
    if (hparams.f_embedding_scale != 0.0f && (ubatch.token || hparams.n_deepstack_layers == 0)) {
        if (!ggml_is_contiguous(cur)) {
            cur = ggml_cont(ctx0, cur);
        }
        cur = ggml_scale(ctx0, cur, hparams.f_embedding_scale);
    }

    cb(cur, "embd", -1);

    res->add_input(std::move(inp));

    // make sure the produced embeddings are immediately materialized in the ggml graph
    // ref: https://github.com/ggml-org/llama.cpp/pull/18599
    ggml_build_forward_expand(gf, cur);

    return cur;
}

bool llm_graph_context::is_chain_form() const {
    return chain && chain->n_cm > 0 && !loop && ubatch.token && ubatch.n_tokens <= (uint32_t) chain->n_cm &&
           !(tokens_dev && tokens_dev->use_dev(ubatch));
}

llm_graph_mtp_attn llm_graph_mtp_attn_of(llm_graph_input_attn_kv * inp) {
    return { inp->mctx, inp->get_k_idxs(), inp->get_v_idxs(), inp->get_kq_mask() };
}

llm_graph_mtp_attn llm_graph_mtp_attn_of(llm_graph_input_attn_k * inp) {
    return { inp->mctx, inp->get_k_idxs(), nullptr, inp->get_kq_mask() };
}

llm_graph_mtp_attn llm_graph_mtp_attn_of(llm_graph_input_attn_kv_iswa * inp, bool swa) {
    const auto * mctx = swa ? inp->mctx->get_swa() : inp->mctx->get_base();
    llm_graph_mtp_attn res = swa
        ? llm_graph_mtp_attn{ mctx, inp->get_k_idxs_swa(), inp->get_v_idxs_swa(), inp->get_kq_mask_swa() }
        : llm_graph_mtp_attn{ mctx, inp->get_k_idxs(),     inp->get_v_idxs(),     inp->get_kq_mask()     };
    res.kv_strm = inp->kv_strm;
    return res;
}

llm_graph_mtp_attn llm_graph_mtp_attn_of(llm_graph_input_attn_k_iswa * inp, bool swa) {
    const auto * mctx = swa ? inp->mctx->get_swa() : inp->mctx->get_base();
    return swa
        ? llm_graph_mtp_attn{ mctx, inp->get_k_idxs_swa(), nullptr, inp->get_kq_mask_swa(), inp->self_k_rot_swa }
        : llm_graph_mtp_attn{ mctx, inp->get_k_idxs(),     nullptr, inp->get_kq_mask(),     inp->self_k_rot     };
}

ggml_tensor * llm_graph_context::build_attn_mtp(
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
                     ggml_tensor * mask_in) const {
    const auto * mctx_cur = attn.mctx;
    const int64_t n_rows  = q->ne[2];

    if (attn.k_rot) {
        q = llama_mul_mat_hadamard(ctx0, q, attn.k_rot);
        if (k) {
            k = llama_mul_mat_hadamard(ctx0, k, attn.k_rot);
        }
    }

    // a drafter attending over another context's cache writes nothing
    if (k) {
        ggml_tensor * k_idx = ggml_view_1d(ctx0, attn.k_idxs, n_rows, row*ggml_element_size(attn.k_idxs));
        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k, k_idx, il));
    }

    if (attn.v_idxs && v) {
        ggml_tensor * v_idx = ggml_view_1d(ctx0, attn.v_idxs, n_rows, row*ggml_element_size(attn.v_idxs));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v, v_idx, il));
    }

    if (kv_only) {
        // a replay only refreshes the cache; nothing reads its output
        return nullptr;
    }

    ggml_tensor * mask = mask_in ? mask_in
        : ggml_view_2d(ctx0, attn.kq_mask, attn.kq_mask->ne[0], n_rows, attn.kq_mask->nb[1], row*attn.kq_mask->nb[1]);

    ggml_tensor * k_all = attn.kv_strm ? mctx_cur->get_k_stream0(ctx0, il) : mctx_cur->get_k(ctx0, il);

    // a cache that holds no separate V reads it out of the K rows, as build_attn does
    ggml_tensor * v_all = k_all;
    if (attn.v_idxs) {
        v_all = attn.kv_strm ? mctx_cur->get_v_stream0(ctx0, il) : mctx_cur->get_v(ctx0, il);
    } else if (v) {
        v_all = ggml_view_4d(ctx0, k_all, v->ne[0], k_all->ne[1], k_all->ne[2], k_all->ne[3], k_all->nb[1], k_all->nb[2], k_all->nb[3], 0);
    }

    fa_kv_strm = attn.kv_strm;
    ggml_tensor * cur = build_attn_mha(q, k_all, v_all,
            nullptr, mask, sinks, v_mla, 0, kq_scale, il);
    fa_kv_strm = nullptr;
    cb(cur, "kqv_out", il);

    if (attn.k_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, attn.k_rot);
    }

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::loop_rows() const {
    GGML_ASSERT(loop);
    ggml_tensor * r = ggml_add(ctx0, ggml_arange(ctx0, 0.0f, (float) n_tokens, 1.0f), loop->counter);
    return ggml_cast(ctx0, r, GGML_TYPE_I32);
}

void llm_graph_context::build_loop_tail() const {
    GGML_ASSERT(loop && "loop tail: no loop");
    GGML_ASSERT(res->t_sampled.size() == 1 && res->t_sampled[0] && "loop body: the row must be sampled on the device");

    ggml_tensor * sampled = res->t_sampled[0]; // I32 [1]
    ggml_tensor * step    = loop_rows();

    // the id joins the ring (kept as floats: set_rows writes floats)
    ggml_tensor * ring = ggml_set_rows(ctx0,
            ggml_reshape_2d(ctx0, loop->ring, 1, loop->n_max),
            ggml_reshape_2d(ctx0, ggml_cast(ctx0, sampled, GGML_TYPE_F32), 1, 1),
            step);
    ggml_build_forward_expand(gf, ring);

    // it is the next step's token, and the step advances
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, sampled, loop->tok));
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_scale_bias(ctx0, loop->counter, 1.0f, 1.0f), loop->counter));
}

// A whole speculative round in one graph. Rows 0 .. n_chain replay the verify
// batch the target just ran: the batch's tokens paired with the target's
// hidden rows, which come straight from the target's graph. How many of them
// the target accepted is decided here, on the device, by comparing its samples
// with the batch's tokens. The rows after them are the chain: the target's
// sample after the last accepted row, then one row per draft, each taking the
// argmax of the previous logits and the previous hidden state. The chain's
// positions follow the accepted rows; the replay rows past them are hidden
// from it by an additive mask and leave the cache before the next round.
void llm_graph_context::build_mtp_round(
         llm_graph_mtp_i & mtp,
const llm_graph_mtp_attn & attn,
             ggml_tensor * tok_embd,
             ggml_tensor * inp_tokens,
             ggml_tensor * inp_pos,
                     int   n_chain,
                 int64_t   n_embd_h) {
    const int K = n_chain;
    if (n_embd_h == 0) {
        n_embd_h = n_embd;
    }
    const int V = n_chain + 1;
    const int64_t n_seq = n_tokens / (V + K);
    GGML_ASSERT(n_seq >= 1 && n_tokens == n_seq*(V + K) && "MTP round: n_chain+1 replay rows and n_chain chain rows per sequence");
    GGML_ASSERT(inp_tokens && "MTP round: needs the verify tokens");
    GGML_ASSERT(cparams.flash_attn && "MTP round: needs flash attention");
    GGML_ASSERT(mtp_round && "MTP round: needs the round state");

    // the round's inputs live in the memory of the device the embedding weights belong to (a table in mapped host
    // memory still belongs to the device that reads it)
    ggml_backend_buffer_type_t round_buft = ggml_backend_buffer_get_type(mtp.mtp_embd_weights()->buffer);
    if (ggml_backend_dev_t round_dev = ggml_backend_buft_get_device(round_buft)) {
        round_buft = ggml_backend_dev_buffer_type(round_dev);
    }
    const auto & sg = mtp_round_ensure_dev(mtp_round, round_buft, n_embd_h, V, n_seq);

    auto rinp = std::make_unique<llm_graph_input_mtp_round>(mtp_round, sched, n_embd_h, V, n_seq);
    rinp->c_ids  = sg.c_ids_dev;
    rinp->h_tgt  = sg.h_tgt_dev;
    rinp->h_pend = sg.h_pend_dev;
    rinp->off    = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_seq);
    ggml_set_input(rinp->off);    ggml_set_name(rinp->off,    "mtp_round_off");
    ggml_tensor * c_ids  = rinp->c_ids;
    ggml_tensor * h_tgt  = rinp->h_tgt;
    ggml_tensor * h_pend = rinp->h_pend;
    ggml_tensor * off    = rinp->off;
    res->add_input(std::move(rinp));

    if (n_seq > 1) {
        build_mtp_round_seqs(mtp, attn, tok_embd, inp_tokens, inp_pos, n_chain, n_seq, c_ids, h_tgt, h_pend, off);
        return;
    }

    // M-RoPE positions are section-major, [n_pos_per_embd][n]: the entries
    // for one row sit n apart
    const int64_t n_pos = hparams.n_pos_per_embd();
    const size_t  ps    = inp_pos ? ggml_element_size(inp_pos) : 0;
    auto pos_rows = [&](int first, int n) -> ggml_tensor * {
        if (!inp_pos) {
            return nullptr;
        }
        ggml_tensor * p = ggml_view_2d(ctx0, inp_pos, n, n_pos, n_tokens*ps, first*ps);
        return ggml_reshape_1d(ctx0, ggml_cont(ctx0, p), n*n_pos);
    };
    auto rows = [&](ggml_tensor * t, int first, int n) {
        return ggml_view_2d(ctx0, t, t->ne[0], n, t->nb[1], first*t->nb[1]);
    };

    // the replay: verify token k with hidden row k-1; row 0 pairs with the
    // hidden row the previous round left
    ggml_tensor * h_replay = ggml_concat(ctx0, h_pend, rows(h_tgt, 0, K), 1);
    cb(h_replay, "mtp_replay_h", -1);
    mtp.mtp_step(rows(tok_embd, 0, V), h_replay, pos_rows(0, V), 0, true, nullptr);

    // accepted rows: the leading verify rows whose target sample equals the
    // next verify token
    const size_t is = ggml_element_size(inp_tokens);
    ggml_tensor * c_f = ggml_cast(ctx0, ggml_view_1d(ctx0, c_ids, K, 0), GGML_TYPE_F32);
    ggml_tensor * x_f = ggml_cast(ctx0, ggml_view_1d(ctx0, inp_tokens, K, is), GGML_TYPE_F32);
    ggml_tensor * eq  = ggml_scale_bias(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, ggml_sub(ctx0, c_f, x_f)), 0.0f, 1.0f), -1.0f, 1.0f);
    cb(eq, "mtp_round_eq", -1);
    ggml_tensor * n_acc = nullptr;
    ggml_tensor * run   = nullptr;
    for (int i = 0; i < K; ++i) {
        ggml_tensor * e = ggml_view_1d(ctx0, eq, 1, i*ggml_element_size(eq));
        run   = run   ? ggml_mul(ctx0, run, e)     : e;
        n_acc = n_acc ? ggml_add(ctx0, n_acc, run) : run;
    }
    cb(n_acc, "mtp_round_nacc", -1);
    ggml_tensor * n_acc_i = ggml_cast(ctx0, n_acc, GGML_TYPE_I32);
    ggml_set_output(n_acc_i);

    // the anchor: the target's sample after the last accepted row, with that
    // row's hidden state, which the next round starts from too
    ggml_tensor * anchor_id = ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, c_ids, 1, V), n_acc_i);
    ggml_tensor * anchor_h  = ggml_get_rows(ctx0, h_tgt, n_acc_i);
    cb(anchor_h, "mtp_round_anchor_h", -1);
    ggml_set_output(anchor_h);
    res->t_mtp_hpend = anchor_h;

    ggml_tensor * tok = ggml_get_rows(ctx0, mtp.mtp_embd_weights(), ggml_reshape_1d(ctx0, anchor_id, 1));
    ggml_tensor * h   = anchor_h;

    // the chain's positions: the replay's first position + n_acc + off, one
    // further per draft
    ggml_tensor * pos_i = nullptr;
    if (inp_pos) {
        ggml_tensor * base_f  = ggml_cast(ctx0, ggml_view_1d(ctx0, inp_pos, 1, 0), GGML_TYPE_F32);
        ggml_tensor * start   = ggml_add(ctx0, ggml_add(ctx0, base_f, n_acc), off);
        ggml_tensor * pos_f   = ggml_add(ctx0, ggml_arange(ctx0, 0.0f, (float) K, 1.0f), start);
        ggml_tensor * pos_all = pos_f;
        for (int s = 1; s < n_pos; ++s) {
            // text rows: every section but the last takes the position, the last is 0
            pos_all = ggml_concat(ctx0, pos_all, s + 1 == n_pos ? ggml_scale(ctx0, pos_f, 0.0f) : pos_f, 0);
        }
        pos_i = ggml_cast(ctx0, pos_all, GGML_TYPE_I32);
    }
    auto pos_chain = [&](int j) -> ggml_tensor * {
        if (!pos_i) {
            return nullptr;
        }
        const size_t pis = ggml_element_size(pos_i);
        ggml_tensor * p = ggml_view_2d(ctx0, pos_i, 1, n_pos, K*pis, j*pis);
        return ggml_reshape_1d(ctx0, ggml_cont(ctx0, p), n_pos);
    };

    // the replay rows past the accepted ones (all of them when the round comes
    // from the host) are hidden from the chain: an additive mask over their
    // cells
    ggml_tensor * kq_mask = attn.kq_mask;
    ggml_tensor * k_idxs  = attn.k_idxs;
    const int64_t n_kv    = kq_mask->ne[0];
    ggml_tensor * thr   = ggml_scale_bias(ctx0, ggml_add(ctx0, n_acc, off), 1.0f, -0.5f);
    ggml_tensor * hide  = ggml_scale(ctx0, ggml_step(ctx0, ggml_sub(ctx0, ggml_arange(ctx0, 0.0f, (float) V, 1.0f), thr)), -30000.0f);
    ggml_tensor * extra = ggml_scale(ctx0, ggml_arange(ctx0, 0.0f, (float) n_kv, 1.0f), 0.0f);
    extra = ggml_set_rows(ctx0, ggml_reshape_2d(ctx0, extra, 1, n_kv), ggml_reshape_2d(ctx0, hide, 1, V), ggml_view_1d(ctx0, k_idxs, V, 0));
    extra = ggml_cast(ctx0, ggml_reshape_2d(ctx0, extra, n_kv, 1), GGML_TYPE_F16);
    cb(extra, "mtp_round_mask", -1);
    auto mask_chain = [&](int j) {
        ggml_tensor * m = ggml_view_2d(ctx0, kq_mask, n_kv, 1, kq_mask->nb[1], (size_t) (V + j)*kq_mask->nb[1]);
        return ggml_add(ctx0, m, extra);
    };

    // the chain drafts through the reduced head where the MTP head multiplies by the LM head it was taken from;
    // an adapter on that head keeps the full one
    bool reduced = draft_head != nullptr;
    for (const auto & lora : *loras) {
        reduced = reduced && lora.first->get_weight(draft_head->src) == nullptr;
    }
    const int64_t n_draft = reduced ? draft_head->w->ne[1] : 0;

    for (int j = 0; j < K; ++j) {
        ggml_tensor * h_next = mtp.mtp_step(tok, h, pos_chain(j), V + j, false, mask_chain(j));
        cb(h_next, "mtp_chain_h", j);

        mm_subst = reduced ? draft_head->src : nullptr;
        ggml_tensor * logits_j = mtp.mtp_head(h_next);
        mm_subst = nullptr;
        cb(logits_j, "mtp_chain_logits", j);

        // the drafted token leaves the graph as an id; nothing else is read back
        ggml_tensor * next = ggml_argmax(ctx0, logits_j);
        if (reduced && logits_j->ne[0] == n_draft) {
            next = ggml_reshape_1d(ctx0, ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, draft_head->ids, 1, n_draft), next), 1);
        } else if (reduced) {
            res->draft_head_unused = true;
        }
        cb(next, "mtp_chain_next", j);
        ggml_set_output(next);
        ggml_build_forward_expand(gf, next);
        res->t_mtp_ids.push_back(next);

        if (j + 1 < K) {
            tok = ggml_get_rows(ctx0, mtp.mtp_embd_weights(), next);
            h   = h_next;
        }
    }

    // the accepted count leaves with the ids
    ggml_build_forward_expand(gf, n_acc_i);
    ggml_build_forward_expand(gf, anchor_h);
    res->t_mtp_ids.push_back(n_acc_i);
}

// the round over n_seq sequences: each one's replay rows in turn, then row n_seq*V + j*n_seq + s is chain step j
// of sequence s; what a single round keeps per sequence becomes a column
void llm_graph_context::build_mtp_round_seqs(
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
             ggml_tensor * off) {
    const int     K = n_chain;
    const int     V = n_chain + 1;
    const int64_t S = n_seq;
    const int64_t n_embd_h = h_tgt->ne[0];

    const int64_t n_pos = hparams.n_pos_per_embd();
    const size_t  ps    = inp_pos ? ggml_element_size(inp_pos) : 0;
    auto pos_rows = [&](int64_t first, int64_t n) -> ggml_tensor * {
        if (!inp_pos) {
            return nullptr;
        }
        ggml_tensor * p = ggml_view_2d(ctx0, inp_pos, n, n_pos, n_tokens*ps, first*ps);
        return ggml_reshape_1d(ctx0, ggml_cont(ctx0, p), n*n_pos);
    };

    ggml_tensor * hp3 = ggml_reshape_3d(ctx0, h_pend, n_embd_h, 1, S);
    ggml_tensor * ht3 = ggml_view_3d(ctx0, h_tgt, n_embd_h, K, S, h_tgt->nb[1], V*h_tgt->nb[1], 0);
    ggml_tensor * h_replay = ggml_reshape_2d(ctx0, ggml_concat(ctx0, hp3, ht3, 1), n_embd_h, V*S);
    cb(h_replay, "mtp_replay_h", -1);
    mtp.mtp_step(ggml_view_2d(ctx0, tok_embd, tok_embd->ne[0], V*S, tok_embd->nb[1], 0), h_replay, pos_rows(0, V*S), 0, true, nullptr);

    const size_t is = ggml_element_size(inp_tokens);
    ggml_tensor * c_f = ggml_cast(ctx0, ggml_view_2d(ctx0, c_ids,      K, S, V*ggml_element_size(c_ids), 0),  GGML_TYPE_F32);
    ggml_tensor * x_f = ggml_cast(ctx0, ggml_view_2d(ctx0, inp_tokens, K, S, V*is,                       is), GGML_TYPE_F32);
    ggml_tensor * eq  = ggml_scale_bias(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, ggml_sub(ctx0, c_f, x_f)), 0.0f, 1.0f), -1.0f, 1.0f);
    eq = ggml_cont(ctx0, ggml_transpose(ctx0, eq)); // [S, K]
    cb(eq, "mtp_round_eq", -1);
    ggml_tensor * n_acc = nullptr;
    ggml_tensor * run   = nullptr;
    for (int i = 0; i < K; ++i) {
        ggml_tensor * e = ggml_view_1d(ctx0, eq, S, i*S*ggml_element_size(eq));
        run   = run   ? ggml_mul(ctx0, run, e)     : e;
        n_acc = n_acc ? ggml_add(ctx0, n_acc, run) : run;
    }
    cb(n_acc, "mtp_round_nacc", -1);
    ggml_tensor * n_acc_i = ggml_cast(ctx0, n_acc, GGML_TYPE_I32);
    ggml_set_output(n_acc_i);

    // sequence s's anchor is row s*V + n_acc[s] of the round's inputs
    ggml_tensor * idx = ggml_cast(ctx0, ggml_add(ctx0, n_acc, ggml_arange(ctx0, 0.0f, (float) (V*S), (float) V)), GGML_TYPE_I32);
    ggml_tensor * anchor_id = ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, c_ids, 1, V*S), idx);
    ggml_tensor * anchor_h  = ggml_get_rows(ctx0, h_tgt, idx);
    cb(anchor_h, "mtp_round_anchor_h", -1);
    ggml_set_output(anchor_h);
    res->t_mtp_hpend = anchor_h;

    ggml_tensor * tok = ggml_get_rows(ctx0, mtp.mtp_embd_weights(), ggml_reshape_1d(ctx0, anchor_id, S));
    ggml_tensor * h   = anchor_h;

    // positions [S, K, n_pos]: the sequence's first replay position + n_acc + off, one further per draft
    ggml_tensor * pos_i = nullptr;
    if (inp_pos) {
        ggml_tensor * base_f  = ggml_reshape_1d(ctx0, ggml_cast(ctx0, ggml_view_2d(ctx0, inp_pos, 1, S, V*ps, 0), GGML_TYPE_F32), S);
        ggml_tensor * start   = ggml_add(ctx0, ggml_add(ctx0, base_f, n_acc), off);
        ggml_tensor * steps   = ggml_repeat_4d(ctx0, ggml_reshape_2d(ctx0, ggml_arange(ctx0, 0.0f, (float) K, 1.0f), 1, K), S, K, 1, 1);
        ggml_tensor * pos_f   = ggml_add(ctx0, steps, ggml_reshape_2d(ctx0, start, S, 1));
        ggml_tensor * pos_all = pos_f;
        for (int s = 1; s < n_pos; ++s) {
            pos_all = ggml_concat(ctx0, pos_all, s + 1 == n_pos ? ggml_scale(ctx0, pos_f, 0.0f) : pos_f, 2);
        }
        pos_i = ggml_cast(ctx0, pos_all, GGML_TYPE_I32);
    }
    auto pos_chain = [&](int j) -> ggml_tensor * {
        if (!pos_i) {
            return nullptr;
        }
        const size_t pis = ggml_element_size(pos_i);
        ggml_tensor * p = ggml_view_2d(ctx0, pos_i, S, n_pos, S*K*pis, j*S*pis);
        return ggml_reshape_1d(ctx0, ggml_cont(ctx0, p), S*n_pos);
    };

    ggml_tensor * kq_mask = attn.kq_mask;
    ggml_tensor * k_idxs  = attn.k_idxs;
    const int64_t n_kv    = kq_mask->ne[0];
    GGML_ASSERT(kq_mask->ne[2] == 1 && kq_mask->ne[3] == 1 && "MTP round over several sequences: needs one KV stream");
    ggml_tensor * thr   = ggml_reshape_2d(ctx0, ggml_scale_bias(ctx0, ggml_add(ctx0, n_acc, off), 1.0f, -0.5f), 1, S);
    ggml_tensor * ar    = ggml_repeat_4d(ctx0, ggml_arange(ctx0, 0.0f, (float) V, 1.0f), V, S, 1, 1);
    ggml_tensor * hide  = ggml_scale(ctx0, ggml_step(ctx0, ggml_sub(ctx0, ar, thr)), -30000.0f);
    ggml_tensor * extra = ggml_reshape_3d(ctx0, ggml_scale(ctx0, ggml_arange(ctx0, 0.0f, (float) (n_kv*S), 1.0f), 0.0f), 1, n_kv, S);
    const size_t kis = ggml_element_size(k_idxs);
    extra = ggml_set_rows(ctx0, extra, ggml_reshape_3d(ctx0, hide, 1, V, S), ggml_view_2d(ctx0, k_idxs, V, S, V*kis, 0));
    extra = ggml_cast(ctx0, ggml_reshape_2d(ctx0, extra, n_kv, S), GGML_TYPE_F16);
    cb(extra, "mtp_round_mask", -1);
    auto mask_chain = [&](int j) {
        ggml_tensor * m = ggml_view_2d(ctx0, kq_mask, n_kv, S, kq_mask->nb[1], (size_t) (S*V + j*S)*kq_mask->nb[1]);
        return ggml_add(ctx0, m, extra);
    };

    for (int j = 0; j < K; ++j) {
        ggml_tensor * h_next = mtp.mtp_step(tok, h, pos_chain(j), (int) (S*V + j*S), false, mask_chain(j));
        cb(h_next, "mtp_chain_h", j);

        ggml_tensor * logits_j = mtp.mtp_head(h_next);
        cb(logits_j, "mtp_chain_logits", j);

        ggml_tensor * next = ggml_argmax(ctx0, logits_j);
        cb(next, "mtp_chain_next", j);
        ggml_set_output(next);
        ggml_build_forward_expand(gf, next);
        res->t_mtp_ids.push_back(next);

        if (j + 1 < K) {
            tok = ggml_get_rows(ctx0, mtp.mtp_embd_weights(), next);
            h   = h_next;
        }
    }

    ggml_build_forward_expand(gf, n_acc_i);
    ggml_build_forward_expand(gf, anchor_h);
    res->t_mtp_ids.push_back(n_acc_i);
}

llm_graph_context::dev_anchors llm_graph_context::build_mtp_dev_anchors(ggml_tensor * inp_tokens, int n_chain, int64_t n_embd_h, ggml_tensor * table) const {
    const int     K = n_chain;
    const int     V = n_chain + 1;
    const int64_t S = n_tokens / V;
    GGML_ASSERT(K >= 1 && S >= 1 && S*V == n_tokens && "device anchors: n_chain + 1 rows per sequence");
    GGML_ASSERT(mtp_round && inp_tokens && "device anchors: needs the round state and the verify's tokens");

    // the round's inputs live in the memory of the device the token table belongs to
    ggml_backend_buffer_type_t round_buft = ggml_backend_buffer_get_type(table->buffer);
    if (ggml_backend_dev_t round_dev = ggml_backend_buft_get_device(round_buft)) {
        round_buft = ggml_backend_dev_buffer_type(round_dev);
    }
    const auto & sg = mtp_round_ensure_dev(mtp_round, round_buft, n_embd_h, V, S);

    auto rinp = std::make_unique<llm_graph_input_mtp_round>(mtp_round, sched, n_embd_h, V, S);
    rinp->c_ids  = sg.c_ids_dev;
    rinp->h_tgt  = sg.h_tgt_dev;
    rinp->h_pend = sg.h_pend_dev;
    ggml_tensor * c_ids = rinp->c_ids;
    ggml_tensor * h_tgt = rinp->h_tgt;
    res->add_input(std::move(rinp));

    // accepted drafts: the leading verify rows whose target sample equals the next verify token
    const size_t is = ggml_element_size(inp_tokens);
    ggml_tensor * c_f = ggml_cast(ctx0, ggml_view_2d(ctx0, c_ids,      K, S, V*ggml_element_size(c_ids), 0),  GGML_TYPE_F32);
    ggml_tensor * x_f = ggml_cast(ctx0, ggml_view_2d(ctx0, inp_tokens, K, S, V*is,                       is), GGML_TYPE_F32);
    ggml_tensor * eq  = ggml_scale_bias(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, ggml_sub(ctx0, c_f, x_f)), 0.0f, 1.0f), -1.0f, 1.0f);
    eq = ggml_cont(ctx0, ggml_transpose(ctx0, eq)); // [S, K]
    cb(eq, "mtp_anchor_eq", -1);
    ggml_tensor * n_acc = nullptr;
    ggml_tensor * run   = nullptr;
    for (int i = 0; i < K; ++i) {
        ggml_tensor * e = ggml_view_1d(ctx0, eq, S, i*S*ggml_element_size(eq));
        run   = run   ? ggml_mul(ctx0, run, e)     : e;
        n_acc = n_acc ? ggml_add(ctx0, n_acc, run) : run;
    }
    cb(n_acc, "mtp_anchor_nacc", -1);

    dev_anchors out;
    out.n_acc     = ggml_cast(ctx0, n_acc, GGML_TYPE_I32);
    out.sel_local = ggml_reshape_2d(ctx0, out.n_acc, 1, S);
    out.sel_flat  = ggml_cast(ctx0, ggml_add(ctx0, n_acc, ggml_arange(ctx0, 0.0f, (float) (V*S), (float) V)), GGML_TYPE_I32);
    out.id        = ggml_reshape_1d(ctx0, ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, c_ids, 1, V*S), out.sel_flat), S);
    out.h         = ggml_get_rows(ctx0, h_tgt, out.sel_flat);
    cb(out.h, "mtp_anchor_h", -1);
    return out;
}

void llm_graph_context::build_draft_logits() {
    if (!mm_subst || !draft_head->whole) {
        return;
    }
    mm_subst = nullptr;

    ggml_tensor * logits  = res->t_logits;
    const int64_t n_draft = draft_head->w->ne[1];
    if (!logits) {
        return;
    }
    if (logits->ne[0] != n_draft) {
        res->draft_head_unused = true;
        return;
    }
    if (!ggml_is_contiguous(logits)) {
        logits = ggml_cont(ctx0, logits);
    }
    const int64_t n_vocab = draft_head->src->ne[1];
    const int64_t n_out   = logits->ne[1];
    ggml_tensor * full = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_vocab, n_out), -INFINITY);
    full = ggml_set_rows(ctx0, full, ggml_reshape_3d(ctx0, logits, 1, n_draft, n_out), ggml_reshape_2d(ctx0, draft_head->ids, n_draft, 1));
    full = ggml_reshape_2d(ctx0, full, n_vocab, n_out);
    cb(full, "result_output", -1);
    res->t_logits = full;
    ggml_build_forward_expand(gf, full);
}

ggml_tensor * llm_graph_context::build_inp_pos() const {
    auto inp = std::make_unique<llm_graph_input_pos>(hparams.n_pos_per_embd());

    auto & cur = inp->pos;

    if (loop) {
        // the loop body picks its step's row of the window's positions
        const int64_t n_pos = hparams.n_pos_per_embd();
        inp->device = true;
        ggml_tensor * p = ggml_get_rows(ctx0, loop->pos_win, loop_rows()); // I32 [n_pos, n_tokens]
        if (n_pos > 1 && n_tokens > 1) {
            p = ggml_cont(ctx0, ggml_transpose(ctx0, p)); // section-major, as the host lays it out
        }
        cur = ggml_reshape_1d(ctx0, p, n_pos*n_tokens);
    } else {
        cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t)n_tokens*hparams.n_pos_per_embd());
        ggml_set_input(cur);
    }

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_attn_scale() const {
    auto inp = std::make_unique<llm_graph_input_attn_temp>(hparams.n_attn_temp_floor_scale, hparams.f_attn_temp_scale, hparams.f_attn_temp_offset);

    auto & cur = inp->attn_scale;

    // this need to be 1x1xN for broadcasting
    cur = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, n_tokens);
    ggml_set_input(cur);
    ggml_set_name(cur, "attn_scale");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_out_ids() const {
    // note: when all tokens are output, we could skip this optimization to spare the ggml_get_rows() calls,
    //       but this would make the graph topology depend on the number of output tokens, which can interfere with
    //       features that require constant topology such as pipeline parallelism
    //       ref: https://github.com/ggml-org/llama.cpp/pull/14275#issuecomment-2987424471
    //if (n_outputs < n_tokens) {
    //    return nullptr;
    //}

    auto inp = std::make_unique<llm_graph_input_out_ids>(hparams, cparams, n_outputs);

    auto & cur = inp->out_ids;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_outputs);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_mean() const {
    auto inp = std::make_unique<llm_graph_input_mean>(cparams);

    auto & cur = inp->mean;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, ubatch.n_seqs_unq);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cls() const {
    auto inp = std::make_unique<llm_graph_input_cls>(cparams, arch);

    auto & cur = inp->cls;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_seqs_unq);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cross_embd() const {
    auto inp = std::make_unique<llm_graph_input_cross_embd>(cross);

    auto & cur = inp->cross_embd;

    // if we have the output embeddings from the encoder, use them directly
    // TODO: needs more work to be correct, for now just use the tensor shape
    //if (cross->t_embd) {
    //    cur = ggml_view_tensor(ctx0, cross->t_embd);

    //    return cur;
    //}

    const auto n_embd = !cross->v_embd.empty() ? cross->n_embd : hparams.n_embd_inp();
    const auto n_enc  = !cross->v_embd.empty() ? cross->n_enc  : hparams.n_ctx_train;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_enc);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_enc() const {
    auto inp = std::make_unique<llm_graph_input_pos_bucket>(hparams);

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_tokens);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_dec() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_pos_bucket_kv>(hparams, mctx_cur);

    const auto n_kv = mctx_cur->get_n_kv();

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_tokens);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_pos_bias(ggml_tensor * pos_bucket, ggml_tensor * attn_rel_b) const {
    ggml_tensor * pos_bucket_1d = ggml_reshape_1d(ctx0, pos_bucket, pos_bucket->ne[0] * pos_bucket->ne[1]);
    cb(pos_bucket_1d, "pos_bucket_1d", -1);

    ggml_tensor * pos_bias = ggml_get_rows(ctx0, attn_rel_b, pos_bucket_1d);

    pos_bias = ggml_reshape_3d(ctx0, pos_bias, pos_bias->ne[0], pos_bucket->ne[0], pos_bucket->ne[1]);
    pos_bias = ggml_permute   (ctx0, pos_bias, 2, 0, 1, 3);
    pos_bias = ggml_cont      (ctx0, pos_bias);

    cb(pos_bias, "pos_bias", -1);

    return pos_bias;
}

// the scales of the FP8 cache layer il reads ({0, 0}: none), see llama_kv_cache::cpy_k
static std::pair<float, float> llm_graph_kv_scales(const llama_memory_context_i * mctx, int il) {
    if (const auto * c = dynamic_cast<const llama_kv_cache_iswa_context *>(mctx)) {
        const auto s = c->get_base()->get_kv_scales(il);
        return s.first != 0.0f || s.second != 0.0f ? s : c->get_swa()->get_kv_scales(il);
    }
    if (const auto * c = dynamic_cast<const llama_kv_cache_context *>(mctx)) {
        return c->get_kv_scales(il);
    }
    return { 0.0f, 0.0f };
}

ggml_tensor * llm_graph_context::build_attn_mha(
         ggml_tensor * q,
         ggml_tensor * k,
         ggml_tensor * v,
         ggml_tensor * kq_b,
         ggml_tensor * kq_mask,
         ggml_tensor * sinks,
         ggml_tensor * v_mla,
             int64_t   n_kv_max,
               float   kq_scale,
                 int   il) const {
    const bool v_trans = v->nb[1] > v->nb[2];

    // split the batch into streams if needed
    const auto n_stream = k->ne[3];

    // a ubatch that skips idle sequences: the cache view spans their streams too, so each stream of the span takes
    // its sequence's queries (the idle ones a copy, masked out) and the outputs come back in batch order below
    llm_graph_input_strm * strm_inp = nullptr;
    int64_t strm_tps = 0;
    const llama_ubatch & ub_attn = ubatch_attn ? *ubatch_attn : ubatch;
    if (n_stream > 1 && !cparams.kv_unified && kq_mask && kq_mask->ne[3] == n_stream && ub_attn.n_seqs_unq > 1) {
        const ubatch_streams strm = { (int64_t) ub_attn.n_seqs_unq, n_stream };
        // the queries are the same number of rows for each of the ubatch's sequences as the mask has per stream:
        // the whole ubatch, or one row of each in a drafter's round
        if (strm.n_seqs < n_stream && q->ne[2] == strm.n_seqs*kq_mask->ne[1]) {
            strm_inp = build_inp_strm(strm.n_seqs, strm.n_span);
            strm_tps = q->ne[2]/strm.n_seqs;

            const int64_t ne0 = q->ne[0];
            const int64_t ne1 = q->ne[1];
            if (!ggml_is_contiguous(q)) {
                q = ggml_cont(ctx0, q);
            }
            q = ggml_reshape_2d(ctx0, q, ne0*ne1*strm_tps, strm.n_seqs);
            q = ggml_get_rows(ctx0, q, strm_inp->strm_q);
            q = ggml_reshape_3d(ctx0, q, ne0, ne1, strm_tps*n_stream);
        }
    }

    // the stream dim steps over one stream's worth of dim-2 (tokens): (ne[2]/n_stream) rows of stride nb[2].
    // q->nb[3]/n_stream only equals that for a contiguous q; nope-only MLA (glm5-next) passes a permuted
    // q_absorbed where nb[3] != ne[2]*nb[2], so using nb[3] read another head's queries for streams s >= 1.
    q = ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream, q->nb[1], q->nb[2], q->nb[2]*(q->ne[2]/n_stream), 0);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);

    ggml_tensor * cur;

    const bool use_flash_attn = cparams.flash_attn && kq_b == nullptr;
    if (!use_flash_attn && (k->type == GGML_TYPE_F8_E4M3 || v->type == GGML_TYPE_F8_E4M3)) {
        GGML_ABORT("an FP8 KV cache needs flash attention");
    }
    if (use_flash_attn) {
        GGML_ASSERT(kq_b == nullptr && "Flash attention does not support KQ bias yet");

        if (v_trans) {
            v = ggml_transpose(ctx0, v);
        }

        // this can happen when KV cache is not used (e.g. an embedding model with non-causal attn)
        if (k->type == GGML_TYPE_F32) {
            k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        }

        if (v->type == GGML_TYPE_F32) {
            v = ggml_cast(ctx0, v, GGML_TYPE_F16);
        }

        // an FP8 cache's scales: the K scale into the scores' scale, the V scale on the output
        float kq_scale_fa = kq_scale;
        float v_scale     = 0.0f;
        if (k->type == GGML_TYPE_F8_E4M3 || v->type == GGML_TYPE_F8_E4M3) {
            const auto [sk, sv] = llm_graph_kv_scales(mctx, il);
            if (k->type == GGML_TYPE_F8_E4M3 && sk > 0.0f) {
                kq_scale_fa *= sk;
            }
            if (v->type == GGML_TYPE_F8_E4M3 && sv > 0.0f) {
                v_scale = sv;
            }
        }

        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale_fa, hparams.f_max_alibi_bias,
                                  hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, cur, il});

        ggml_flash_attn_ext_add_sinks(cur, sinks);
        if (fa_kv_strm) {
            // the K/V stream of each sequence comes from this input (the CUDA kernels read it, see fattn-common.cuh)
            cur->src[5] = fa_kv_strm;
        }
        GGML_ASSERT(n_kv_max >= 0 && n_kv_max <= INT32_MAX);
        ggml_flash_attn_ext_set_n_kv_max(cur, static_cast<int32_t>(n_kv_max));
        ggml_prec_set_acc(cur, GGML_PREC_F32);

        if (v_mla) {
#if 0
            // v_mla can be applied as a matrix-vector multiplication with broadcasting across dimension 3 == n_tokens.
            // However, the code is optimized for dimensions 0 and 1 being large, so this is inefficient.
            cur = ggml_reshape_4d(ctx0, cur, v_mla->ne[0], 1, n_head, n_tokens);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
#else
            // It's preferable to do the calculation as a matrix-matrix multiplication with n_tokens in dimension 1.
            // The permutations are noops and only change how the tensor data is interpreted.
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
            cb(cur, "fattn_mla", il);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_cont(ctx0, cur); // Needed because ggml_reshape_2d expects contiguous inputs.
#endif
        }

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
        if (v_scale > 0.0f) {
            cur = ggml_scale(ctx0, cur, v_scale);
        }
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        cb(kq, "kq", il);

        // note: this op tends to require high floating point range
        //       while for some models F16 is enough, for others it is not, so we default to F32 here
        ggml_prec_set_acc(kq, GGML_PREC_F32);

        if (arch == LLM_ARCH_GROK) {
            // need to do the following:
            // multiply by attn_output_multiplier
            // and then :
            // kq = 30 * tanh(kq / 30)
            // before the softmax below

            kq = ggml_tanh(ctx0, ggml_scale(ctx0, kq, hparams.f_attn_out_scale / hparams.f_attn_logit_softcapping));
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled", il);
        }

        if (hparams.attn_soft_cap) {
            kq = ggml_scale(ctx0, kq, 1.0f / hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_1", il);
            kq = ggml_tanh (ctx0, kq);
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_2", il);
        }

        if (kq_b) {
            kq = ggml_add(ctx0, kq, kq_b);
            cb(kq, "kq_plus_kq_b", il);
        }

        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        ggml_soft_max_add_sinks(kq, sinks);
        cb(kq, "kq_soft_max", il);

        if (!v_trans) {
            // note: avoid this branch
            v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
            cb(v, "v_cont", il);
        }

        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        cb(kqv, "kqv", il);

        // for MLA with the absorption optimization, we need to "decompress" from MQA back to MHA
        if (v_mla) {
            kqv = ggml_mul_mat(ctx0, v_mla, kqv);
            cb(kqv, "kqv_mla", il);
        }

        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);

        // recombine streams
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);

        if (!cparams.offload_kqv) {
            // all nodes between the KV store and the attention output are run on the CPU
            ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
        }
    }

    if (strm_inp) {
        const int64_t ne0 = cur->ne[0];
        cur = ggml_reshape_2d(ctx0, cur, ne0*strm_tps, n_stream);
        cur = ggml_get_rows(ctx0, cur, strm_inp->strm_o);
        cur = ggml_reshape_2d(ctx0, cur, ne0, strm_tps*strm_inp->n_seqs);
    }

    ggml_build_forward_expand(gf, cur);

    return cur;
}

llm_graph_input_strm * llm_graph_context::build_inp_strm(int64_t n_seqs, int64_t n_span) const {
    if (strm_group >= 0) {
        GGML_ASSERT(strm_group < 16);
        llm_graph_input_strm *& cached = inp_strm_g[strm_group];
        if (cached) {
            GGML_ASSERT(cached->n_seqs == n_seqs && cached->n_span == n_span);
            return cached;
        }
        auto inp = std::make_unique<llm_graph_input_strm>(n_seqs, n_span, cparams.kv_unified ? 1 : cparams.n_seq_max);
        inp->group = strm_group;
        inp->gmctx = strm_gmctx;
        inp->strm_q = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_span);
        ggml_set_input(inp->strm_q);
        ggml_set_name(inp->strm_q, "inp_strm_q");
        inp->strm_o = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_seqs);
        ggml_set_input(inp->strm_o);
        ggml_set_name(inp->strm_o, "inp_strm_o");
        cached = (llm_graph_input_strm *) res->add_input(std::move(inp));
        return cached;
    }

    if (inp_strm) {
        GGML_ASSERT(inp_strm->n_seqs == n_seqs && inp_strm->n_span == n_span);
        return inp_strm;
    }

    auto inp = std::make_unique<llm_graph_input_strm>(n_seqs, n_span, cparams.kv_unified ? 1 : cparams.n_seq_max);

    inp->strm_q = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_span);
    ggml_set_input(inp->strm_q);
    ggml_set_name(inp->strm_q, "inp_strm_q");

    inp->strm_o = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_seqs);
    ggml_set_input(inp->strm_o);
    ggml_set_name(inp->strm_o, "inp_strm_o");

    inp_strm = (llm_graph_input_strm *) res->add_input(std::move(inp));

    return inp_strm;
}

llm_graph_input_attn_no_cache * llm_graph_context::build_attn_inp_no_cache() const {
    auto inp = std::make_unique<llm_graph_input_attn_no_cache>(hparams, cparams);

    // flash attention requires an f16 mask
    const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    // note: there is no KV cache, so the number of KV values is equal to the number of tokens in the batch
    inp->self_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_tokens, n_tokens, 1, 1);
    ggml_set_input(inp->self_kq_mask);

    inp->self_kq_mask_cnv = inp->self_kq_mask;

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        inp->self_kq_mask_swa = ggml_new_tensor_4d(ctx0, type_mask, n_tokens, n_tokens, 1, 1);
        ggml_set_input(inp->self_kq_mask_swa);

        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;
    } else {
        inp->self_kq_mask_swa     = nullptr;
        inp->self_kq_mask_swa_cnv = nullptr;
    }

    return (llm_graph_input_attn_no_cache *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_no_cache * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    GGML_UNUSED(n_tokens);

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const bool is_swa = hparams.is_swa(il);

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    // [TAG_NO_CACHE_PAD]
    // TODO: if ubatch.equal_seqs() == true, we can split the three tensors below into ubatch.n_seqs_unq streams
    //       but it might not be worth it: https://github.com/ggml-org/llama.cpp/pull/15636
    //assert(!ubatch.equal_seqs() || (k_cur->ne[3] == 1 && k_cur->ne[3] == ubatch.n_seqs_unq));

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_kv> build_attn_inp_kv_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur,
    const llama_loop_state * loop = nullptr,
    ggml_tensor * loop_rows = nullptr) {

    auto inp = std::make_unique<llm_graph_input_attn_kv>(hparams, cparams, mctx_cur);

    if (loop) {
        // the loop body picks its step's cache row and mask row from the window
        GGML_ASSERT(cparams.flash_attn && "loop body: needs flash attention");
        inp->device = true;
        const int64_t n_tokens = ubatch.n_tokens;
        inp->self_k_idxs = ggml_reshape_1d(ctx0, ggml_get_rows(ctx0, loop->kidx_win, loop_rows), n_tokens);
        inp->self_v_idxs = ggml_reshape_1d(ctx0, ggml_get_rows(ctx0, loop->vidx_win, loop_rows), n_tokens);
        ggml_tensor * mask = ggml_get_rows(ctx0, loop->mask_win, loop_rows); // F32 [n_kv, n_tokens]
        inp->self_kq_mask     = ggml_cast(ctx0, mask, GGML_TYPE_F16);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    } else {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur, ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    inp->self_k_rot = mctx_cur->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_cur->build_input_v_rot(ctx0);

    return inp;
}

llm_graph_input_attn_kv * llm_graph_context::build_attn_inp_kv() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur, loop, loop ? loop_rows() : nullptr);

    return (llm_graph_input_attn_kv *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla, // TODO: remove
            float     kq_scale,
            int       il) const {
    GGML_ASSERT(v_mla == nullptr);

    if (inp->self_k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    ggml_tensor * kq_mask = inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (inp->self_v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
    }

    if (wo) {
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_JAIS2) {
            // GLM4, GLM4_MOE, and JAIS2 seem to have numerical issues with half-precision accumulators
            cur = build_lora_mm(wo, cur);
            ggml_prec_set_acc(cur, GGML_PREC_F32);
            if (wo_s) {
                cur = ggml_mul(ctx0, cur, wo_s);
            }
        } else {
            cur = build_lora_mm(wo, cur, wo_s);
        }
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_k> build_attn_inp_k_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_k>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur, ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    return inp;
}

llm_graph_input_attn_k * llm_graph_context::build_attn_inp_k() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_k_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_k *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE) {
            // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
            cur = build_lora_mm(wo, cur);
            ggml_prec_set_acc(cur, GGML_PREC_F32);
            if (wo_s) {
                cur = ggml_mul(ctx0, cur, wo_s);
            }
        } else {
            cur = build_lora_mm(wo, cur, wo_s);
        }
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k_dsa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
        ggml_tensor * top_k,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx->get_mla();

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs_mla();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = inp->get_kq_mask_mla();

    // prepare new kq mask - starts filled with -INFINITY
    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);

    // reshape KQ mask into tensor with rows of size 1:
    // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3], kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    // reshape top_k indices: [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    // prepare zero-filled tensor with rows of size 1: [1, n_top_k, n_batch, n_stream]
    // this will be our source of zero values for unmasking top k mask elements
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    // modify KQ mask by unmasking elements that are in top_k indices
    // ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])
    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);

    // reshape to restore the original shape of KQ mask:
    // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3], kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    // combine with the original kq mask
    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask_top_k, sinks, v_mla, top_k->ne[0], kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv_iswa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    const bool is_swa = hparams.is_swa(il);

    auto * k_rot = is_swa ? inp->self_k_rot_swa : inp->self_k_rot;
    auto * v_rot = is_swa ? inp->self_v_rot_swa : inp->self_v_rot;

    if (k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, k_rot);
        if (k_cur) {
            k_cur = llama_mul_mat_hadamard(ctx0, k_cur, k_rot);
        }
    }
    if (v_rot) {
        if (v_cur) {
            v_cur = llama_mul_mat_hadamard(ctx0, v_cur, v_rot);
        }
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);

    if (k_cur) {
        ggml_build_forward_expand(gf, k_cur);
    }

    if (v_cur) {
        ggml_build_forward_expand(gf, v_cur);
    }

    // A grouped forward in the variable-length layout (vLLM's unified attention): every row stores its K/V and
    // attends in one call, each tile of rows through its sequence's stream from the table
    if (inp->varlen_tab) {
        const auto * mv = is_swa ? inp->mctx->get_swa() : inp->mctx->get_base();
        if (k_cur) {
            ggml_build_forward_expand(gf, mv->cpy_k(ctx0, k_cur, is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs(), il));
        }
        if (v_cur) {
            ggml_build_forward_expand(gf, mv->cpy_v(ctx0, v_cur, is_swa ? inp->get_v_idxs_swa() : inp->get_v_idxs(), il));
        }
        const uint32_t n_kv = (uint32_t) (is_swa ? inp->n_kv_varlen_swa : inp->n_kv_varlen);

        fa_kv_strm = inp->varlen_tab;
        ggml_tensor * cur = build_attn_mha(q_cur, mv->get_k_varlen(ctx0, il, n_kv), mv->get_v_varlen(ctx0, il, n_kv), kq_b,
                is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask(), sinks, v_mla, 0, kq_scale, il);
        fa_kv_strm = nullptr;
        cb(cur, "kqv_out", il);

        if (v_rot) {
            cur = llama_mul_mat_hadamard(ctx0, cur, v_rot);
        }
        if (wo) {
            cur = build_lora_mm(wo, cur, wo_s);
        }
        if (wo_b) {
            cur = ggml_add(ctx0, cur, wo_b);
        }
        return cur;
    }

    // A grouped forward (vLLM's decodes and prefills in one batch): the projections ran over every row; each group's
    // rows store their K/V and attend through the group's own cache view and mask, then the outputs are rejoined
    if (!inp->groups.empty()) {
        ggml_tensor * cur = nullptr;
        int64_t row0 = 0;
        for (size_t g = 0; g < inp->groups.size(); ++g) {
            auto * gi = inp->groups[g];
            const llama_ubatch & gu = gi->mctx->get_ubatch();
            const int64_t n = gu.n_tokens;
            auto rows = [&](ggml_tensor * t) -> ggml_tensor * {
                return t ? ggml_view_3d(ctx0, t, t->ne[0], t->ne[1], n, t->nb[1], t->nb[2], row0*t->nb[2]) : nullptr;
            };
            ggml_tensor * qg = rows(q_cur);
            ggml_tensor * kg = rows(k_cur);
            ggml_tensor * vg = rows(v_cur);

            const auto * mg = is_swa ? gi->mctx->get_swa() : gi->mctx->get_base();
            if (kg) {
                ggml_build_forward_expand(gf, mg->cpy_k(ctx0, kg, is_swa ? gi->get_k_idxs_swa() : gi->get_k_idxs(), il));
            }
            if (vg) {
                ggml_build_forward_expand(gf, mg->cpy_v(ctx0, vg, is_swa ? gi->get_v_idxs_swa() : gi->get_v_idxs(), il));
            }

            ubatch_attn = &gu;
            strm_group  = (int) g;
            strm_gmctx  = gi->mctx;
            fa_kv_strm  = gi->kv_strm;
            ggml_tensor * kv_k = gi->kv_strm ? mg->get_k_stream0(ctx0, il) : mg->get_k(ctx0, il);
            ggml_tensor * kv_v = gi->kv_strm ? mg->get_v_stream0(ctx0, il) : mg->get_v(ctx0, il);
            ggml_tensor * out = build_attn_mha(qg, kv_k, kv_v, kq_b,
                    is_swa ? gi->get_kq_mask_swa() : gi->get_kq_mask(), sinks, v_mla, 0, kq_scale, il);
            ubatch_attn = nullptr;
            strm_group  = -1;
            strm_gmctx  = nullptr;
            fa_kv_strm  = nullptr;

            cur = cur ? ggml_concat(ctx0, cur, out, 1) : out;
            row0 += n;
        }
        cb(cur, "kqv_out", il);

        if (v_rot) {
            cur = llama_mul_mat_hadamard(ctx0, cur, v_rot);
        }
        if (wo) {
            cur = build_lora_mm(wo, cur, wo_s);
        }
        if (wo_b) {
            cur = ggml_add(ctx0, cur, wo_b);
        }
        return cur;
    }

    const auto * mctx_iswa = inp->mctx;

    const auto * mctx_cur = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();

    // optionally store to KV cache
    if (k_cur) {
        const auto & k_idxs = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    if (v_cur) {
        const auto & v_idxs = is_swa ? inp->get_v_idxs_swa() : inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = inp->kv_strm ? mctx_cur->get_k_stream0(ctx0, il) : mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = inp->kv_strm ? mctx_cur->get_v_stream0(ctx0, il) : mctx_cur->get_v(ctx0, il);

    fa_kv_strm = inp->kv_strm;
    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
    fa_kv_strm = nullptr;
    cb(cur, "kqv_out", il);

    if (v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, v_rot);
    }

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k_iswa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    const bool is_swa = hparams.is_swa(il);

    auto * k_rot = is_swa ? inp->self_k_rot_swa : inp->self_k_rot;

    if (k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, k_rot);
        if (k_cur) {
            k_cur = llama_mul_mat_hadamard(ctx0, k_cur, k_rot);
        }
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);

    if (k_cur) {
        ggml_build_forward_expand(gf, k_cur);
    }

    const auto * mctx_iswa = inp->mctx;
    const auto * mctx_cur = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();

    // optionally store to KV cache
    if (k_cur) {
        const auto & k_idxs = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    // MLA-style attention: the cached K is used as V
    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (k_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, k_rot);
    }

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

llm_graph_input_attn_cross * llm_graph_context::build_attn_inp_cross() const {
    auto inp = std::make_unique<llm_graph_input_attn_cross>(cross);

    const int32_t n_enc = !cross->v_embd.empty() ? cross->n_enc : hparams.n_ctx_train;

    // flash attention requires an f16 mask
    const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    inp->cross_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_enc, n_tokens, 1, 1);
    ggml_set_input(inp->cross_kq_mask);

    inp->cross_kq_mask_cnv = inp->cross_kq_mask;

    return (llm_graph_input_attn_cross *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_cross * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const auto & kq_mask = inp->get_kq_mask_cross();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_k_dsa> build_attn_inp_k_dsa_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_dsa_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_k_dsa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs_mla = mctx_cur->get_mla()->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask_mla = build_attn_inp_kq_mask(ctx0, mctx_cur->get_mla(), ubatch, cparams);
        inp->self_kq_mask_mla_cnv = inp->self_kq_mask_mla;
    }

    {
        inp->self_k_idxs_lid = mctx_cur->get_lid()->build_input_k_idxs(ctx0, ubatch);

        // ensure that mask type matches fused lightning indexer use (requires f16 mask)
        auto cparams_copy = cparams;
        cparams_copy.flash_attn = cparams.fused_lid;

        inp->self_kq_mask_lid = build_attn_inp_kq_mask(ctx0, mctx_cur->get_lid(), ubatch, cparams_copy);
        inp->self_kq_mask_lid_cnv = inp->self_kq_mask_lid;

        inp->self_k_rot_lid = mctx_cur->get_lid()->build_input_k_rot(ctx0);
    }

    return inp;
}

llm_graph_input_attn_k_dsa * llm_graph_context::build_attn_inp_k_dsa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsa_context *>(mctx);

    auto inp = build_attn_inp_k_dsa_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_k_dsa *) res->add_input(std::move(inp));
}

llm_graph_input_attn_k_dsa_iswa * llm_graph_context::build_attn_inp_k_dsa_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsa_iswa_context *>(mctx);

    auto inp_dsa = build_attn_inp_k_dsa_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_dsa());

    // build_attn_inp_k_impl rejects SWA caches, so construct the input directly
    auto inp_swa = std::make_unique<llm_graph_input_attn_k>(hparams, cparams, mctx_cur->get_swa());

    inp_swa->self_k_idxs = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);

    inp_swa->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur->get_swa(), ubatch, cparams);
    inp_swa->self_kq_mask_cnv = inp_swa->self_kq_mask;

    auto inp = std::make_unique<llm_graph_input_attn_k_dsa_iswa>(std::move(inp_dsa), std::move(inp_swa), mctx_cur);

    return (llm_graph_input_attn_k_dsa_iswa *) res->add_input(std::move(inp));
}

llm_graph_input_attn_kv_msa * llm_graph_context::build_attn_inp_kv_msa(bool msa_enabled) const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_msa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_kv_msa>(hparams, cparams, mctx_cur);

    const auto * mctx_base = mctx_cur->get_base();
    const auto * mctx_idx  = mctx_cur->get_idx();

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_base->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_base->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_base, ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    inp->self_k_rot = mctx_base->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_base->build_input_v_rot(ctx0);

    if (msa_enabled) {
        inp->self_k_idxs_idx = mctx_idx->build_input_k_idxs(ctx0, ubatch);
    }

    return (llm_graph_input_attn_kv_msa *) res->add_input(std::move(inp));
}

// TODO: maybe separate the inner implementation into a separate function
//       like with the non-sliding window equivalent
//       once sliding-window hybrid caches are a thing.
llm_graph_input_attn_kv_iswa * llm_graph_context::build_attn_inp_kv_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_iswa_context *>(mctx);

    // a grouped forward in the variable-length layout: the rows' indices and masks as one ubatch's, and the table
    int64_t n_kv_varlen     = 0;
    int64_t n_kv_varlen_swa = 0;
    if (llm_graph_attn_varlen(cparams, hparams, mctx_cur, &n_kv_varlen, &n_kv_varlen_swa)) {
        auto inp = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, mctx_cur);
        inp->n_kv_varlen     = n_kv_varlen;
        inp->n_kv_varlen_swa = n_kv_varlen_swa;

        const int64_t n_tokens = ubatch.n_tokens;

        inp->self_k_idxs = mctx_cur->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->get_base()->build_input_v_idxs(ctx0, ubatch);
        GGML_ASSERT(inp->self_v_idxs->ne[0] == n_tokens);
        inp->self_kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, n_kv_varlen, n_tokens, 1, 1);
        ggml_set_input(inp->self_kq_mask);
        ggml_set_name(inp->self_kq_mask, "attn_inp_kq_mask");
        inp->self_kq_mask_cnv = inp->self_kq_mask;

        inp->self_k_idxs_swa = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs_swa = mctx_cur->get_swa()->build_input_v_idxs(ctx0, ubatch);
        inp->self_kq_mask_swa = ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, n_kv_varlen_swa, n_tokens, 1, 1);
        ggml_set_input(inp->self_kq_mask_swa);
        ggml_set_name(inp->self_kq_mask_swa, "attn_inp_kq_mask");
        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;

        inp->varlen_tab = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1 + 3*(int64_t) cparams.n_seq_max);
        ggml_set_input(inp->varlen_tab);
        ggml_set_name(inp->varlen_tab, "inp_varlen_tab");

        inp->self_k_rot = mctx_cur->get_base()->build_input_k_rot(ctx0);
        inp->self_v_rot = mctx_cur->get_base()->build_input_v_rot(ctx0);
        inp->self_k_rot_swa = mctx_cur->get_swa()->build_input_k_rot(ctx0);
        inp->self_v_rot_swa = mctx_cur->get_swa()->build_input_v_rot(ctx0);

        return (llm_graph_input_attn_kv_iswa *) res->add_input(std::move(inp));
    }

    // a grouped forward: the parent, and a child per group built over the group's view and ubatch
    if (mctx_cur->n_groups() > 1) {
        auto parent = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, mctx_cur);
        for (uint32_t g = 0; g < mctx_cur->n_groups(); ++g) {
            const auto * gm = mctx_cur->get_group(g);
            GGML_ASSERT(gm && "grouped forward: the groups' views exist once the context is applied");
            const llama_ubatch & gu = gm->get_ubatch();

            auto inp = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, gm);
            inp->group = (int) g;

            // the group's attention reads each sequence's stream from an input and covers only its sequences, not
            // the streams between them (two prompts in 32 slots computed all 32)
            const bool per_seq = llm_graph_attn_streams_in(cparams);

            inp->self_k_idxs = gm->get_base()->build_input_k_idxs(ctx0, gu);
            inp->self_v_idxs = gm->get_base()->build_input_v_idxs(ctx0, gu);
            inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, gm->get_base(), gu, cparams, per_seq);
            inp->self_kq_mask_cnv = inp->self_kq_mask;

            inp->self_k_idxs_swa = gm->get_swa()->build_input_k_idxs(ctx0, gu);
            inp->self_v_idxs_swa = gm->get_swa()->build_input_v_idxs(ctx0, gu);
            inp->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, gm->get_swa(), gu, cparams, per_seq);
            inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;

            inp->self_k_rot = gm->get_base()->build_input_k_rot(ctx0);
            inp->self_v_rot = gm->get_base()->build_input_v_rot(ctx0);
            inp->self_k_rot_swa = gm->get_swa()->build_input_k_rot(ctx0);
            inp->self_v_rot_swa = gm->get_swa()->build_input_v_rot(ctx0);

            if (per_seq) {
                inp->kv_strm = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t) gm->get_base()->get_streams().size());
                ggml_set_input(inp->kv_strm);
                ggml_set_name(inp->kv_strm, "inp_kv_strm");
            }

            parent->groups.push_back((llm_graph_input_attn_kv_iswa *) res->add_input(std::move(inp)));
        }
        return (llm_graph_input_attn_kv_iswa *) res->add_input(std::move(parent));
    }

    auto inp = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, mctx_cur);

    // a step over some of the slots attends over their streams only, as a group does (vLLM's attention reads each
    // sequence's blocks from its block table): 8 sequences in 32 slots computed all 32
    const bool per_seq = llm_graph_attn_streams_in(cparams);

    {
        inp->self_k_idxs = mctx_cur->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur->get_base(), ubatch, cparams, per_seq);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    {
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache for non-SWA");

        inp->self_k_idxs_swa = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs_swa = mctx_cur->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, mctx_cur->get_swa(), ubatch, cparams, per_seq);
        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;
    }

    if (per_seq) {
        inp->kv_strm = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t) mctx_cur->get_base()->get_streams().size());
        ggml_set_input(inp->kv_strm);
        ggml_set_name(inp->kv_strm, "inp_kv_strm");
    }

    inp->self_k_rot = mctx_cur->get_base()->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_cur->get_base()->build_input_v_rot(ctx0);

    inp->self_k_rot_swa = mctx_cur->get_swa()->build_input_k_rot(ctx0);
    inp->self_v_rot_swa = mctx_cur->get_swa()->build_input_v_rot(ctx0);

    return (llm_graph_input_attn_kv_iswa *) res->add_input(std::move(inp));
}

llm_graph_input_attn_k_iswa * llm_graph_context::build_attn_inp_k_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_iswa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_k_iswa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs = mctx_cur->get_base()->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur->get_base(), ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    {
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache for non-SWA");

        inp->self_k_idxs_swa = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, mctx_cur->get_swa(), ubatch, cparams);
        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;
    }

    inp->self_k_rot = mctx_cur->get_base()->build_input_k_rot(ctx0);

    inp->self_k_rot_swa = mctx_cur->get_swa()->build_input_k_rot(ctx0);

    return (llm_graph_input_attn_k_iswa *) res->add_input(std::move(inp));
}

llm_graph_input_dsv4 * llm_graph_context::build_inp_dsv4() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsv4_context *>(mctx);
    const auto * raw_ctx  = mctx_cur->get_raw();

    auto inp_raw = std::make_unique<llm_graph_input_dsv4_raw>(cparams, raw_ctx);

    const int64_t n_stream = mctx_cur->get_csa_plan(ubatch).n_stream;

    GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "DSV4 expects SWA raw cache");

    inp_raw->self_k_idxs = raw_ctx->build_input_k_idxs(ctx0, ubatch);
    inp_raw->self_kq_mask = dsv4_build_raw_kq_mask(ctx0, raw_ctx, ubatch, cparams, n_stream);
    inp_raw->self_kq_mask_cnv = inp_raw->self_kq_mask;

    inp_raw->self_k_rot = raw_ctx->build_input_k_rot(ctx0);
    auto inp = std::make_unique<llm_graph_input_dsv4>(cparams, std::move(inp_raw), mctx_cur);

    dsv4_build_comp_inputs(ctx0, inp->inp_csa, mctx_cur->get_csa_plan(ubatch), "csa", cparams, n_stream);
    dsv4_build_comp_inputs(ctx0, inp->inp_hca, mctx_cur->get_hca_plan(ubatch), "hca", cparams, n_stream);
    dsv4_build_comp_inputs(ctx0, inp->inp_lid, mctx_cur->get_lid_plan(ubatch), "lid", cparams, n_stream);
    inp->inp_csa.k_rot = mctx_cur->get_csa()->build_input_k_rot(ctx0);
    inp->inp_hca.k_rot = mctx_cur->get_hca()->build_input_k_rot(ctx0);
    inp->inp_lid.k_rot = mctx_cur->get_lid()->build_input_k_rot(ctx0);

    return (llm_graph_input_dsv4 *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        ggml_tensor * s,
        ggml_tensor * state_copy_main,
        ggml_tensor * state_copy_extra,
            int32_t   state_size,
            int32_t   n_seqs,
           uint32_t   n_rs,
           uint32_t   rs_head,
           uint32_t   rs_size,
            int32_t   rs_zero,
        const llm_graph_get_rows_fn & get_state_rows,
        ggml_tensor * step) const {

    GGML_UNUSED(rs_size);
    ggml_tensor * states = ggml_reshape_2d(ctx0, s, state_size, s->ne[1]);

    // Clear a single state which will then be copied to the other cleared states.
    // Note that this is a no-op when the view is zero-sized.
    ggml_tensor * state_zero = ggml_view_1d(ctx0, states, state_size*(rs_zero >= 0), rs_zero*states->nb[1]*(rs_zero >= 0));
    ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0, state_zero, 0));

    // copy states
    // NOTE: assuming the copy destinations are ALL contained between rs_head and rs_head + n_rs
    // {state_size, rs_size} -> {state_size, n_seqs}
    ggml_tensor * output_states = get_state_rows(ctx0, states, state_copy_main);

    // a chained step gathers the states it reads into the step rows of their cells, for seq_rm to put back; only a
    // plain gather is kept, since the state's readers must depend on the step rows
    if (step) {
        ggml_tensor * step_rows = ggml_view_2d(ctx0, step, state_size, n_seqs, step->nb[1], rs_head*step->nb[1]);
        if (output_states->op == GGML_OP_GET_ROWS && output_states->src[0] == states && output_states->src[1] == state_copy_main) {
            output_states = ggml_cpy(ctx0, output_states, step_rows);
            res->n_step_states++;
        } else {
            res->chain_bad = true;
        }
    }
    ggml_build_forward_expand(gf, output_states);

    // copy extra states which won't be changed further (between n_seqs and n_rs)
    ggml_tensor * states_extra = ggml_get_rows(ctx0, states, state_copy_extra);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0,
            states_extra,
            ggml_view_2d(ctx0, s, state_size, (n_rs - n_seqs), s->nb[1], (rs_head + n_seqs)*s->nb[1])));

    return output_states;
}

static std::unique_ptr<llm_graph_input_rs> build_rs_inp_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_memory_recurrent_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_rs>(mctx_cur);

    const int64_t n_rs   = mctx_cur->get_n_rs();
    const int64_t n_seqs = ubatch.n_seqs;

    inp->s_copy = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rs);
    ggml_set_input(inp->s_copy);

    inp->s_copy_main  = ggml_view_1d(ctx0, inp->s_copy, n_seqs, 0);
    inp->s_copy_extra = ggml_view_1d(ctx0, inp->s_copy, n_rs - n_seqs, n_seqs * inp->s_copy->nb[0]);

    inp->head = mctx_cur->get_head();
    inp->rs_z = mctx_cur->get_rs_z();

    return inp;
}

llm_graph_input_rs * llm_graph_context::build_rs_inp() const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    auto inp = build_rs_inp_impl(ctx0, ubatch, mctx_cur);

    return (llm_graph_input_rs *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        llm_graph_input_rs * inp,
        ggml_tensor * s,
            int32_t   state_size,
            int32_t   n_seqs,
        const llm_graph_get_rows_fn & get_state_rows) const {
    const auto * kv_state = inp->mctx;

    ggml_tensor * step = nullptr;
    if (is_chain_form()) {
        step = kv_state->get_step_for(s);
        res->chain_bad |= step == nullptr || step->ne[0] != state_size;
        if (res->chain_bad) {
            step = nullptr;
        }
    }

    return build_rs(s, inp->s_copy_main, inp->s_copy_extra, state_size, n_seqs,
                    kv_state->get_n_rs(), kv_state->get_head(), kv_state->get_size(), kv_state->get_rs_z(),
                    get_state_rows, step);
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_load(
    llm_graph_input_rs * inp,
    const llama_ubatch & ubatch,
                   int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;

    const int64_t n_seqs  = ubatch.n_seqs;

    ggml_tensor * token_shift_all = mctx_cur->get_r_l(il);

    ggml_tensor * token_shift = build_rs(
            inp, token_shift_all,
            hparams.n_embd_r(), n_seqs);

    token_shift = ggml_reshape_3d(ctx0, token_shift, hparams.n_embd, token_shift_count, n_seqs);

    return token_shift;
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_store(
         ggml_tensor * token_shift,
  const llama_ubatch & ubatch,
                 int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;
    const auto n_embd = hparams.n_embd;

    const int64_t n_seqs = ubatch.n_seqs;

    const auto kv_head = mctx_cur->get_head();

    return ggml_cpy(
        ctx0,
        ggml_view_1d(ctx0, token_shift, n_embd * n_seqs * token_shift_count, 0),
        ggml_view_1d(ctx0, mctx_cur->get_r_l(il), hparams.n_embd_r()*n_seqs, hparams.n_embd_r()*kv_head*ggml_element_size(mctx_cur->get_r_l(il)))
    );
}

llm_graph_input_mem_hybrid * llm_graph_context::build_inp_mem_hybrid() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn(), loop, loop ? loop_rows() : nullptr);

    auto inp = std::make_unique<llm_graph_input_mem_hybrid>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid *) res->add_input(std::move(inp));
}

llm_graph_input_mem_hybrid_k * llm_graph_context::build_inp_mem_hybrid_k() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_k_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());

    auto inp = std::make_unique<llm_graph_input_mem_hybrid_k>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid_k *) res->add_input(std::move(inp));
}

llm_graph_input_mem_hybrid_iswa * llm_graph_context::build_inp_mem_hybrid_iswa() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_iswa_context *>(mctx);

    auto inp_rs = build_rs_inp_impl(ctx0, ubatch, mctx_cur->get_recr());

    // build iswa attention input
    const auto * attn_ctx = mctx_cur->get_attn();

    auto inp_attn = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, attn_ctx);

    {
        inp_attn->self_k_idxs = attn_ctx->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs = attn_ctx->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_kq_mask = build_attn_inp_kq_mask(ctx0, attn_ctx->get_base(), ubatch, cparams);
        inp_attn->self_kq_mask_cnv = inp_attn->self_kq_mask;
    }

    {
        inp_attn->self_k_idxs_swa = attn_ctx->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs_swa = attn_ctx->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, attn_ctx->get_swa(), ubatch, cparams);
        inp_attn->self_kq_mask_swa_cnv = inp_attn->self_kq_mask_swa;
    }

    auto inp = std::make_unique<llm_graph_input_mem_hybrid_iswa>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid_iswa *) res->add_input(std::move(inp));
}

void llm_graph_context::build_dense_out(
    ggml_tensor * dense_2,
    ggml_tensor * dense_2_b,
    ggml_tensor * dense_3) const {
    if (!cparams.embeddings || !(dense_2 || dense_2_b || dense_3)) {
        return;
    }
    ggml_tensor * cur = res->t_embd_pooled != nullptr ? res->t_embd_pooled : res->t_embd;
    GGML_ASSERT(cur != nullptr && "missing t_embd_pooled/t_embd");

    if (dense_2) {
        cur = ggml_mul_mat(ctx0, dense_2, cur);
    }
    if (dense_2_b) {
        cur = ggml_add(ctx0, cur, dense_2_b);
    }
    if (dense_3) {
        cur = ggml_mul_mat(ctx0, dense_3, cur);
    }
    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;
    ggml_build_forward_expand(gf, cur);
}


void llm_graph_context::build_pooling(
        ggml_tensor * cls,
        ggml_tensor * cls_b,
        ggml_tensor * cls_out,
        ggml_tensor * cls_out_b,
        ggml_tensor * cls_norm) const {
    if (!cparams.embeddings) {
        return;
    }

    ggml_tensor * inp = res->t_embd;

    //// find result_norm tensor for input
    //for (int i = ggml_graph_n_nodes(gf) - 1; i >= 0; --i) {
    //    inp = ggml_graph_node(gf, i);
    //    if (strcmp(inp->name, "result_norm") == 0 || strcmp(inp->name, "result_embd") == 0) {
    //        break;
    //    }

    //    inp = nullptr;
    //}

    GGML_ASSERT(inp != nullptr && "missing result_norm/result_embd tensor");

    ggml_tensor * cur;

    switch (pooling_type) {
        case LLAMA_POOLING_TYPE_NONE:
            {
                cur = inp;
            } break;
        case LLAMA_POOLING_TYPE_MEAN:
            {
                ggml_tensor * inp_mean = build_inp_mean();
                cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
            } break;
        case LLAMA_POOLING_TYPE_CLS:
        case LLAMA_POOLING_TYPE_LAST:
            {
                ggml_tensor * inp_cls = build_inp_cls();
                cur = ggml_get_rows(ctx0, inp, inp_cls);
            } break;
        case LLAMA_POOLING_TYPE_RANK:
            {
                if (arch == LLM_ARCH_MODERN_BERT) {
                    // modern bert gte reranker builds mean first then applies prediction head and classifier
                    // https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/modular_modernbert.py#L1404-1411
                    ggml_tensor * inp_mean = build_inp_mean();
                    cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
                } else {
                    ggml_tensor * inp_cls = build_inp_cls();
                    cur = ggml_get_rows(ctx0, inp, inp_cls);
                }

                // classification head
                // https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/roberta/modeling_roberta.py#L1566
                if (cls) {
                    cur = ggml_mul_mat(ctx0, cls, cur);
                    if (cls_b) {
                        cur = ggml_add(ctx0, cur, cls_b);
                    }
                    if (arch == LLM_ARCH_MODERN_BERT) {
                        cur = ggml_gelu(ctx0, cur);
                    } else {
                        cur = ggml_tanh(ctx0, cur);
                    }
                    if (cls_norm) {
                        // head norm
                        cur = build_norm(cur, cls_norm, NULL, LLM_NORM, -1);
                    }
                }

                // some models don't have `cls_out`, for example: https://huggingface.co/jinaai/jina-reranker-v1-tiny-en
                // https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/blob/cb5347e43979c3084a890e3f99491952603ae1b7/modeling_bert.py#L884-L896
                // Single layer classification head (direct projection)
                // https://github.com/huggingface/transformers/blob/f4fc42216cd56ab6b68270bf80d811614d8d59e4/src/transformers/models/bert/modeling_bert.py#L1476
                if (cls_out) {
                    cur = ggml_mul_mat(ctx0, cls_out, cur);
                    if (cls_out_b) {
                        cur = ggml_add(ctx0, cur, cls_out_b);
                    }
                }

                // softmax for qwen3 reranker
                if (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_QWEN3VL) {
                    cur = ggml_soft_max(ctx0, cur);
                }
            } break;
        default:
            {
                GGML_ABORT("unknown pooling type");
            }
    }

    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;

    ggml_build_forward_expand(gf, cur);
}

void llm_graph_context::build_sampling() const {
    if (samplers.empty() || !res->t_logits) {
        return;
    }

    std::array<ggml_tensor *, 2> outs;
    outs[0] = res->t_logits;

    std::map<llama_seq_id, std::vector<uint32_t>> sampling_rows;
    std::map<llama_seq_id, std::vector<uint32_t>> sampling_toks;
    uint32_t n_rows = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.output[i]) {
            sampling_rows[ubatch.seq_id[i][0]].push_back(n_rows++);
            sampling_toks[ubatch.seq_id[i][0]].push_back(i);
        }
    }

    // a chunk that samples nothing has no sampling graph: it is the same graph whichever samplers are registered
    if (n_rows == 0) {
        return;
    }

    auto inp_sampling = std::make_unique<llm_graph_input_sampling>(samplers);
    auto * inp_sampling_ptr = inp_sampling.get();
    res->add_input(std::move(inp_sampling));

    // the ids a sequence's row i takes in before it within the batch: the inputs of its output rows 1..i
    auto batch_tokens = [&](llama_seq_id seq_id, uint32_t i) -> ggml_tensor * {
        if (i == 0 || res->t_inp_tokens == nullptr) {
            return nullptr;
        }
        const auto & toks = sampling_toks.at(seq_id);
        bool contiguous = true;
        for (uint32_t j = 2; j <= i; ++j) {
            contiguous = contiguous && toks[j] == toks[1] + j - 1;
        }
        const size_t es = ggml_element_size(res->t_inp_tokens);
        if (contiguous) {
            return ggml_view_1d(ctx0, res->t_inp_tokens, i, toks[1]*es);
        }
        ggml_tensor * ids = ggml_view_1d(ctx0, res->t_inp_tokens, 1, toks[1]*es);
        for (uint32_t j = 2; j <= i; ++j) {
            ids = ggml_concat(ctx0, ids, ggml_view_1d(ctx0, res->t_inp_tokens, 1, toks[j]*es), 0);
        }
        return ids;
    };

    res->t_sampled.resize(n_rows, nullptr);
    res->t_sampled_probs.resize(n_rows, nullptr);
    res->t_sampled_logits.resize(n_rows, nullptr);
    res->t_candidates.resize(n_rows, nullptr);

    // res->t_logits will contain logits for all tokens that want the logits calculated (logits=1 or output=1)
    GGML_ASSERT(res->t_logits != nullptr && "missing t_logits tensor");

    for (const auto & entry : samplers) {
        if (entry.second->iface->backend_reset) {
            entry.second->iface->backend_reset(entry.second);
        }
    }

    // every row takes the argmax of its own logits: one argmax over all rows and one gather of the chosen logits.
    // The argmax kernel treats each row alone, so the ids and logits are the ones the per-row graphs give. Samplers
    // of sequences without a row here play no part, so the graph is the same whichever slots the rows belong to
    const bool greedy_rows = n_rows > 0 && llm_graph_samplers_greedy(ubatch, samplers) &&
        res->t_logits->ne[1] == n_rows && ggml_is_contiguous(res->t_logits);
    res->set_samplers_greedy(greedy_rows);
    inp_sampling_ptr->greedy = greedy_rows;
    if (greedy_rows) {
        // the host reads no row's logits: the argmax and the gather are their only readers, which a backend may fold
        // into the work that produces them
        res->logits_internal = llm_graph_logits_internal(ubatch, samplers);
        const int64_t n_vocab = res->t_logits->ne[0];
        ggml_tensor * ids = ggml_argmax(ctx0, res->t_logits);
        ggml_set_name(ids, "greedy_argmax");
        ggml_tensor * lgs = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, res->t_logits, 1, n_vocab, n_rows), ggml_reshape_2d(ctx0, ids, 1, n_rows));
        ggml_set_name(lgs, "greedy_logit");
        ggml_build_forward_expand(gf, lgs);
        for (uint32_t r = 0; r < n_rows; ++r) {
            ggml_tensor * id = ggml_view_1d(ctx0, ids, 1, r*ids->nb[0]);
            ggml_tensor * lg = ggml_view_1d(ctx0, lgs, 1, r*lgs->nb[2]);
            ggml_build_forward_expand(gf, id);
            ggml_build_forward_expand(gf, lg);
            res->t_sampled[r]        = id;
            res->t_candidates[r]     = id;
            res->t_sampled_logits[r] = lg;
        }
        // the chain form's ids for the next decode, output row r at row r of tok_io
        if (res->chain_ok) {
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, ids, ggml_view_1d(ctx0, chain->tok_io, n_rows, 0)));
        }
        return;
    }

    // the batched form (vLLM's sampler): one top-k over every row's logits wide enough for what the penalties and the
    // bias can push out (they only lower logits, so the top-k after them is among the first k + lw + kp + nb), then
    // the candidates' penalties, the top-k, top-p, min-p, temperature and the draw, each one op over all rows. The
    // parameters, draws and penalty windows are inputs: the graph is the same for any samplers of this shape
    {
        int32_t lw = 0, kp = 0, nb = 0, k = 0;
        bool rep = false, minp = false, t0 = false, sd = false;
        const uint64_t bkey = llm_graph_samplers_batch_key(ubatch, samplers, &lw, &kp, &nb, &k, &rep, &minp, &t0, &sd);
        const int64_t n_vocab = res->t_logits->ne[0];
        const int64_t kc = (int64_t) k + lw + kp + nb;
        if (bkey != 0 && res->t_logits->ne[1] == n_rows && ggml_is_contiguous(res->t_logits) && kc <= n_vocab &&
                (kp == 0 || res->t_inp_tokens != nullptr) && (!sd || res->t_inp_tokens != nullptr)) {
            res->set_samplers_batch(bkey);
            // no row's logits go to the host: no graph output, which lets a backend fold their softcap into the select
            res->logits_internal = llm_graph_logits_internal(ubatch, samplers);
            inp_sampling_ptr->batch_key = bkey;
            inp_sampling_ptr->batch_lw  = lw;
            inp_sampling_ptr->batch_kp  = kp;
            inp_sampling_ptr->batch_nb  = nb;
            const int64_t R = n_rows;

            ggml_tensor * par = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 12, R);
            ggml_set_input(par);
            ggml_set_name(par, "smpl_par");
            inp_sampling_ptr->inp_par = par;
            auto col = [&](int c) {
                return ggml_view_2d(ctx0, par, 1, R, par->nb[1], c*par->nb[0]);
            };

            ggml_tensor * L   = res->t_logits;
            ggml_tensor * top = ggml_top_k(ctx0, L, kc); // [kc, R], unordered
            ggml_set_name(top, "smpl_top");
            ggml_tensor * z = ggml_reshape_2d(ctx0, ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, L, 1, n_vocab, R), top), kc, R);
            // the select and the gather right behind the logits' last op (a backend may fold the two into it)
            ggml_build_forward_expand(gf, z);
            ggml_tensor * top_f = ggml_cast(ctx0, top, GGML_TYPE_F32);

            // [n, kc, R]: 1 where list[j] equals candidate c
            auto equal = [&](ggml_tensor * list, int64_t n) {
                ggml_tensor * shape = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n, kc, R);
                ggml_tensor * diff  = ggml_sub(ctx0, ggml_repeat(ctx0, ggml_reshape_3d(ctx0, list, n, 1, R), shape),
                                                     ggml_repeat(ctx0, ggml_reshape_3d(ctx0, top_f, 1, kc, R), shape));
                ggml_tensor * differ = ggml_step(ctx0, ggml_abs(ctx0, diff));
                return ggml_sub(ctx0, ggml_fill(ctx0, differ, 1.0f), differ);
            };

            if (nb > 0) {
                ggml_tensor * btok = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, nb, R);
                ggml_tensor * bval = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, nb, R);
                ggml_set_input(btok);
                ggml_set_input(bval);
                inp_sampling_ptr->inp_btok = btok;
                inp_sampling_ptr->inp_bval = bval;
                ggml_tensor * w = ggml_mul(ctx0, equal(btok, nb), ggml_reshape_3d(ctx0, bval, nb, 1, R));
                z = ggml_add(ctx0, z, ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, w), kc, R));
            }

            if (lw > 0) {
                ggml_tensor * hist = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, lw, R);
                ggml_set_input(hist);
                ggml_set_name(hist, "smpl_hist");
                inp_sampling_ptr->inp_hist = hist;
                ggml_tensor * h = hist;
                if (kp > 0) {
                    ggml_tensor * ppos  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t) kp*R);
                    ggml_tensor * pmask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, kp, R);
                    ggml_set_input(ppos);
                    ggml_set_input(pmask);
                    inp_sampling_ptr->inp_ppos  = ppos;
                    inp_sampling_ptr->inp_pmask = pmask;
                    // the ids the rows before take in, from the batch's own token input (on the device when drafted
                    // there); -1 where a row takes in fewer
                    ggml_tensor * toks = res->t_inp_tokens;
                    ggml_tensor * pf = ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, toks, 1, ggml_nelements(toks)), ppos);
                    pf = ggml_reshape_2d(ctx0, ggml_cast(ctx0, pf, GGML_TYPE_F32), kp, R);
                    pf = ggml_add(ctx0, ggml_mul(ctx0, pf, pmask), ggml_scale_bias(ctx0, pmask, 1.0f, -1.0f));
                    h = ggml_concat(ctx0, hist, pf, 0);
                }
                // every candidate's count in its row's window, and the penalties of llama_sampler_penalties_apply
                ggml_tensor * cnt = ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, equal(h, lw + kp)), kc, R);
                ggml_tensor * act = ggml_step(ctx0, cnt);
                if (rep) {
                    // logit <= 0: times the penalty, else divided by it
                    ggml_tensor * pos = ggml_step(ctx0, z);
                    ggml_tensor * neg = ggml_sub(ctx0, ggml_fill(ctx0, pos, 1.0f), pos);
                    ggml_tensor * d = ggml_add(ctx0, ggml_mul(ctx0, pos, col(4)), ggml_mul(ctx0, neg, col(5)));
                    z = ggml_mul(ctx0, z, ggml_scale_bias(ctx0, ggml_mul(ctx0, d, act), 1.0f, 1.0f));
                }
                z = ggml_sub(ctx0, z, ggml_mul(ctx0, cnt, col(6)));
                z = ggml_sub(ctx0, z, ggml_mul(ctx0, act, col(7)));
            }

            // the top k in order
            ggml_tensor * ord   = ggml_argsort(ctx0, z, GGML_SORT_ORDER_DESC);
            ggml_tensor * z_all = ggml_reshape_2d(ctx0, ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, z, 1, kc, R), ord), kc, R);
            ggml_tensor * i_all = ggml_reshape_2d(ctx0, ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, top, 1, kc, R), ord), kc, R);
            ggml_tensor * zk = ggml_cont(ctx0, ggml_view_2d(ctx0, z_all, k, R, z_all->nb[1], 0));
            ggml_tensor * ik = ggml_cont(ctx0, ggml_view_2d(ctx0, i_all, k, R, i_all->nb[1], 0));
            ggml_set_name(zk, "smpl_topk_logits");

            // top-p: a candidate stays while the probability before it is under p (llama_sampler_top_p_apply)
            ggml_tensor * probs = ggml_soft_max(ctx0, zk);
            ggml_tensor * excl  = ggml_sub(ctx0, ggml_cumsum(ctx0, probs), probs);
            ggml_tensor * keep  = ggml_step(ctx0, ggml_scale(ctx0, ggml_sub(ctx0, excl, col(0)), -1.0f));
            if (minp) {
                // min-p: logit >= the top logit + log(min-p)
                ggml_tensor * zmax = ggml_view_2d(ctx0, zk, 1, R, zk->nb[1], 0);
                ggml_tensor * thr  = ggml_add(ctx0, zmax, col(1));
                keep = ggml_mul(ctx0, keep, ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_sub(ctx0, zk, thr), 1.0f, 1e-6f)));
            }
            ggml_tensor * zt = ggml_div(ctx0, ggml_add(ctx0, zk, ggml_log(ctx0, keep)), col(2));

            // the draw: the first candidate whose cumulative probability passes the row's uniform
            ggml_tensor * pf   = ggml_soft_max(ctx0, zt);
            ggml_tensor * cdf  = ggml_cumsum(ctx0, pf);
            ggml_tensor * past = ggml_step(ctx0, ggml_sub(ctx0, cdf, col(3)));
            ggml_tensor * idxf = ggml_clamp(ctx0, ggml_sum_rows(ctx0, past), 1.0f, (float) k);
            ggml_tensor * idx  = ggml_cast(ctx0, ggml_scale_bias(ctx0, idxf, -1.0f, (float) k), GGML_TYPE_I32); // [1, R]
            ggml_tensor * ids  = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, ik, 1, k, R), ggml_reshape_2d(ctx0, idx, 1, R));
            ggml_tensor * lgs  = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, zk, 1, k, R), ggml_reshape_2d(ctx0, idx, 1, R));
            ids = ggml_reshape_1d(ctx0, ids, R);
            lgs = ggml_reshape_1d(ctx0, lgs, R);
            if (sd) {
                // speculative sampling (Leviathan et al.): a row that checks a draft d the drafter drew from its q keeps
                // it with probability min(1, p(d)/q(d)), else draws from max(p - q, 0), so its output follows p. With
                // no q from the drafter (a draft the device did not draw) q is d alone: d with probability p(d), else
                // p without d, which follows p as well (and is what comparing a draw with the draft gives)
                constexpr int64_t KQ = LLAMA_MTP_DRAFT_TOP_K;
                ggml_tensor * dpos   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, R);
                ggml_tensor * qstage = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 2*KQ, R);
                ggml_tensor * qidx   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, R);
                ggml_set_input(dpos);
                ggml_set_input(qstage);
                ggml_set_input(qidx);
                ggml_set_name(qstage, "smpl_qstage");
                inp_sampling_ptr->inp_dpos   = dpos;
                inp_sampling_ptr->inp_qstage = qstage;
                inp_sampling_ptr->inp_qidx   = qidx;
                inp_sampling_ptr->tokens_dev = tokens_dev;
                inp_sampling_ptr->sched      = sched;
                // the stage on the device: the drafter's distributions come over as a device copy
                for (int ib = 0; sched && ib < ggml_backend_sched_get_n_backends(sched); ++ib) {
                    ggml_backend_t b = ggml_backend_sched_get_backend(sched, ib);
                    if (ggml_backend_dev_type(ggml_backend_get_device(b)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                        ggml_backend_sched_set_tensor_backend(sched, qstage, b);
                        break;
                    }
                }
                ggml_tensor * qinfo = ggml_get_rows(ctx0, qstage, qidx); // [2*KQ, R]

                auto same = [&](ggml_tensor * a, ggml_tensor * b) {
                    ggml_tensor * differ = ggml_step(ctx0, ggml_abs(ctx0, ggml_sub(ctx0, a, b)));
                    return ggml_sub(ctx0, ggml_fill(ctx0, differ, 1.0f), differ);
                };
                ggml_tensor * toks = res->t_inp_tokens;
                ggml_tensor * d    = ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, toks, 1, ggml_nelements(toks)), dpos);
                ggml_tensor * d_f  = ggml_reshape_2d(ctx0, ggml_cast(ctx0, d, GGML_TYPE_F32), 1, R);
                ggml_tensor * ik_f = ggml_cast(ctx0, ik, GGML_TYPE_F32);
                ggml_tensor * m_d  = same(ik_f, d_f);                              // [k, R]: the draft among p's candidates
                ggml_tensor * p_d  = ggml_sum_rows(ctx0, ggml_mul(ctx0, pf, m_d)); // [1, R]
                ggml_tensor * qi   = ggml_cont(ctx0, ggml_view_2d(ctx0, qinfo, KQ, R, qinfo->nb[1], 0));
                ggml_tensor * qp   = ggml_cont(ctx0, ggml_view_2d(ctx0, qinfo, KQ, R, qinfo->nb[1], KQ*qinfo->nb[0]));
                ggml_tensor * q_d  = ggml_sum_rows(ctx0, ggml_mul(ctx0, qp, same(qi, d_f)));
                // q at each of p's candidates
                ggml_tensor * shp  = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, KQ, k, R);
                ggml_tensor * mq   = same(ggml_repeat(ctx0, ggml_reshape_3d(ctx0, qi, KQ, 1, R), shp),
                                          ggml_repeat(ctx0, ggml_reshape_3d(ctx0, ik_f, 1, k, R), shp));
                ggml_tensor * qk   = ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, ggml_mul(ctx0, mq,
                                          ggml_repeat(ctx0, ggml_reshape_3d(ctx0, qp, KQ, 1, R), shp))), k, R);
                ggml_tensor * hq   = ggml_cont(ctx0, col(11)); // contiguous: scale reads its source as such
                ggml_tensor * nhq  = ggml_scale_bias(ctx0, hq, -1.0f, 1.0f);
                qk  = ggml_add(ctx0, ggml_mul(ctx0, qk, hq), ggml_mul(ctx0, m_d, nhq));
                ggml_tensor * qd   = ggml_add(ctx0, ggml_mul(ctx0, q_d, hq), nhq);
                // d kept when u q(d) < p(d), else the draw from the residual with the second uniform
                ggml_tensor * a    = ggml_reshape_1d(ctx0, ggml_step(ctx0, ggml_sub(ctx0, p_d, ggml_mul(ctx0, qd, col(3)))), R);
                ggml_tensor * na   = ggml_scale_bias(ctx0, a, -1.0f, 1.0f);
                ggml_tensor * rr   = ggml_relu(ctx0, ggml_sub(ctx0, pf, qk));
                ggml_tensor * rs   = ggml_sum_rows(ctx0, rr);
                ggml_tensor * pst  = ggml_step(ctx0, ggml_sub(ctx0, ggml_cumsum(ctx0, rr), ggml_mul(ctx0, rs, col(10))));
                ggml_tensor * i2f  = ggml_clamp(ctx0, ggml_sum_rows(ctx0, pst), 1.0f, (float) k);
                ggml_tensor * i2   = ggml_reshape_2d(ctx0, ggml_cast(ctx0, ggml_scale_bias(ctx0, i2f, -1.0f, (float) k), GGML_TYPE_I32), 1, R);
                ggml_tensor * id2  = ggml_reshape_1d(ctx0, ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, ik, 1, k, R), i2), R);
                ggml_tensor * lg2  = ggml_reshape_1d(ctx0, ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, zk, 1, k, R), i2), R);
                ggml_tensor * lgd  = ggml_reshape_1d(ctx0, ggml_sum_rows(ctx0, ggml_mul(ctx0, zk, m_d)), R);
                ggml_tensor * idd  = ggml_add(ctx0, ggml_mul(ctx0, ggml_reshape_1d(ctx0, d_f, R), a),
                                                    ggml_mul(ctx0, ggml_cast(ctx0, id2, GGML_TYPE_F32), na));
                ggml_tensor * lgdd = ggml_add(ctx0, ggml_mul(ctx0, lgd, a), ggml_mul(ctx0, lg2, na));
                // rows without a draft keep their own draw
                ggml_tensor * hd   = ggml_reshape_1d(ctx0, ggml_cont(ctx0, col(9)), R);
                ggml_tensor * nhd  = ggml_scale_bias(ctx0, hd, -1.0f, 1.0f);
                ggml_tensor * idf  = ggml_add(ctx0, ggml_mul(ctx0, idd, hd), ggml_mul(ctx0, ggml_cast(ctx0, ids, GGML_TYPE_F32), nhd));
                ids = ggml_cast(ctx0, idf, GGML_TYPE_I32);
                lgs = ggml_add(ctx0, ggml_mul(ctx0, lgdd, hd), ggml_mul(ctx0, lgs, nhd));
            }
            if (t0) {
                // temperature 0 takes the argmax as the greedy sampler does: the lowest id among the largest logits
                // (all of which are candidates), from the candidates after the penalties
                ggml_tensor * am  = ggml_argmax(ctx0, z);
                ggml_tensor * zm  = ggml_reshape_2d(ctx0, ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, z, 1, kc, R), ggml_reshape_2d(ctx0, am, 1, R)), 1, R);
                ggml_tensor * low = ggml_step(ctx0, ggml_scale(ctx0, ggml_sub(ctx0, z, zm), -1.0f)); // 1 below the largest
                ggml_tensor * sc  = ggml_mul(ctx0, ggml_sub(ctx0, ggml_fill(ctx0, low, 1.0f), low), ggml_scale_bias(ctx0, top_f, -1.0f, 16777216.0f));
                ggml_tensor * gi  = ggml_argmax(ctx0, sc);
                ggml_tensor * gid = ggml_reshape_1d(ctx0, ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, top, 1, kc, R), ggml_reshape_2d(ctx0, gi, 1, R)), R);
                ggml_tensor * f   = ggml_reshape_1d(ctx0, ggml_cont(ctx0, col(8)), R);
                ggml_tensor * nf  = ggml_scale_bias(ctx0, f, -1.0f, 1.0f);
                ggml_tensor * idf = ggml_add(ctx0, ggml_mul(ctx0, ggml_cast(ctx0, ids, GGML_TYPE_F32), nf), ggml_mul(ctx0, ggml_cast(ctx0, gid, GGML_TYPE_F32), f));
                ids = ggml_cast(ctx0, idf, GGML_TYPE_I32);
                lgs = ggml_add(ctx0, ggml_mul(ctx0, lgs, nf), ggml_mul(ctx0, ggml_reshape_1d(ctx0, zm, R), f));
            }
            ggml_set_name(ids, "smpl_ids");
            ggml_build_forward_expand(gf, ids);
            ggml_build_forward_expand(gf, lgs);
            for (uint32_t r = 0; r < n_rows; ++r) {
                ggml_tensor * id = ggml_view_1d(ctx0, ids, 1, r*ids->nb[0]);
                ggml_tensor * lg = ggml_view_1d(ctx0, lgs, 1, r*lgs->nb[0]);
                ggml_build_forward_expand(gf, id);
                ggml_build_forward_expand(gf, lg);
                res->t_sampled[r]        = id;
                res->t_candidates[r]     = id;
                res->t_sampled_logits[r] = lg;
            }
            if (res->chain_ok) {
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, ids, ggml_view_1d(ctx0, chain->tok_io, n_rows, 0)));
            }
            return;
        }
    }
    // the per-row chains below read their samplers: no replay by other sampler objects
    res->set_samplers_batch(0);

    // add a dummy row to keep the single-output graph static regardless of active samplers
    // multi-output graphs can still vary with the number of output rows
    ggml_tensor * logits_t = ggml_pad(ctx0, res->t_logits, 0, 1, 0, 0);

    static const std::vector<uint32_t> dummy_row = { 0 };

    for (const auto & [seq_id, sampler] : samplers) {
        const auto it = sampling_rows.find(seq_id);

        // inactive samplers always work on the first row
        const bool active = it != sampling_rows.end();
        const auto & rows = active ? it->second : dummy_row;
        const int i_out   = active ? 1          : 0;

        for (uint32_t i = 0; i < rows.size(); ++i) {
            ggml_tensor * logits_seq = ggml_view_1d(ctx0, logits_t, logits_t->ne[0], rows[i] * logits_t->nb[1]);
            ggml_format_name(logits_seq, "logits_seq_%d_%u", seq_id, i);

            struct llama_sampler_data data = {
                /*.logits       =*/ logits_seq,
                /*.probs        =*/ nullptr,
                /*.sampled      =*/ nullptr,
                /*.candidates   =*/ nullptr,
                /*.batch_tokens =*/ active ? batch_tokens(seq_id, i) : nullptr,
            };

            assert(sampler->iface->backend_apply);
            sampler->iface->backend_apply(sampler, ctx0, gf, &data);

            if (data.sampled != nullptr) {
                if (active) {
                    res->t_sampled[rows[i]] = data.sampled;
                }
                outs[1] = data.sampled;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }

            if (data.probs != nullptr) {
                if (active) {
                    res->t_sampled_probs[rows[i]] = data.probs;
                }
                outs[1] = data.probs;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }

            if (data.logits != nullptr) {
                if (active) {
                    res->t_sampled_logits[rows[i]] = data.logits;
                }
                outs[1] = data.logits;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }

            if (data.candidates != nullptr) {
                if (active) {
                    res->t_candidates[rows[i]] = data.candidates;
                }
                outs[1] = data.candidates;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }
        }
    }

    // in the chain form every sampled id goes to its output row of tok_io, where the next decode can take it
    if (res->chain_ok) {
        for (uint32_t r = 0; r < n_rows; ++r) {
            ggml_tensor * t = res->t_sampled[r];
            if (t == nullptr) {
                continue;
            }
            if (ggml_nelements(t) != 1 || t->type != GGML_TYPE_I32) {
                res->chain_ok = false;
                break;
            }
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, t, ggml_view_1d(ctx0, chain->tok_io, 1, r*ggml_element_size(chain->tok_io))));
        }
    }

    // TODO: Call backend_accept after all samplers have been applied.
    /*
    for (const auto & [seq_id, sampler] : samplers) {
        const auto it = sampling_rows.find(seq_id);
        if (it == sampling_rows.end()) {
            continue;
        }

        for (uint32_t row : it->second) {
            ggml_tensor * selected_token = res->t_sampled[row];
            if (selected_token != nullptr && sampler->iface->backend_accept) {
                sampler->iface->backend_accept(sampler, ctx0, gf, selected_token);
            }
        }
    }
    */
}

int32_t llama_relative_position_bucket(llama_pos x, llama_pos y, uint64_t n_buckets, bool bidirectional) {
    // TODO move to hparams if a T5 variant appears that uses a different value
    const int64_t max_distance = 128;

    if (bidirectional) {
        n_buckets >>= 1;
    }

    const int64_t max_exact = n_buckets >> 1;

    int32_t relative_position = x - y;
    int32_t relative_bucket = 0;

    if (bidirectional) {
        relative_bucket += (relative_position > 0) * n_buckets;
        relative_position = std::abs(relative_position);
    } else {
        relative_position = -std::min<int32_t>(relative_position, 0);
    }

    int32_t relative_position_if_large = floorf(max_exact + logf(1.0 * relative_position / max_exact) * (n_buckets - max_exact) / log(1.0 * max_distance / max_exact));
    relative_position_if_large = std::min<int32_t>(relative_position_if_large, n_buckets - 1);
    relative_bucket += (relative_position < max_exact ? relative_position : relative_position_if_large);

    return relative_bucket;
}
