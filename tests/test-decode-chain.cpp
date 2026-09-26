// chained decodes must give the serial tokens, and seq_rm of the step queued past the last read must undo it exactly;
// skips unless the context can chain (a GPU, LLAMA_INPUT_ON_DEVICE=1, LLAMA_DECODE_CHAIN_RECURRENT=1 for recurrent)

#include "llama.h"
#include "../src/llama-ext.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const char * prompts[] = {
    "The history of the printing press begins",
    "Write a short function that reverses a linked list in C.",
};

static constexpr int32_t n_seq = 2;

struct run_state {
    llama_context * ctx = nullptr;
    llama_sampler * smpl[n_seq] = {};
    llama_pos       pos[n_seq]  = {};
};

static bool init_ctx(llama_model * model, run_state & st) {
    llama_sampler_seq_config configs[n_seq];
    for (int32_t s = 0; s < n_seq; ++s) {
        st.smpl[s] = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(st.smpl[s], llama_sampler_init_greedy());
        configs[s] = { s, st.smpl[s] };
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = 1024;
    cparams.n_batch    = 512;
    cparams.n_seq_max  = n_seq;
    cparams.samplers   = configs;
    cparams.n_samplers = n_seq;

    st.ctx = llama_init_from_model(model, cparams);
    return st.ctx != nullptr;
}

static void free_ctx(run_state & st) {
    llama_free(st.ctx);
    for (auto * s : st.smpl) {
        llama_sampler_free(s);
    }
}

// one row per sequence at its next position, ids from tok (or a placeholder when the rows are armed)
static int decode_rows(run_state & st, llama_batch & batch, const llama_token * tok) {
    batch.n_tokens = n_seq;
    for (int32_t s = 0; s < n_seq; ++s) {
        batch.token[s]     = tok ? tok[s] : 0;
        batch.pos[s]       = st.pos[s]++;
        batch.n_seq_id[s]  = 1;
        batch.seq_id[s][0] = s;
        batch.logits[s]    = true;
    }
    return llama_decode(st.ctx, batch);
}

static bool decode_prompts(run_state & st, const llama_vocab * vocab, llama_token * tok) {
    llama_batch batch = llama_batch_init(512, 0, 1);
    int32_t idx[n_seq];
    for (int32_t s = 0; s < n_seq; ++s) {
        std::vector<llama_token> ids(256);
        const int n = llama_tokenize(vocab, prompts[s], strlen(prompts[s]), ids.data(), ids.size(), true, false);
        GGML_ASSERT(n > 0);
        for (int i = 0; i < n; ++i) {
            const int32_t k = batch.n_tokens++;
            batch.token[k]     = ids[i];
            batch.pos[k]       = st.pos[s]++;
            batch.n_seq_id[k]  = 1;
            batch.seq_id[k][0] = s;
            batch.logits[k]    = i == n - 1;
        }
        idx[s] = batch.n_tokens - 1;
    }
    const bool ok = llama_decode(st.ctx, batch) == 0;
    for (int32_t s = 0; ok && s < n_seq; ++s) {
        tok[s] = llama_get_sampled_token_ith(st.ctx, idx[s]);
    }
    llama_batch_free(batch);
    return ok;
}

int main(int argc, char ** argv) {
    std::string model_path;
    int32_t n_gen    = 48;
    int32_t n_follow = 16;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            n_gen = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-ngl") && i + 1 < argc) {
            ++i;
        }
    }
    if (model_path.empty()) {
        fprintf(stderr, "usage: %s -m MODEL [-n N_GEN]\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "failed to load %s, skipping\n", model_path.c_str());
        return 0;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    run_state ser;
    run_state chn;
    GGML_ASSERT(init_ctx(model, ser));
    GGML_ASSERT(init_ctx(model, chn));

    if (!llama_decode_chain_enable(chn.ctx)) {
        fprintf(stderr, "the context cannot chain, skipping\n");
        free_ctx(ser);
        free_ctx(chn);
        llama_model_free(model);
        return 0;
    }

    llama_batch batch = llama_batch_init(n_seq, 0, 1);

    // serial reference: every decode's ids are read before the next one
    std::vector<llama_token> out_ser[n_seq];
    {
        llama_token tok[n_seq];
        GGML_ASSERT(decode_prompts(ser, vocab, tok));
        for (int32_t k = 0; k < n_gen; ++k) {
            for (int32_t s = 0; s < n_seq; ++s) {
                out_ser[s].push_back(tok[s]);
            }
            GGML_ASSERT(decode_rows(ser, batch, tok) == 0);
            for (int32_t s = 0; s < n_seq; ++s) {
                tok[s] = llama_get_sampled_token_ith(ser.ctx, s);
            }
        }
        for (int32_t s = 0; s < n_seq; ++s) {
            out_ser[s].push_back(tok[s]);
        }
    }

    // chained: each decode is queued, taking its ids on the device, before the last one's are read
    std::vector<llama_token> out_chn[n_seq];
    {
        llama_token tok[n_seq];
        GGML_ASSERT(decode_prompts(chn, vocab, tok));
        for (int32_t s = 0; s < n_seq; ++s) {
            out_chn[s].push_back(tok[s]);
        }
        GGML_ASSERT(decode_rows(chn, batch, tok) == 0);

        int32_t rows[n_seq];
        for (int32_t s = 0; s < n_seq; ++s) {
            rows[s] = s;
        }

        if (!llama_set_tokens_sampled(chn.ctx, rows, rows, n_seq)) {
            fprintf(stderr, "the graph does not take the chain form, skipping\n");
            llama_batch_free(batch);
            free_ctx(ser);
            free_ctx(chn);
            llama_model_free(model);
            return 0;
        }

        // an armed decode with another layout does not run and leaves the chain as it was
        GGML_ASSERT(llama_set_tokens_sampled(chn.ctx, rows, rows, 1));
        batch.n_tokens = 1;
        batch.token[0] = 0; batch.pos[0] = chn.pos[0]; batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = true;
        GGML_ASSERT(llama_decode(chn.ctx, batch) == 3);

        // the last round queues one step past the last read, which the seq_rm below undoes
        for (int32_t k = 1; k <= n_gen; ++k) {
            GGML_ASSERT(llama_set_tokens_sampled(chn.ctx, rows, rows, n_seq));
            GGML_ASSERT(decode_rows(chn, batch, nullptr) == 0);
            GGML_ASSERT(!llama_set_tokens_sampled(chn.ctx, rows, rows, n_seq));
            llama_outputs_wait(chn.ctx);
            for (int32_t s = 0; s < n_seq; ++s) {
                out_chn[s].push_back(llama_get_sampled_token_ith(chn.ctx, s));
            }
            llama_outputs_advance(chn.ctx);
        }
        for (int32_t s = 0; s < n_seq; ++s) {
            chn.pos[s]--;
            GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(chn.ctx), s, chn.pos[s], -1));
        }
    }

    int n_fail = 0;
    for (int32_t s = 0; s < n_seq; ++s) {
        if (out_ser[s] != out_chn[s]) {
            fprintf(stderr, "seq %d: chained tokens differ from serial\n", s);
            n_fail++;
        }
    }

    // a token other than the undone step's at its position: any trace of that step would change what follows
    {
        llama_token tok_ser[n_seq];
        llama_token tok_chn[n_seq];
        for (int32_t s = 0; s < n_seq; ++s) {
            GGML_ASSERT(ser.pos[s] == chn.pos[s]);
            tok_ser[s] = tok_chn[s] = out_ser[s][0];
        }
        for (int32_t k = 0; k < n_follow; ++k) {
            GGML_ASSERT(decode_rows(ser, batch, tok_ser) == 0);
            GGML_ASSERT(decode_rows(chn, batch, tok_chn) == 0);
            for (int32_t s = 0; s < n_seq; ++s) {
                tok_ser[s] = llama_get_sampled_token_ith(ser.ctx, s);
                tok_chn[s] = llama_get_sampled_token_ith(chn.ctx, s);
                if (tok_ser[s] != tok_chn[s]) {
                    fprintf(stderr, "seq %d: after the rollback, step %d gives %d, serial %d\n", s, k, tok_chn[s], tok_ser[s]);
                    n_fail++;
                    k = n_follow;
                }
            }
        }
    }

    llama_batch_free(batch);
    free_ctx(ser);
    free_ctx(chn);
    llama_model_free(model);
    llama_backend_free();

    fprintf(stderr, "%s: %d sequences, %d chained steps, %d follow-up steps: %s\n", __func__, n_seq, n_gen, n_follow,
            n_fail == 0 ? "OK" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
