// The gate for the device decode loop: the same prompt, greedy, through the
// host loop (one llama_decode per token) and through llama_decode_loop; the
// ids must match, and both are timed.
//
//   llama-loop-test <model.gguf> [n_tokens]

#include "llama.h"
#include "../../src/llama-ext.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static const char * PROMPT = "Write a Go function that walks a directory tree concurrently with a worker pool and returns the ten largest files. Handle errors and include the struct definitions.\n";

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static llama_context * make_ctx(llama_model * model, int n_ctx) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = n_ctx;
    cp.n_batch    = 512;
    cp.n_ubatch   = 512;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_rs_seq   = 0;
    cp.n_seq_max  = 1;
    cp.n_outputs_max = 1;
    cp.n_outputs_max_per_seq = 1;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx) {
        // the backend takes a chain, greedy inside it
        llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
        if (!llama_set_sampler(ctx, 0, chain)) {
            fprintf(stderr, "the sampler did not go to the backend\n");
            exit(1);
        }
    }
    return ctx;
}

// feeds the prompt, returns the first sampled id
static llama_token feed_prompt(llama_context * ctx, std::vector<llama_token> & toks) {
    llama_batch b = llama_batch_get_one(toks.data(), (int32_t) toks.size());
    if (llama_decode(ctx, b) != 0) {
        fprintf(stderr, "prompt decode failed\n");
        exit(1);
    }
    return llama_get_sampled_token_ith(ctx, -1);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [n_tokens]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const int n_gen = argc > 2 ? atoi(argv[2]) : 64;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> toks(512);
    const int n_tok = llama_tokenize(vocab, PROMPT, (int32_t) strlen(PROMPT), toks.data(), (int32_t) toks.size(), true, true);
    if (n_tok <= 0) {
        fprintf(stderr, "tokenize failed\n");
        return 1;
    }
    toks.resize(n_tok);

    // ---- the host loop
    std::vector<llama_token> host;
    double host_ms = 0;
    {
        llama_context * ctx = make_ctx(model, 4096);
        llama_token cur = feed_prompt(ctx, toks);
        llama_pos pos = n_tok;
        const double t0 = now_ms();
        for (int i = 0; i < n_gen; ++i) {
            llama_batch b = llama_batch_get_one(&cur, 1);
            b.pos = &pos;
            if (llama_decode(ctx, b) != 0) {
                fprintf(stderr, "host decode failed at %d\n", i);
                return 1;
            }
            cur = llama_get_sampled_token_ith(ctx, -1);
            host.push_back(cur);
            pos++;
        }
        host_ms = now_ms() - t0;
        llama_free(ctx);
    }

    // ---- the device loop
    std::vector<llama_token> dev(n_gen, -1);
    double dev_ms = 0;
    {
        llama_context * ctx = make_ctx(model, 4096);
        llama_token first = feed_prompt(ctx, toks);
        const double t0 = now_ms();
        const int32_t n = llama_decode_loop(ctx, first, n_tok, n_gen, -1, dev.data());
        dev_ms = now_ms() - t0;
        if (n != n_gen) {
            fprintf(stderr, "device loop returned %d\n", n);
            return 1;
        }
        printf("loop ran %s\n", llama_decode_loop_on_device(ctx) ? "as one device graph" : "stepped by the host");
        llama_free(ctx);
    }

    int same = 0;
    for (int i = 0; i < n_gen; ++i) {
        if (host[i] == dev[i]) {
            same++;
        } else {
            break;
        }
    }
    printf("prompt %d tokens, %d generated\n", n_tok, n_gen);
    printf("host loop:   %.1f ms, %.1f tok/s\n", host_ms, n_gen * 1000.0 / host_ms);
    printf("device loop: %.1f ms, %.1f tok/s\n", dev_ms,  n_gen * 1000.0 / dev_ms);
    printf("identical prefix: %d of %d\n", same, n_gen);
    if (same < n_gen) {
        printf("first difference at %d: host %d device %d\n", same, host[same], dev[same]);
    }

    std::string text;
    for (int i = 0; i < n_gen && i < 40; ++i) {
        char buf[64];
        const int n = llama_token_to_piece(vocab, dev[i], buf, sizeof(buf), 0, true);
        if (n > 0) {
            text.append(buf, n);
        }
    }
    printf("device text: %s\n", text.c_str());

    llama_model_free(model);
    llama_backend_free();
    return same == n_gen ? 0 : 2;
}
