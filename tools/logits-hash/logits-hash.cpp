// The bit-exact gate for engine changes: the prompt, N greedy tokens one at a time, then the same
// tokens again as batches of 2, 4 and 8 (the verify shapes), with every logits row hashed. Two builds
// agree when every line agrees.
//
//   llama-logits-hash <model.gguf> [n_tokens] [prompt_repeat] [dump.bin]

#include "llama.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static const char * PROMPT = "Write a Go function that walks a directory tree concurrently with a worker pool and returns the ten "
    "largest files. Handle errors and symlink loops, keep memory bounded, and explain the design choices. ";

static uint64_t fnv(const void * p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 1099511628211ull;
    }
    return h;
}

static llama_context * make_ctx(llama_model * model) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = 4096;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_seq_max = 1;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "context failed\n");
        exit(1);
    }
    return ctx;
}

struct hasher {
    int n_vocab;
    FILE * dump;
    uint64_t all = 1469598103934665603ull;
    int rows = 0;
    uint64_t row(const float * logits) {
        const uint64_t h = fnv(logits, (size_t) n_vocab * sizeof(float));
        all = fnv(&h, sizeof(h), all);
        rows++;
        if (dump) {
            fwrite(logits, sizeof(float), n_vocab, dump);
        }
        return h;
    }
};

static int argmax(const float * logits, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return best;
}

// decodes toks[0..n) at pos0 with logits on every row, hashes each row
static void decode_rows(llama_context * ctx, hasher & hs, const std::vector<llama_token> & toks, llama_pos pos0, const char * tag) {
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        b.token[i] = toks[i];
        b.pos[i] = pos0 + (llama_pos) i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = 1;
    }
    b.n_tokens = (int32_t) toks.size();
    if (llama_decode(ctx, b) != 0) {
        fprintf(stderr, "%s decode failed\n", tag);
        exit(1);
    }
    for (size_t i = 0; i < toks.size(); ++i) {
        const uint64_t h = hs.row(llama_get_logits_ith(ctx, (int32_t) i));
        printf("%s row %zu tok %d hash %016llx\n", tag, i, toks[i], (unsigned long long) h);
    }
    llama_batch_free(b);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [n_tokens] [prompt_repeat] [dump.bin]\n", argv[0]);
        return 1;
    }
    const int n_gen    = argc > 2 ? atoi(argv[2]) : 32;
    const int n_repeat = argc > 3 ? atoi(argv[3]) : 1;
    FILE * dump        = argc > 4 ? fopen(argv[4], "wb") : nullptr;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    hasher hs{n_vocab, dump};

    std::string prompt;
    for (int i = 0; i < n_repeat; ++i) {
        prompt += PROMPT;
    }
    std::vector<llama_token> toks(prompt.size() + 16);
    const int n_tok = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), toks.data(), (int32_t) toks.size(), true, true);
    if (n_tok <= 0) {
        fprintf(stderr, "tokenize failed\n");
        return 1;
    }
    toks.resize(n_tok);

    llama_context * ctx = make_ctx(model);

    // the prompt: every row's logits
    decode_rows(ctx, hs, toks, 0, "prompt");

    // greedy, one token per decode
    std::vector<llama_token> gen;
    llama_token cur = argmax(llama_get_logits_ith(ctx, n_tok - 1), n_vocab);
    llama_pos pos = n_tok;
    for (int i = 0; i < n_gen; ++i) {
        gen.push_back(cur);
        llama_batch b = llama_batch_get_one(&cur, 1);
        b.pos = &pos;
        if (llama_decode(ctx, b) != 0) {
            fprintf(stderr, "decode failed at %d\n", i);
            return 1;
        }
        const float * logits = llama_get_logits_ith(ctx, -1);
        printf("step %d tok %d hash %016llx\n", i, cur, (unsigned long long) hs.row(logits));
        cur = argmax(logits, n_vocab);
        pos++;
    }

    // the same generated tokens as batches of T, from a fresh context (the verify shapes)
    for (int T : {2, 4, 8}) {
        llama_free(ctx);
        ctx = make_ctx(model);
        llama_batch b = llama_batch_get_one(toks.data(), n_tok);
        if (llama_decode(ctx, b) != 0) {
            fprintf(stderr, "prompt decode failed\n");
            return 1;
        }
        char tag[16];
        snprintf(tag, sizeof(tag), "T%d", T);
        for (int i = 0; i + T <= n_gen; i += T) {
            std::vector<llama_token> part(gen.begin() + i, gen.begin() + i + T);
            decode_rows(ctx, hs, part, n_tok + i, tag);
        }
    }

    std::string text;
    for (int i = 0; i < n_gen && i < 24; ++i) {
        char buf[64];
        const int n = llama_token_to_piece(vocab, gen[i], buf, sizeof(buf), 0, true);
        if (n > 0) {
            text.append(buf, n);
        }
    }
    printf("rows %d  all %016llx\n", hs.rows, (unsigned long long) hs.all);
    printf("text: %s\n", text.c_str());

    if (dump) {
        fclose(dump);
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
