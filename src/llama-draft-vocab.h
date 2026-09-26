#pragma once

#include "llama.h"

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

struct llama_vocab;
struct ggml_tensor;

#define LLAMA_DRAFT_VOCAB_ROWS     65536
#define LLAMA_DRAFT_VOCAB_TRUSTED  16384 // counts rank the rows once this many tokens were seen

struct llama_draft_head {
    ggml_tensor * src = nullptr; // the LM head the rows come from
    ggml_tensor * w   = nullptr; // [n_embd, n] its rows for the ranked tokens
    ggml_tensor * ids = nullptr; // I32 [n] their token ids, ascending

    std::vector<uint8_t> mask;   // [n_vocab] 1 when the token has a row
    float saved = 0.0f;          // share of the head's rows it leaves out
    bool  whole = false;         // a draft model's: every graph of its contexts multiplies by it

    std::string path;            // the counts file contexts add to, empty for none
    uint64_t    fp = 0;          // the vocabulary's fingerprint
};

struct llama_draft_vocab_token {
    std::string text;
    uint32_t    attr  = 0; // llama_token_attr
    float       score = 0.0f;
    bool        eog   = false;
};

struct llama_draft_vocab_counts {
    uint64_t              n_seen = 0;
    std::vector<uint32_t> counts; // [n_vocab]
    std::vector<int32_t>  seed;   // token ids, most likely first
};

std::vector<llama_draft_vocab_token> llama_draft_vocab_tokens(const llama_vocab & vocab);

// FNV-1a-64 over n_vocab and the token texts: attributes are rewritten at load, the texts are not
uint64_t llama_draft_vocab_fingerprint(const std::vector<llama_draft_vocab_token> & tokens);

// LLAMA_DRAFT_VOCAB=off or 0: no reduced head; "on" keeps its counts next to the model, a directory keeps them there
bool llama_draft_vocab_off();

// the counts file for a vocabulary, empty when there is none; never throws
std::string llama_draft_vocab_path(const std::string & path_model, uint64_t fp);

// false when the file is missing or was written for another vocabulary
bool llama_draft_vocab_read(const std::string & path, uint64_t fp, uint32_t n_vocab, llama_draft_vocab_counts & out);

// adds counts to the file, halving them all past 2^26 tokens; a file of another vocabulary is left alone
bool llama_draft_vocab_add(const std::string & path, uint64_t fp, const std::vector<uint32_t> & counts, uint64_t n_seen);

// the n token ids a reduced head keeps, ascending; source names what ranked them
std::vector<int32_t> llama_draft_vocab_rank(
        const std::vector<llama_draft_vocab_token> & tokens,
                                           int64_t   n,
                   const llama_draft_vocab_counts * counts,
                                     const char ** source);

// counts what a context produces and adds it to the file every 65,536 tokens on
// a thread of its own, and once more when it is destroyed
class llama_draft_vocab_counter {
public:
    llama_draft_vocab_counter(std::string path, uint64_t fp, uint32_t n_vocab);
    ~llama_draft_vocab_counter();

    void note(const llama_token * tokens, int32_t n);

private:
    void flush(bool wait);

    const std::string path;
    const uint64_t    fp;

    std::vector<uint32_t> counts;
    uint64_t              n_counted = 0;

    std::thread       saver;
    std::atomic<bool> saving { false };
};
