#pragma once

#include "llama.h"

#include <string>
#include <vector>

struct llama_vocab;
struct llama_grammar;

// sampler chain

struct llama_sampler_chain {
    llama_sampler_chain_params params;

    // has .backend_init() been called?
    bool is_init = false;

    uint32_t n_nodes = 0;

    struct info {
        bool is_backend;

        llama_sampler * ptr;
    };

    std::vector<info> samplers;

    // pre-allocated buffer for llama_sampler_sample to avoid repeated allocations
    std::vector<llama_token_data> cur;

    // timing

    mutable int64_t t_sample_us;

    mutable int32_t n_sample;
};

uint32_t llama_sampler_backend_n_nodes(const llama_sampler * sampler);
std::string llama_sampler_backend_signature(const llama_sampler * sampler);
// chained: the decode was queued before the host accepted the last decode's token
void llama_sampler_backend_begin(llama_sampler * sampler, bool chained = false);
// the sampler's backend part is one greedy argmax over the raw logits
bool llama_sampler_is_backend_greedy(const llama_sampler * sampler);

// the backend part reads no token history from the host, so a decode can be queued before the last token is read
bool llama_sampler_backend_chainable(const llama_sampler * sampler);

// the backend part picks the token, so it is on the device after a decode; false when the CPU picks from candidates
bool llama_sampler_backend_selects(const llama_sampler * sampler);

// The batched form of a backend sampler chain, vLLM's sampler: a graph's rows go through one set of ops (a top-k
// over all rows, then the candidates' penalties, top-p, min-p, temperature and the draw) instead of a chain each.
// The chain's samplers, in its order: logit bias (no positive bias), penalties (only lowering logits), top-k, top-p,
// min-p, temperature (no dynamic range), dist; any other sampler only at its defaults
struct llama_sampler_batch_params {
    int32_t top_k           = 0;
    float   top_p           = 1.0f;
    float   min_p           = 0.0f;
    float   temp            = 1.0f;
    int32_t penalty_last_n  = 0; // 0: no penalties
    float   penalty_repeat  = 1.0f;
    float   penalty_freq    = 0.0f;
    float   penalty_present = 0.0f;
    std::vector<llama_logit_bias> bias;
};
bool llama_sampler_batch_params_get(const llama_sampler * chain, llama_sampler_batch_params & out);
// the next uniform of the chain's draw, as its own backend input takes it (the transactional draws of several rows)
float llama_sampler_batch_uniform(llama_sampler * chain);
// the seed of the chain's draw (0 without one): speculative sampling's other uniforms hash it with the position
uint32_t llama_sampler_batch_seed(const llama_sampler * chain);
// the chain's penalty window, oldest first (empty without penalties)
void llama_sampler_batch_window(const llama_sampler * chain, std::vector<llama_token> & out);

struct llama_sampler * llama_sampler_init_dry_testing(
        float   dry_multiplier,
        float   dry_base,
        int32_t dry_allowed_length,
        int32_t dry_penalty_last_n,
        const std::vector<std::vector<llama_token>> & seq_breakers);
