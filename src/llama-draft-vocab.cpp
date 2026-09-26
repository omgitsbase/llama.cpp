#include "llama-draft-vocab.h"

#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-vocab.h"

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <system_error>

#if defined(_WIN32)
#include <process.h>
#define llama_draft_vocab_getpid _getpid
#else
#include <unistd.h>
#define llama_draft_vocab_getpid getpid
#endif

namespace fs = std::filesystem;

static constexpr uint32_t LLAMA_DRAFT_VOCAB_MAGIC   = 0x4356444c; // "LDVC"
static constexpr uint32_t LLAMA_DRAFT_VOCAB_VERSION = 1;
static constexpr uint64_t LLAMA_DRAFT_VOCAB_HALVE   = 1ull << 26;
static constexpr uint64_t LLAMA_DRAFT_VOCAB_SAVE    = 65536;

std::vector<llama_draft_vocab_token> llama_draft_vocab_tokens(const llama_vocab & vocab) {
    const uint32_t n_vocab = vocab.n_tokens();
    std::vector<llama_draft_vocab_token> res(n_vocab);
    for (uint32_t id = 0; id < n_vocab; ++id) {
        auto & t = res[id];
        t.text  = vocab.token_get_text(id);
        t.attr  = vocab.token_get_attr(id);
        t.score = vocab.token_get_score(id);
        t.eog   = vocab.is_eog(id);
    }
    return res;
}

uint64_t llama_draft_vocab_fingerprint(const std::vector<llama_draft_vocab_token> & tokens) {
    uint64_t h = 0xcbf29ce484222325ull;
    auto add = [&](const uint8_t * p, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            h ^= p[i];
            h *= 0x100000001b3ull;
        }
    };
    const uint32_t n_vocab = (uint32_t) tokens.size();
    const uint8_t  n_le[4] = { uint8_t(n_vocab), uint8_t(n_vocab >> 8), uint8_t(n_vocab >> 16), uint8_t(n_vocab >> 24) };
    add(n_le, sizeof(n_le));
    const uint8_t zero = 0;
    for (const auto & t : tokens) {
        add((const uint8_t *) t.text.data(), t.text.size());
        add(&zero, 1);
    }
    return h;
}

bool llama_draft_vocab_off() {
    const char * env = getenv("LLAMA_DRAFT_VOCAB");
    return env == nullptr || env[0] == '\0' || strcmp(env, "off") == 0 || strcmp(env, "0") == 0;
}

// getenv returns ANSI code page bytes under MSVC, which u8path rejects
static fs::path llama_draft_vocab_env_dir() {
#if defined(_WIN32)
    const wchar_t * env = _wgetenv(L"LLAMA_DRAFT_VOCAB");
    return env ? fs::path(env) : fs::path();
#else
    const char * env = getenv("LLAMA_DRAFT_VOCAB");
    return env ? fs::u8path(env) : fs::path();
#endif
}

std::string llama_draft_vocab_path(const std::string & path_model, uint64_t fp) {
    if (llama_draft_vocab_off()) {
        return {};
    }
    char name[64];
    snprintf(name, sizeof(name), "%016" PRIx64 ".draftvocab", fp);

    try {
        fs::path dir = llama_draft_vocab_env_dir();
        if (dir == fs::path("on") || dir == fs::path("1")) {
            if (path_model.empty()) {
                return {};
            }
            dir = fs::u8path(path_model).parent_path();
        }
        const auto p = (dir / fs::u8path(name)).u8string();
        return std::string(p.begin(), p.end());
    } catch (const std::exception & e) {
        LLAMA_LOG_WARN("%s: no draft vocabulary counts file: %s\n", __func__, e.what());
        return {};
    }
}

struct llama_draft_vocab_header {
    uint32_t magic;
    uint32_t version;
    uint32_t n_vocab;
    uint32_t n_seed;
    uint64_t fp;
    uint64_t n_seen;
};
static_assert(sizeof(llama_draft_vocab_header) == 32, "packed header");

enum llama_draft_vocab_status {
    LLAMA_DRAFT_VOCAB_OK,
    LLAMA_DRAFT_VOCAB_MISSING,
    LLAMA_DRAFT_VOCAB_FOREIGN,
};

static llama_draft_vocab_status llama_draft_vocab_read_impl(const std::string & path, uint64_t fp, uint32_t n_vocab, llama_draft_vocab_counts & out) {
    try {
        std::error_code ec;
        if (!fs::exists(fs::u8path(path), ec)) {
            return LLAMA_DRAFT_VOCAB_MISSING;
        }
        llama_file f(path.c_str(), "rb");
        llama_draft_vocab_header hdr;
        if (f.size() < sizeof(hdr)) {
            return LLAMA_DRAFT_VOCAB_FOREIGN;
        }
        f.read_raw(&hdr, sizeof(hdr));
        const size_t size_exp = sizeof(hdr) + (size_t) hdr.n_vocab*sizeof(uint32_t) + (size_t) hdr.n_seed*sizeof(int32_t);
        if (hdr.magic != LLAMA_DRAFT_VOCAB_MAGIC || hdr.version != LLAMA_DRAFT_VOCAB_VERSION || hdr.n_vocab != n_vocab ||
                hdr.fp != fp || hdr.n_seed > n_vocab || f.size() != size_exp) {
            return LLAMA_DRAFT_VOCAB_FOREIGN;
        }
        out.n_seen = hdr.n_seen;
        out.counts.resize(n_vocab);
        out.seed.resize(hdr.n_seed);
        f.read_raw(out.counts.data(), out.counts.size()*sizeof(uint32_t));
        if (!out.seed.empty()) {
            f.read_raw(out.seed.data(), out.seed.size()*sizeof(int32_t));
        }
    } catch (const std::exception & e) {
        LLAMA_LOG_WARN("%s: %s: %s\n", __func__, path.c_str(), e.what());
        return LLAMA_DRAFT_VOCAB_FOREIGN;
    }
    return LLAMA_DRAFT_VOCAB_OK;
}

bool llama_draft_vocab_read(const std::string & path, uint64_t fp, uint32_t n_vocab, llama_draft_vocab_counts & out) {
    return llama_draft_vocab_read_impl(path, fp, n_vocab, out) == LLAMA_DRAFT_VOCAB_OK;
}

bool llama_draft_vocab_add(const std::string & path, uint64_t fp, const std::vector<uint32_t> & counts, uint64_t n_seen) {
    const uint32_t n_vocab = (uint32_t) counts.size();

    llama_draft_vocab_counts cur;
    switch (llama_draft_vocab_read_impl(path, fp, n_vocab, cur)) {
        case LLAMA_DRAFT_VOCAB_OK:      break;
        case LLAMA_DRAFT_VOCAB_MISSING: cur.counts.assign(n_vocab, 0); break;
        case LLAMA_DRAFT_VOCAB_FOREIGN: return false;
    }

    cur.n_seen += n_seen;
    for (uint32_t i = 0; i < n_vocab; ++i) {
        cur.counts[i] = (uint32_t) std::min<uint64_t>((uint64_t) cur.counts[i] + counts[i], UINT32_MAX);
    }
    while (cur.n_seen > LLAMA_DRAFT_VOCAB_HALVE) {
        cur.n_seen /= 2;
        for (auto & c : cur.counts) {
            c /= 2;
        }
    }

    const std::string tmp = path + ".tmp." + std::to_string((long long) llama_draft_vocab_getpid());
    try {
        llama_file f(tmp.c_str(), "wb");
        const llama_draft_vocab_header hdr = {
            LLAMA_DRAFT_VOCAB_MAGIC, LLAMA_DRAFT_VOCAB_VERSION, n_vocab, (uint32_t) cur.seed.size(), fp, cur.n_seen,
        };
        f.write_raw(&hdr, sizeof(hdr));
        f.write_raw(cur.counts.data(), cur.counts.size()*sizeof(uint32_t));
        if (!cur.seed.empty()) {
            f.write_raw(cur.seed.data(), cur.seed.size()*sizeof(int32_t));
        }
    } catch (const std::exception & e) {
        LLAMA_LOG_WARN("%s: cannot write %s: %s\n", __func__, tmp.c_str(), e.what());
        std::error_code ec;
        fs::remove(fs::u8path(tmp), ec);
        return false;
    }

    try {
        std::error_code ec;
        fs::rename(fs::u8path(tmp), fs::u8path(path), ec);
        if (ec) {
            LLAMA_LOG_WARN("%s: cannot replace %s: %s\n", __func__, path.c_str(), ec.message().c_str());
            fs::remove(fs::u8path(tmp), ec);
            return false;
        }
    } catch (const std::exception & e) {
        LLAMA_LOG_WARN("%s: cannot replace %s: %s\n", __func__, path.c_str(), e.what());
        return false;
    }
    return true;
}

std::vector<int32_t> llama_draft_vocab_rank(
        const std::vector<llama_draft_vocab_token> & tokens,
                                           int64_t   n,
                   const llama_draft_vocab_counts * counts,
                                     const char ** source) {
    const int32_t n_vocab = (int32_t) tokens.size();

    std::vector<int32_t> res;
    if (n >= n_vocab) {
        res.resize(n_vocab);
        for (int32_t i = 0; i < n_vocab; ++i) {
            res[i] = i;
        }
        if (source) {
            *source = "all";
        }
        return res;
    }

    const bool trusted = counts && counts->n_seen >= LLAMA_DRAFT_VOCAB_TRUSTED && (int32_t) counts->counts.size() == n_vocab;
    const bool seeded  = counts && !counts->seed.empty();

    std::vector<int32_t> seed_rank(n_vocab, INT32_MAX);
    if (seeded) {
        for (size_t i = 0; i < counts->seed.size(); ++i) {
            const int32_t id = counts->seed[i];
            if (id >= 0 && id < n_vocab && seed_rank[id] == INT32_MAX) {
                seed_rank[id] = (int32_t) i;
            }
        }
    }

    // the vocabulary's own order: score order when the scores say something, id order otherwise (BPE merge order)
    std::vector<int32_t> prior(n_vocab);
    for (int32_t i = 0; i < n_vocab; ++i) {
        prior[i] = i;
    }
    bool scores_vary = false;
    for (int32_t i = 1; i < n_vocab && !scores_vary; ++i) {
        scores_vary = tokens[i].score != tokens[0].score;
    }
    if (scores_vary) {
        std::vector<int32_t> order(prior);
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return tokens[a].score > tokens[b].score; });
        for (int32_t i = 0; i < n_vocab; ++i) {
            prior[order[i]] = i;
        }
    }

    auto unused = [&](int32_t id) {
        return (tokens[id].attr & (LLAMA_TOKEN_ATTR_UNUSED | LLAMA_TOKEN_ATTR_UNKNOWN)) != 0;
    };
    auto before = [&](int32_t a, int32_t b) {
        const bool ua = unused(a);
        const bool ub = unused(b);
        if (ua != ub) {
            return ub;
        }
        if (trusted && counts->counts[a] != counts->counts[b]) {
            return counts->counts[a] > counts->counts[b];
        }
        if (seed_rank[a] != seed_rank[b]) {
            return seed_rank[a] < seed_rank[b];
        }
        return prior[a] < prior[b];
    };

    // forced rows are capped at n/8
    std::vector<int32_t> forced_a;
    std::vector<int32_t> forced_b;
    for (int32_t id = 0; id < n_vocab; ++id) {
        const uint32_t attr = tokens[id].attr;
        if (tokens[id].eog || ((attr & LLAMA_TOKEN_ATTR_BYTE) && !unused(id))) {
            forced_a.push_back(id);
        } else if ((attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED)) && !unused(id)) {
            forced_b.push_back(id);
        }
    }
    std::sort(forced_a.begin(), forced_a.end(), before);
    std::sort(forced_b.begin(), forced_b.end(), before);

    std::vector<uint8_t> taken(n_vocab, 0);
    const size_t n_forced_max = (size_t) (n / 8);
    for (const auto * list : { &forced_a, &forced_b }) {
        for (const int32_t id : *list) {
            if (res.size() >= n_forced_max) {
                break;
            }
            res.push_back(id);
            taken[id] = 1;
        }
    }

    std::vector<int32_t> rest;
    rest.reserve(n_vocab - res.size());
    for (int32_t id = 0; id < n_vocab; ++id) {
        if (!taken[id]) {
            rest.push_back(id);
        }
    }
    const size_t n_rest = (size_t) n - res.size();
    std::nth_element(rest.begin(), rest.begin() + n_rest, rest.end(), before);
    res.insert(res.end(), rest.begin(), rest.begin() + n_rest);
    std::sort(res.begin(), res.end());

    if (source) {
        *source = trusted ? "counts" : seeded ? "seed" : scores_vary ? "token scores" : "token ids";
    }
    return res;
}

llama_draft_vocab_counter::llama_draft_vocab_counter(std::string path, uint64_t fp, uint32_t n_vocab)
    : path(std::move(path)), fp(fp), counts(n_vocab, 0) {}

llama_draft_vocab_counter::~llama_draft_vocab_counter() {
    flush(true);
}

void llama_draft_vocab_counter::note(const llama_token * tokens, int32_t n) {
    const int32_t n_vocab = (int32_t) counts.size();
    for (int32_t i = 0; i < n; ++i) {
        if (tokens[i] >= 0 && tokens[i] < n_vocab) {
            counts[tokens[i]]++;
            n_counted++;
        }
    }
    if (n_counted >= LLAMA_DRAFT_VOCAB_SAVE) {
        flush(false);
    }
}

void llama_draft_vocab_counter::flush(bool wait) {
    if (saving.load() && !wait) {
        return;
    }
    if (saver.joinable()) {
        saver.join();
    }
    if (n_counted == 0) {
        return;
    }

    std::vector<uint32_t> delta(counts.size(), 0);
    delta.swap(counts);
    const uint64_t n_delta = n_counted;
    n_counted = 0;

    auto save = [this](std::vector<uint32_t> delta, uint64_t n_delta) {
        if (!llama_draft_vocab_add(path, fp, delta, n_delta)) {
            static std::atomic<bool> warned { false };
            if (!warned.exchange(true)) {
                LLAMA_LOG_WARN("%s: draft vocab counts are not saved to %s\n", __func__, path.c_str());
            }
        }
        saving.store(false);
    };
    saving.store(true);
    if (wait) {
        save(std::move(delta), n_delta);
    } else {
        try {
            saver = std::thread(save, std::move(delta), n_delta);
        } catch (const std::exception &) {
            saving.store(false);
        }
    }
}
