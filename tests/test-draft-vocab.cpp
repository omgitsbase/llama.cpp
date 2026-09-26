#include "../src/llama-draft-vocab.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); exit(1); } } while (0)

static std::vector<llama_draft_vocab_token> make_vocab(int32_t n_vocab) {
    std::vector<llama_draft_vocab_token> v(n_vocab);
    for (int32_t i = 0; i < n_vocab; ++i) {
        v[i].text = "t" + std::to_string(i);
        v[i].attr = LLAMA_TOKEN_ATTR_NORMAL;
    }
    return v;
}

static bool has(const std::vector<int32_t> & ids, int32_t id) {
    return std::binary_search(ids.begin(), ids.end(), id);
}

static void check_shape(const std::vector<int32_t> & ids, int64_t n, int32_t n_vocab) {
    CHECK((int64_t) ids.size() == n);
    CHECK(std::is_sorted(ids.begin(), ids.end()));
    CHECK(std::adjacent_find(ids.begin(), ids.end()) == ids.end());
    CHECK(ids.front() >= 0 && ids.back() < n_vocab);
}

static void test_forced() {
    const int32_t n_vocab = 4096;
    const int64_t n       = 256;
    auto v = make_vocab(n_vocab);

    // high ids that id order alone would never pick
    v[4000].eog  = true;
    v[4001].attr = LLAMA_TOKEN_ATTR_BYTE;
    v[4002].attr = LLAMA_TOKEN_ATTR_CONTROL;
    v[4003].attr = LLAMA_TOKEN_ATTR_USER_DEFINED;
    v[4004].attr = LLAMA_TOKEN_ATTR_UNUSED | LLAMA_TOKEN_ATTR_CONTROL;
    v[4005].attr = LLAMA_TOKEN_ATTR_UNKNOWN;
    // low ids that id order would pick, but that are never produced
    v[3].attr = LLAMA_TOKEN_ATTR_UNUSED;
    v[4].attr = LLAMA_TOKEN_ATTR_UNKNOWN | LLAMA_TOKEN_ATTR_CONTROL;

    const char * source = nullptr;
    const auto ids = llama_draft_vocab_rank(v, n, nullptr, &source);
    check_shape(ids, n, n_vocab);
    CHECK(strcmp(source, "token ids") == 0);
    CHECK(has(ids, 4000) && has(ids, 4001) && has(ids, 4002) && has(ids, 4003));
    CHECK(!has(ids, 4004) && !has(ids, 4005) && !has(ids, 3) && !has(ids, 4));
    CHECK(has(ids, 0) && has(ids, 5));

    // the forced set is capped at n/8, end of generation and bytes first
    for (int32_t i = 2000; i < 2100; ++i) {
        v[i].attr = LLAMA_TOKEN_ATTR_CONTROL;
    }
    const auto capped = llama_draft_vocab_rank(v, n, nullptr, nullptr);
    check_shape(capped, n, n_vocab);
    int32_t n_forced = 0;
    for (const int32_t id : capped) {
        n_forced += id >= 2000 && id < 2100;
    }
    CHECK(n_forced <= n/8);
    CHECK(has(capped, 4000) && has(capped, 4001));
}

static void test_prior() {
    const int32_t n_vocab = 1000;
    auto v = make_vocab(n_vocab);

    const auto by_id = llama_draft_vocab_rank(v, 100, nullptr, nullptr);
    for (int32_t i = 0; i < 100; ++i) {
        CHECK(by_id[i] == i);
    }

    // SPM-style scores: the highest scores win whatever their ids
    for (int32_t i = 0; i < n_vocab; ++i) {
        v[i].score = (float) i;
    }
    const char * source = nullptr;
    const auto by_score = llama_draft_vocab_rank(v, 100, nullptr, &source);
    CHECK(strcmp(source, "token scores") == 0);
    for (int32_t i = 0; i < 100; ++i) {
        CHECK(by_score[i] == n_vocab - 100 + i);
    }
}

static void test_counts_and_seed() {
    const int32_t n_vocab = 1000;
    const auto v = make_vocab(n_vocab);

    llama_draft_vocab_counts c;
    c.counts.assign(n_vocab, 0);
    for (int32_t i = 900; i < 950; ++i) {
        c.counts[i] = 10;
    }
    c.n_seen = LLAMA_DRAFT_VOCAB_TRUSTED - 1;

    const char * source = nullptr;
    auto ids = llama_draft_vocab_rank(v, 100, &c, &source);
    CHECK(strcmp(source, "token ids") == 0);
    CHECK(!has(ids, 900));

    c.n_seen = LLAMA_DRAFT_VOCAB_TRUSTED;
    ids = llama_draft_vocab_rank(v, 100, &c, &source);
    CHECK(strcmp(source, "counts") == 0);
    check_shape(ids, 100, n_vocab);
    for (int32_t i = 900; i < 950; ++i) {
        CHECK(has(ids, i));
    }
    // the seed breaks the ties among the tokens never counted
    for (int32_t i = 999; i >= 500; --i) {
        c.seed.push_back(i);
    }
    ids = llama_draft_vocab_rank(v, 100, &c, &source);
    check_shape(ids, 100, n_vocab);
    for (int32_t i = 900; i < 1000; ++i) {
        CHECK(has(ids, i));
    }

    // untrusted counts leave the seed in charge
    c.n_seen = 0;
    ids = llama_draft_vocab_rank(v, 100, &c, &source);
    CHECK(strcmp(source, "seed") == 0);
    for (int32_t i = 900; i < 1000; ++i) {
        CHECK(has(ids, i));
    }
}

static void test_file(const fs::path & dir) {
    const auto v  = make_vocab(300);
    const auto fp = llama_draft_vocab_fingerprint(v);
    CHECK(fp == llama_draft_vocab_fingerprint(make_vocab(300)));
    CHECK(fp != llama_draft_vocab_fingerprint(make_vocab(301)));

    const std::string path = (dir / "test.draftvocab").u8string();
    fs::remove(path);

    llama_draft_vocab_counts c;
    CHECK(!llama_draft_vocab_read(path, fp, 300, c));

    std::vector<uint32_t> delta(300, 0);
    delta[7] = 5;
    CHECK(llama_draft_vocab_add(path, fp, delta, 5));
    CHECK(llama_draft_vocab_add(path, fp, delta, 5)); // renames over the file it read
    CHECK(llama_draft_vocab_read(path, fp, 300, c));
    CHECK(c.n_seen == 10 && c.counts[7] == 10 && c.seed.empty());

    // another vocabulary's file is neither read nor overwritten
    CHECK(!llama_draft_vocab_read(path, fp + 1, 300, c));
    CHECK(!llama_draft_vocab_read(path, fp, 301, c));
    CHECK(!llama_draft_vocab_add(path, fp + 1, delta, 5));
    CHECK(llama_draft_vocab_read(path, fp, 300, c) && c.n_seen == 10);

    // past 2^26 tokens every count halves
    delta[7] = 1u << 26;
    CHECK(llama_draft_vocab_add(path, fp, delta, 1ull << 26));
    CHECK(llama_draft_vocab_read(path, fp, 300, c));
    CHECK(c.n_seen == ((1ull << 26) + 10) / 2 && c.counts[7] == ((1u << 26) + 10) / 2);

    // a truncated file or a bad magic is ignored and left alone
    {
        FILE * f = fopen(path.c_str(), "r+b");
        CHECK(f);
        const char bad[4] = { 'X', 'X', 'X', 'X' };
        fwrite(bad, 1, 4, f);
        fclose(f);
    }
    CHECK(!llama_draft_vocab_read(path, fp, 300, c));
    CHECK(!llama_draft_vocab_add(path, fp, delta, 1));
    fs::remove(path);

    // a directory that does not exist: the save is skipped, nothing throws
    CHECK(!llama_draft_vocab_add((dir / "missing" / "x.draftvocab").u8string(), fp, delta, 1));

    // a non-ASCII directory
    const fs::path dir_u8 = dir / fs::u8path("\xd1\x81\xd1\x87\xd1\x91\xd1\x82");
    fs::create_directories(dir_u8);
    const std::string path_u8 = (dir_u8 / "x.draftvocab").u8string();
    CHECK(llama_draft_vocab_add(path_u8, fp, delta, 3));
    CHECK(llama_draft_vocab_read(path_u8, fp, 300, c) && c.n_seen == 3);
    fs::remove_all(dir_u8);

    // the counter adds what it counted when it goes away
    {
        llama_draft_vocab_counter counter(path, fp, 300);
        const llama_token toks[] = { 1, 2, 2, -1, 300, LLAMA_TOKEN_NULL };
        counter.note(toks, 6);
    }
    CHECK(llama_draft_vocab_read(path, fp, 300, c));
    CHECK(c.n_seen == 3 && c.counts[1] == 1 && c.counts[2] == 2);
    fs::remove(path);
}

static void set_env_dir(const char * value) {
#if defined(_WIN32)
    _wputenv_s(L"LLAMA_DRAFT_VOCAB", value ? fs::u8path(value).wstring().c_str() : L"");
#else
    if (value) {
        setenv("LLAMA_DRAFT_VOCAB", value, 1);
    } else {
        unsetenv("LLAMA_DRAFT_VOCAB");
    }
#endif
}

static void test_env_path(const fs::path & dir) {
    const std::string model = (dir / "m.gguf").u8string();

    set_env_dir(nullptr);
    CHECK(llama_draft_vocab_off() && llama_draft_vocab_path(model, 1).empty());
    set_env_dir("off");
    CHECK(llama_draft_vocab_off() && llama_draft_vocab_path(model, 1).empty());

    set_env_dir("on");
    CHECK(!llama_draft_vocab_off());
    CHECK(llama_draft_vocab_path(model, 1) == (dir / "0000000000000001.draftvocab").u8string());
    CHECK(llama_draft_vocab_path("", 1).empty());

    // a non-ASCII directory, which is not in the ANSI code page on Windows
    const fs::path dir_u8 = dir / fs::u8path("\xd1\x81\xd1\x87\xd1\x91\xd1\x82");
    set_env_dir(dir_u8.u8string().c_str());
    CHECK(llama_draft_vocab_path(model, 1) == (dir_u8 / "0000000000000001.draftvocab").u8string());

    set_env_dir(nullptr);
}

int main(int argc, char ** argv) {
    const fs::path dir = argc > 1 ? fs::u8path(argv[1]) : fs::temp_directory_path();

    test_forced();
    test_prior();
    test_counts_and_seed();
    test_file(dir);
    test_env_path(dir);

    printf("test-draft-vocab: OK\n");
    return 0;
}
