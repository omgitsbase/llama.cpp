#!/usr/bin/env python3
"""Write a draft vocabulary seed for a GGUF's tokenizer.

A model with an MTP head drafts through the LM head's rows for a ranked subset of its vocabulary
(src/llama-draft-vocab.cpp). Tokens the model has seen rank those rows once enough were counted; before that a
seed ranking decides, and without one the vocabulary's own order does. This writes the seed into
<out_dir>/<fingerprint>.draftvocab, keeping any counts a file of the same vocabulary already holds.

    python scripts/draft-vocab-seed.py model.gguf ranking.npy out_dir
    python scripts/draft-vocab-seed.py model.gguf --fingerprint

ranking.npy holds token ids, most frequent first (int32). Point LLAMA_DRAFT_VOCAB at out_dir, or put the file next
to the model.
"""

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np

if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent.parent / "gguf-py"))
import gguf  # noqa: E402

MAGIC   = 0x4356444c  # "LDVC"
VERSION = 1
HEADER  = struct.Struct("<IIIIQQ")


def token_texts(path: str) -> list[bytes]:
    reader = gguf.GGUFReader(path)
    field = reader.fields["tokenizer.ggml.tokens"]
    # the engine reads a token's text as a C string
    return [bytes(field.parts[i]).split(b"\0")[0] for i in field.data]


def fingerprint(texts: list[bytes]) -> int:
    h = 0xcbf29ce484222325
    data = struct.pack("<I", len(texts)) + b"".join(t + b"\0" for t in texts)
    for b in data:
        h ^= b
        h = (h * 0x100000001b3) & 0xffffffffffffffff
    return h


def read_counts(path: Path, fp: int, n_vocab: int):
    if not path.exists():
        return 0, np.zeros(n_vocab, dtype=np.uint32)
    raw = path.read_bytes()
    magic, version, n, n_seed, fp_file, n_seen = HEADER.unpack_from(raw)
    if magic != MAGIC or version != VERSION or n != n_vocab or fp_file != fp or len(raw) != HEADER.size + 4 * (n + n_seed):
        raise SystemExit(f"{path} belongs to another vocabulary, not touching it")
    return n_seen, np.frombuffer(raw, dtype="<u4", count=n, offset=HEADER.size).copy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("ranking", nargs="?")
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("--fingerprint", action="store_true", help="print the tokenizer's fingerprint and exit")
    args = ap.parse_args()

    texts = token_texts(args.model)
    fp = fingerprint(texts)
    if args.fingerprint or not args.ranking:
        print(f"{fp:016x} n_vocab={len(texts)}")
        return
    if not args.out_dir:
        ap.error("out_dir is required with a ranking")

    n_vocab = len(texts)
    rank = np.load(args.ranking).astype(np.int64).ravel()
    rank = rank[(rank >= 0) & (rank < n_vocab)]
    _, first = np.unique(rank, return_index=True)
    seed = rank[np.sort(first)].astype("<i4")

    out = Path(args.out_dir) / f"{fp:016x}.draftvocab"
    n_seen, counts = read_counts(out, fp, n_vocab)

    tmp = out.with_name(out.name + f".tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(HEADER.pack(MAGIC, VERSION, n_vocab, len(seed), fp, n_seen))
        f.write(counts.astype("<u4").tobytes())
        f.write(seed.tobytes())
    os.replace(tmp, out)
    print(f"{out}: seed of {len(seed)} tokens, {n_seen} counted tokens kept")


if __name__ == "__main__":
    main()
