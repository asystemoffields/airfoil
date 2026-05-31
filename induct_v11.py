#!/usr/bin/env python3
"""
v11: parameterized fragments earn their keep — reuse a PATTERN, not just exact
structure.

v1-v10 reused *exact* substructure (BPE merges / exact subtrees). That only ever
sees CONTIGUOUS repeats. The more powerful notion of reuse is a SCHEMA: a fixed
frame with a varying slot — e.g. `sqr( ?(x) ) + 1`, the frame (sqr, ?, inc) where
? ranges over operations. Antiunification (least-general generalization) recovers
such a schema by aligning examples and punching a hole where they disagree.

This experiment is built to be the case where exact reuse is BLIND: every program
shares the frame (sqr, ?, inc) but the middle op varies, so NO contiguous pair
ever repeats — BPE learns nothing. Antiunification recovers the schema, and then:
  - DESCRIBES a novel instance in (schema + filler) instead of 3 raw ops;
  - SOLVES a novel instance by filling one hole (~6 candidates) instead of
    searching whole 3-op programs;
  - GENERALIZES to fillers never seen in training.

Pure stdlib.  (Clean best case: one schema, single-op holes, equal lengths.
General antiunification — many schemas needing clustering, multi-op holes — is the
flagged v12 step.)
"""
import itertools

BASE = ["inc", "dec", "dbl", "tpl", "sqr", "neg"]
OP = {"inc": lambda x: x + 1, "dec": lambda x: x - 1, "dbl": lambda x: 2 * x,
      "tpl": lambda x: 3 * x, "sqr": lambda x: x * x, "neg": lambda x: -x}
EX = [-2, -1, 0, 1, 2, 3]
HOLE = "?"

# hidden generative schema: frame (sqr, ?, inc) ; ? varies over operations
FRAME = ("sqr", HOLE, "inc")
TRAIN_FILLERS = ["neg", "dec", "inc"]
TEST_FILLERS = ["dbl", "tpl", "sqr"]            # novel fillers, never trained


def fill(frame, f):
    return tuple(f if t == HOLE else t for t in frame)


TRAIN = [fill(FRAME, f) for f in TRAIN_FILLERS]
TEST = [fill(FRAME, f) for f in TEST_FILLERS]


def run(seq, x):
    for op in seq:
        x = OP[op](x)
    return x


# ---- exact (BPE) abstraction: the v1-v10 mechanism ----
def seg(seq, vocab):
    n = len(seq)
    f = [0] + [10**9] * n
    bk = [None] * (n + 1)
    for i in range(1, n + 1):
        for s in vocab:
            L = len(s)
            if L <= i and seq[i - L:i] == s and f[i - L] + 1 < f[i]:
                f[i] = f[i - L] + 1
                bk[i] = (i - L, s)
    out, i = [], n
    while i > 0:
        j, s = bk[i]
        out.append(s)
        i = j
    return out[::-1]


def bpe_learn(corpus):
    from collections import Counter
    vocab = [(b,) for b in BASE]
    while True:
        pairs = Counter()
        for q in corpus:
            s = seg(q, vocab)
            for a, b in zip(s, s[1:]):
                pairs[(a, b)] += 1
        if not pairs:
            break
        (a, b), n = max(pairs.items(), key=lambda kv: kv[1])
        if n < 2:
            break
        vocab.append(a + b)
    return vocab


# ---- parameterized abstraction: positional antiunification ----
def antiunify(corpus):
    """Least-general generalization of equal-length sequences: keep agreeing
    positions, punch a HOLE where they differ."""
    L = len(corpus[0])
    if any(len(q) != L for q in corpus):
        return None
    return tuple(corpus[0][i] if all(q[i] == corpus[0][i] for q in corpus) else HOLE
                 for i in range(L))


def matches(prog, exs):
    return all(run(prog, i) == o for i, o in exs)


def search_ground(exs):
    """Solve from scratch by enumerating programs (the v3 baseline)."""
    n = 0
    for d in range(1, 5):
        for combo in itertools.product(BASE, repeat=d):
            n += 1
            if matches(combo, exs):
                return n
    return n


def search_schema(exs, schema):
    """Solve a novel instance by filling the schema's hole(s)."""
    holes = [i for i, t in enumerate(schema) if t == HOLE]
    n = 0
    for fillers in itertools.product(BASE, repeat=len(holes)):
        cand = list(schema)
        for h, fop in zip(holes, fillers):
            cand[h] = fop
        n += 1
        if matches(tuple(cand), exs):
            return n
    return n


def main():
    print("=" * 78)
    print("v11  PARAMETERIZED FRAGMENTS — reuse a PATTERN, not just exact structure")
    print("=" * 78)
    print(f"  hidden schema: {FRAME}   (frame sqr(?(x))+1, the middle op varies)")
    print(f"  train fillers {TRAIN_FILLERS}  ->  held-out novel fillers {TEST_FILLERS}\n")

    ground = bpe_learn(TRAIN)
    learned_exact = [s for s in ground if len(s) > 1]
    schema = antiunify(TRAIN)

    print(f"  EXACT (BPE) learned: {learned_exact or 'NOTHING — no contiguous pair repeats'}")
    print(f"  ANTIUNIFY learned schema: {schema}\n")

    print("  held-out novel instances:")
    print(f"  {'program':<22}{'exact desc':>12}{'schema desc':>13}{'exact search':>14}{'schema search':>15}")
    print("  " + "-" * 74)
    eg = es = gd = gs = 0
    for f, prog in zip(TEST_FILLERS, TEST):
        exs = [(i, run(prog, i)) for i in EX]
        d_exact = len(seg(prog, ground))                 # raw symbols (BPE learned nothing)
        d_schema = 1 + sum(1 for t in schema if t == HOLE)  # schema-ref + filler(s)
        n_ground = search_ground(exs)
        n_schema = search_schema(exs, schema)
        gd += d_exact; es += d_schema; gd_ = 0
        eg += n_ground; gs += n_schema
        print(f"  {'->'.join(prog):<22}{d_exact:>12}{d_schema:>13}{n_ground:>14}{n_schema:>15}")

    nt = len(TEST)
    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"  EXACT reuse is BLIND here: BPE found no contiguous repeat, so novel")
    print(f"    instances cost the full {gd//nt} symbols and a full {eg//nt}-node search.")
    print(f"  PARAMETERIZED reuse: the schema describes each in {es//nt} tokens "
          f"({(gd/ max(es,1)):.2f}x shorter)")
    print(f"    and solves it by filling one hole in ~{gs//nt} nodes "
          f"({eg/max(gs,1):.0f}x less search) — for fillers it never saw.")
    print()
    print("  Parameterized fragments earn their keep exactly where exact ones can't:")
    print("  when the shared structure is a TEMPLATE with a varying slot. This is the")
    print("  step from 'reuse what recurs verbatim' to 'reuse the pattern' — reusing a")
    print("  schema and binding its hole is what lets a few examples cover an open set.")


if __name__ == "__main__":
    main()
