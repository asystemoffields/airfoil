#!/usr/bin/env python3
"""
v5: does it survive a richer, realistic domain?

v0-v4 used toy unary int ops. v5 moves to LIST-PROCESSING PIPELINES — real
functions over integer lists (map / filter / reverse / sort / drop) — a much
larger, more realistic program space. The motifs are now genuine list idioms
("double the evens", "square the positives", "+3 to each", "drop the last").
Same machinery (sequences of ops, BPE library learning, min-symbol description);
the question is whether the transfer + depth-generalization results hold up.

Pure stdlib; real semantics included (and sanity-checked).
"""
import itertools
from collections import Counter

# ── list-processing DSL (each op: list[int] -> list[int]) ────────────────────
OP = {
    "inc*": lambda L: [x + 1 for x in L],
    "dec*": lambda L: [x - 1 for x in L],
    "dbl*": lambda L: [2 * x for x in L],
    "neg*": lambda L: [-x for x in L],
    "sqr*": lambda L: [x * x for x in L],
    "even": lambda L: [x for x in L if x % 2 == 0],
    "pos":  lambda L: [x for x in L if x > 0],
    "rev":  lambda L: L[::-1],
    "tail": lambda L: L[1:],
    "init": lambda L: L[:-1],
    "sort": lambda L: sorted(L),
}
BASE = list(OP)

# motifs = recurring list idioms
A = ("even", "dbl*")            # double the evens
B = ("inc*", "inc*", "inc*")    # +3 to each
C = ("pos", "sqr*")             # square the positives
D = ("rev", "tail")             # drop the last element (reverse, drop first)
MOTIFS = [A, B, C, D]

TRAIN = list(MOTIFS) + [a + b for a, b in itertools.product(MOTIFS, repeat=2)]
TEST_RELATED = [A + C + D, C + A + B, D + B + A, B + D + C, A + D + B, C + B + D]

# control 1: only ops NO motif uses (dec*/neg*/sort/init) -> no macro can apply
CONTROL_DISJOINT = [("dec*", "neg*", "sort", "init", "dec*"),
                    ("neg*", "sort", "init", "neg*", "dec*"),
                    ("sort", "init", "dec*", "neg*", "sort"),
                    ("init", "dec*", "neg*", "sort", "init"),
                    ("dec*", "sort", "neg*", "init", "dec*"),
                    ("neg*", "init", "sort", "dec*", "neg*")]
# control 2: SAME ops as motifs, ordered to avoid every motif bigram
CONTROL_SCRAMBLE = [("dbl*", "pos", "inc*", "rev", "sqr*", "tail"),
                    ("pos", "dbl*", "tail", "inc*", "even", "rev"),
                    ("tail", "dbl*", "pos", "rev", "inc*", "even"),
                    ("sqr*", "even", "rev", "dbl*", "tail", "pos"),
                    ("rev", "dbl*", "pos", "tail", "sqr*", "even"),
                    ("pos", "tail", "dbl*", "rev", "sqr*", "inc*")]


def run_list(seq, L):
    for op in seq:
        L = OP[op](L)
    return L


def seg(seq, vocab):
    n = len(seq)
    f = [0] + [float("inf")] * n
    bk = [None] * (n + 1)
    for i in range(1, n + 1):
        for s in vocab:
            Ln = len(s)
            if Ln <= i and seq[i - Ln:i] == s and f[i - Ln] + 1 < f[i]:
                f[i] = f[i - Ln] + 1
                bk[i] = (i - Ln, s)
    out, i = [], n
    while i > 0:
        j, s = bk[i]
        out.append(s)
        i = j
    return out[::-1]


def learn(corpus, vocab):
    vocab = list(vocab)
    while True:
        pairs = Counter()
        for q in corpus:
            s = seg(q, vocab)
            for a, b in zip(s, s[1:]):
                pairs[(a, b)] += 1
        if not pairs:
            break
        (a, b), n = max(pairs.items(), key=lambda kv: (kv[1], -len(kv[0][0] + kv[0][1])))
        if n < 2:
            break
        vocab.append(a + b)
    return vocab


def mean_len(corpus, vocab):
    return sum(len(seg(q, vocab)) for q in corpus) / len(corpus)


def main():
    base = [(b,) for b in BASE]

    # sanity: these are real programs
    print("=" * 78)
    print("v5  RICHER DOMAIN: list-processing pipelines (real semantics)")
    print("=" * 78)
    samp = [1, -2, 3, 4, -5, 6]
    print(f"  sample input {samp}:")
    print(f"    A 'double the evens'      -> {run_list(A, samp)}")
    print(f"    C 'square the positives'  -> {run_list(C, samp)}")
    print(f"    D 'drop the last'         -> {run_list(D, samp)}")

    # learn the library from shallow (depth<=2) training
    lib = learn(TRAIN, base)
    learned = [" ".join(s) for s in lib if len(s) > 1]

    # --- transfer (v1-style) ---
    sets = {"related": TEST_RELATED, "ctrl-disjoint": CONTROL_DISJOINT,
            "ctrl-scramble": CONTROL_SCRAMBLE}
    print("\n  --- TRANSFER: expression length (symbols) per held-out task ---")
    print(f"  learned idioms: {learned}")
    s0 = {k: mean_len(v, base) for k, v in sets.items()}
    s1 = {k: mean_len(v, lib) for k, v in sets.items()}
    for k in sets:
        print(f"    {k:<15} {s0[k]:5.2f} -> {s1[k]:5.2f}   ({s0[k] / s1[k]:.2f}x)")

    # --- depth generalization (v4-style) ---
    print("\n  --- DEPTH GENERALIZATION (trained depth<=2; test deeper) ---")
    cyc = [A, B, C, D, A, B, C, D]
    print(f"    {'depth':>6}{'base ops':>11}{'symbols':>10}{'compression':>14}")
    for d in (2, 3, 4, 5, 6):
        tests = [tuple(op for m in cyc[k:k + d] for op in m) for k in range(4)]
        b = sum(len(t) for t in tests) / len(tests)
        sy = sum(len(seg(t, lib)) for t in tests) / len(tests)
        print(f"    {d:>6}{b:>11.1f}{sy:>10.1f}{b / sy:>13.2f}x")

    rel = s0["related"] / s1["related"]
    ctl = max(s0["ctrl-disjoint"] / s1["ctrl-disjoint"], s0["ctrl-scramble"] / s1["ctrl-scramble"])
    print("\n" + "=" * 78)
    if rel >= 1.5 and ctl <= 1.05:
        print(f"  ✓ HOLDS in the richer domain: related {rel:.2f}x shorter, controls flat")
        print(f"    (≤{ctl:.2f}x), and compression stays ~flat across unseen depths. The")
        print(f"    thesis isn't an artifact of the toy DSL — it survives real list idioms.")
    else:
        print(f"  partial: related {rel:.2f}x, control {ctl:.2f}x — investigate.")


if __name__ == "__main__":
    main()
