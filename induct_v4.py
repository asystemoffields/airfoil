#!/usr/bin/env python3
"""
v4: depth generalization, and the memorize-vs-generalize boundary.

Two questions the earlier versions set up:

  A. DEPTH GENERALIZATION. Train only on shallow compositions (depth <= 2). Does
     the learned motif library shorten DEEPER compositions (depth 3,4,5,6) it
     never saw, by a consistent per-motif factor? If the compression ratio is
     ~constant across depth, the abstractions generalize compositionally rather
     than interpolating within the trained depth.

  B. MEMORIZE vs GENERALIZE. BPE on the training set naturally stops at the 4
     generic motifs (every whole-pair adjacency occurs only once -> freq < 2 ->
     no merge). What if we FORCE it past that, memorizing each seen training
     pair as its own symbol? That lowers the TRAIN description further -- but
     does it help held-out compositions whose motif-adjacencies were never
     trained? If train cost drops while novel held-out cost does not, then
     compressing past the generic-motif level is memorization, not learning.
     (I.e. the objective's *own* stopping point is the generalization optimum --
     the same Occam result v2 hinted at, now tied to generalization.)

Pure stdlib.
"""
import itertools
from collections import Counter

BASE = ["inc", "dec", "dbl", "tpl", "sqr", "neg"]
MOT = {1: ("dbl", "inc"), 2: ("inc", "inc", "inc"), 3: ("sqr", "inc"), 4: ("dbl", "dbl")}


def compose(idxs):
    return tuple(op for i in idxs for op in MOT[i])


def seg(seq, vocab):
    n = len(seq)
    f = [0] + [float("inf")] * n
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


def syms(corpus, vocab):
    return sum(len(seg(q, vocab)) for q in corpus)


def main():
    base = [(b,) for b in BASE]

    # TRAIN: the 4 motifs (singles) + all i<=j pairs (the "seen" depth-2 set).
    S_pairs = [(i, j) for i in range(1, 5) for j in range(i, 5)]   # 10 pairs
    TRAIN = [MOT[i] for i in range(1, 5)] + [compose([i, j]) for i, j in S_pairs]
    motif_lib = learn(TRAIN, base)
    learned = [" ".join(s) for s in motif_lib if len(s) > 1]

    print("=" * 78)
    print("v4  DEPTH GENERALIZATION  (train depth <= 2; test deeper, never seen)")
    print("=" * 78)
    print(f"  learned from shallow train: {learned}\n")
    print(f"  {'depth':>6}{'mean base ops':>16}{'mean symbols':>15}{'compression':>14}")
    print("  " + "-" * 49)
    # novel compositions at increasing depth (cycle the motifs)
    cyc = [1, 2, 3, 4, 1, 2, 3, 4]
    for d in (2, 3, 4, 5, 6):
        tests = [compose(cyc[k:k + d]) for k in range(4)]
        b = sum(len(t) for t in tests) / len(tests)
        s = sum(len(seg(t, motif_lib)) for t in tests) / len(tests)
        seen = "  <- trained depth" if d == 2 else ""
        print(f"  {d:>6}{b:>16.1f}{s:>15.1f}{b / s:>13.2f}x{seen}")
    print("\n  -> compression ratio is ~flat across depth: the motifs generalize to")
    print("     compositional depths far beyond the (depth<=2) training. Not interpolation.")

    print("\n" + "=" * 78)
    print("v4  MEMORIZE vs GENERALIZE")
    print("=" * 78)
    # force memorization: motif lib + a macro for every SEEN training pair
    overfit_lib = list(motif_lib) + [compose([i, j]) for i, j in S_pairs]
    # held-out: descending-index compositions -> every adjacency (i>j) was NEVER
    # a trained pair (train pairs are i<=j), so memorized pairs cannot apply.
    NOVEL = [compose(x) for x in ([3, 2, 1], [4, 3, 1], [4, 2, 1], [4, 3, 2], [4, 3, 2, 1])]

    tr_m, tr_o = syms(TRAIN, motif_lib), syms(TRAIN, overfit_lib)
    nv_m, nv_o = syms(NOVEL, motif_lib), syms(NOVEL, overfit_lib)
    print(f"  library                     TRAIN symbols   held-out(novel) symbols")
    print("  " + "-" * 62)
    print(f"  generic motifs ({len(motif_lib)} syms){'':6}{tr_m:>10}{nv_m:>22}")
    print(f"  + memorized {len(S_pairs)} seen pairs{'':1}{tr_o:>10}{nv_o:>22}")
    print(f"\n  memorizing seen pairs cut TRAIN {tr_m}->{tr_o} symbols "
          f"({tr_m / tr_o:.2f}x) ...")
    print(f"  ... but novel held-out stayed {nv_m}->{nv_o} "
          f"({'UNCHANGED' if nv_m == nv_o else f'{nv_m/nv_o:.2f}x'}).")
    print("\n  -> Compressing past the generic motifs is MEMORIZATION: it shrinks the")
    print("     training description but does nothing for novel compositions. BPE's own")
    print("     stopping point (freq<2, = the generic motifs) is the generalization")
    print("     optimum. Shortness that generalizes stops exactly where reuse stops.")


if __name__ == "__main__":
    main()
