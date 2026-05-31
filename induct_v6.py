#!/usr/bin/env python3
"""
v6: the domesticated learner.

The design thesis: making compression the objective doesn't banish the learner —
it *domesticates* it into a guide for the search over short programs. v3 showed
the library cuts search by letting enumeration take big steps (uniform order).
v6 adds a learned PROPOSER — a bigram model over the library symbols, fit on the
same training solutions — that ORDERS the search by what's likely to come next.
It changes nothing about the objective (still finding a program that fits); it
just tries plausible compositions first.

We compare, on held-out tasks, the nodes (programs checked) to find a solution:
    uniform   : iterative-deepening over the library vocab (v3's method).
    guided    : best-first over the proposer's program distribution.

Expectation: on RELATED tasks the proposer has learned the motif-to-motif
transitions, so it should find solutions in far fewer nodes than uniform. On
CONTROL tasks it has no useful prior, so guidance shouldn't help — the learner
is powerful exactly where there's structure to have learned, and inert elsewhere.

Pure stdlib.
"""
import heapq
import itertools
import math
from collections import Counter

BASE = ["inc", "dec", "dbl", "tpl", "sqr", "neg"]
OP = {"inc": lambda x: x + 1, "dec": lambda x: x - 1, "dbl": lambda x: 2 * x,
      "tpl": lambda x: 3 * x, "sqr": lambda x: x * x, "neg": lambda x: -x}
INPUTS = [-1, 0, 1, 2, 3, 4]
CAP = 400_000
MAXLEN = 10  # max base-op length of a candidate program

M1, M2, M3, M4 = ("dbl", "inc"), ("inc", "inc", "inc"), ("sqr", "inc"), ("dbl", "dbl")
MOTIFS = [M1, M2, M3, M4]
TRAIN = list(MOTIFS) + [a + b for a, b in itertools.product(MOTIFS, repeat=2)]
TEST_RELATED = [M1 + M3 + M4, M3 + M1 + M2, M4 + M2 + M1,
                M2 + M4 + M3, M1 + M4 + M2, M3 + M2 + M4]
CONTROL = [("tpl", "neg", "dec", "tpl", "neg"), ("neg", "dec", "tpl", "neg", "dec"),
           ("dec", "tpl", "neg", "dec", "tpl"), ("tpl", "dec", "neg", "tpl", "dec"),
           ("neg", "tpl", "dec", "neg", "tpl"), ("dec", "neg", "tpl", "dec", "neg")]


def run(seq, x):
    for op in seq:
        x = OP[op](x)
    return x


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


def matches(prog, exs):
    for i, o in exs:
        if run(prog, i) != o:
            return False
    return True


def uniform_solve(exs, vocab):
    """v3-style iterative deepening; count programs checked."""
    nodes = 0
    for d in range(1, MAXLEN + 1):
        for combo in itertools.product(vocab, repeat=d):
            prog = tuple(op for s in combo for op in s)
            if len(prog) > MAXLEN:
                continue
            nodes += 1
            if nodes > CAP:
                return False, nodes
            if matches(prog, exs):
                return True, nodes
    return False, nodes


def fit_bigram(vocab):
    """P(next | prev) over library symbols, from train segmentations. prev=None=START."""
    trans = Counter()
    ctx = Counter()
    for q in TRAIN:
        prev = None
        for s in seg(q, vocab):
            trans[(prev, s)] += 1
            ctx[prev] += 1
            prev = s
    V = len(vocab)

    def logp(prev, s):  # add-1 smoothed
        return math.log((trans[(prev, s)] + 1) / (ctx[prev] + V))
    return logp


def guided_solve(exs, vocab, logp):
    """Best-first over the proposer's distribution; count programs checked."""
    nodes = 0
    ctr = 0
    # frontier items: (neg_logprob, tiebreak, last_symbol, expansion_tuple)
    heap = []
    for s in vocab:
        ctr += 1
        heapq.heappush(heap, (-logp(None, s), ctr, s, s))
    while heap:
        nlp, _, last, prog = heapq.heappop(heap)
        nodes += 1
        if nodes > CAP:
            return False, nodes
        if matches(prog, exs):
            return True, nodes
        if len(prog) >= MAXLEN:
            continue
        for s in vocab:
            if len(prog) + len(s) > MAXLEN:
                continue
            ctr += 1
            heapq.heappush(heap, (nlp - logp(last, s), ctr, s, prog + s))
    return False, nodes


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def main():
    base = [(b,) for b in BASE]
    lib = learn(TRAIN, base)
    logp = fit_bigram(lib)
    print("=" * 80)
    print("v6  THE DOMESTICATED LEARNER — proposer-guided vs uniform search")
    print("=" * 80)
    print(f"  library: {[' '.join(s) for s in lib if len(s) > 1]}")
    print(f"  proposer: bigram over library symbols, fit on the training solutions\n")
    print(f"  {'task set':<12}{'uniform nodes':>16}{'guided nodes':>15}{'speedup':>12}")
    print("  " + "-" * 55)
    out = {}
    for name, tasks in [("related", TEST_RELATED), ("control", CONTROL)]:
        uni, gui = [], []
        for t in tasks:
            exs = [(i, run(t, i)) for i in INPUTS]
            uni.append(uniform_solve(exs, lib)[1])
            gui.append(guided_solve(exs, lib, logp)[1])
        u, g = median(uni), median(gui)
        out[name] = (u, g)
        print(f"  {name:<12}{u:>16,}{g:>15,}{u / max(g,1):>11.1f}x")
    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    ur, gr = out["related"]
    uc, gc = out["control"]
    print(f"  RELATED:  uniform {ur:,} -> guided {gr:,} nodes  ({ur/max(gr,1):.0f}x further speedup)")
    print(f"  CONTROL:  uniform {uc:,} -> guided {gc:,} nodes  "
          f"({'no real gain' if gc >= uc * 0.7 else f'{uc/max(gc,1):.1f}x'})")
    print("\n  The library made big steps possible (v3); the learned proposer points")
    print("  them in the right direction — a further cut where structure was learned,")
    print("  ~nothing where it wasn't. The learner serves the search; the objective")
    print("  (find a short program) never changed. Domesticated, and earning its keep.")


if __name__ == "__main__":
    main()
