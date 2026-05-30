#!/usr/bin/env python3
"""
v0: Compression-as-generalization.

Thesis (Alex's): a system that learns reusable abstractions from some tasks
should be able to *describe* the solutions to new, structurally-related tasks
in fewer bits -- "within the particular is contained the universal."

Smallest honest test of that claim:
  1. A tiny DSL of unary integer ops. A task = input/output examples.
  2. INDUCTION sanity-check: recover programs from examples by search.
  3. SLEEP: BPE-style library learning over the training programs -- repeatedly
     fold the most-reused adjacent sub-program into one new abstraction.
  4. MEASURE description length (bits) of held-out solutions vs. #abstractions:
       - related : novel COMPOSITIONS of motifs the library can learn.
       - control : SAME primitive ops, compositions it never sees.
     If related compresses and control does not, the gain is real structural
     transfer, not generic compression. Pure stdlib; runs in <1s on CPU.
"""
import math
from collections import Counter

# ── base DSL: unary int -> int ───────────────────────────────────────────────
BASE_OPS = {"inc": lambda x: x + 1, "dec": lambda x: x - 1,
            "dbl": lambda x: 2 * x, "tpl": lambda x: 3 * x,
            "sqr": lambda x: x * x, "neg": lambda x: -x}
OP_NAMES = list(BASE_OPS)
INPUTS = [-3, -1, 0, 1, 2, 4, 7]


def run(seq, x):
    for op in seq:
        x = BASE_OPS[op](x)
    return x


def examples(seq):
    return [(i, run(seq, i)) for i in INPUTS]


# ── reusable motifs; train/test/control built from them ──────────────────────
M = {"M1": ("dbl", "inc"), "M2": ("inc", "inc", "inc"),
     "M3": ("sqr", "inc"), "M4": ("dbl", "dbl")}

TRAIN = [M["M1"], M["M2"], M["M3"], M["M4"],
         M["M1"] + ("dbl",), M["M2"] + M["M1"], M["M3"] + ("dbl",),
         M["M4"] + ("inc",), M["M1"] + M["M2"], M["M2"] + M["M2"],
         ("inc",) + M["M1"], M["M4"] + M["M1"], M["M3"] + ("dbl", "dbl"),
         M["M1"] + M["M1"]]

# novel compositions of the SAME motifs (none of these exact programs in train)
TEST_RELATED = [M["M3"] + M["M1"], M["M1"] + M["M3"], M["M4"] + M["M2"],
                M["M2"] + M["M4"], M["M1"] + M["M4"], M["M3"] + M["M4"]]

# control: only motif ops (inc/dbl/sqr), but orderings the library never learns
CONTROL = [("inc", "dbl"), ("inc", "sqr"), ("dbl", "sqr"), ("sqr", "dbl"),
           ("inc", "dbl", "sqr"), ("dbl", "sqr", "dbl")]


# ── induction: shortest base-op program matching the examples ────────────────
def solve(exs, max_len=6):
    if all(o == i for i, o in exs):
        return ()
    frontier = [()]
    for _ in range(max_len):
        nxt = []
        for prog in frontier:
            for op in OP_NAMES:
                cand = prog + (op,)
                if all(run(cand, i) == o for i, o in exs):
                    return cand
                nxt.append(cand)
        frontier = nxt
    return None


# ── library + frequency code (MDL), greedy BPE-style segmentation ────────────
class Library:
    def __init__(self):
        self.syms = [(op,) for op in OP_NAMES]   # always keep base ops
        self.code = {}

    def greedy(self, seq):
        """Longest-match segmentation into vocabulary symbols."""
        order = sorted(self.syms, key=len, reverse=True)
        out, i, n = [], 0, len(seq)
        while i < n:
            for s in order:
                if seq[i:i + len(s)] == s:
                    out.append(s)
                    i += len(s)
                    break
        return out

    def _fit(self, counts, alpha=0.5):
        tot = sum(counts.get(s, 0) for s in self.syms) + alpha * len(self.syms)
        self.code = {s: -math.log2((counts.get(s, 0) + alpha) / tot) for s in self.syms}

    def refit(self, corpus):
        c = Counter()
        for seq in corpus:
            c.update(self.greedy(seq))
        self._fit(c)

    def bits(self, seq):
        return sum(self.code[s] for s in self.greedy(seq))

    def best_merge(self, corpus):
        pairs = Counter()
        for seq in corpus:
            segs = self.greedy(seq)
            for a, b in zip(segs, segs[1:]):
                pairs[(a, b)] += 1
        if not pairs:
            return None
        (a, b), n = max(pairs.items(), key=lambda kv: (kv[1], -len(kv[0][0] + kv[0][1])))
        return a + b if n >= 2 else None

    def add(self, expansion):
        if expansion not in self.syms:
            self.syms.append(expansion)


def mean_bits(lib, progs):
    return sum(lib.bits(p) for p in progs) / len(progs)


def main():
    print("=" * 72)
    print("INDUCTION  (sanity: recover programs from I/O examples by search)")
    print("=" * 72)
    ntr = sum(solve(examples(s)) is not None for s in TRAIN)
    nre = sum(solve(examples(s)) is not None for s in TEST_RELATED)
    nco = sum(solve(examples(s)) is not None for s in CONTROL)
    print(f"  solved: train {ntr}/{len(TRAIN)} | related {nre}/{len(TEST_RELATED)}"
          f" | control {nco}/{len(CONTROL)}")

    print("\n" + "=" * 72)
    print("SLEEP + MEASURE  (bits/task on held-out sets as abstractions are learned)")
    print("=" * 72)
    lib = Library()
    lib.refit(TRAIN)
    print(f"\n  {'#abs':>4} {'vocab':>5} {'related b/task':>14} {'control b/task':>14}"
          f"   compression of related")
    print("  " + "-" * 68)
    rel0 = ctl0 = None
    curve = []
    for _ in range(10):
        rel, ctl = mean_bits(lib, TEST_RELATED), mean_bits(lib, CONTROL)
        if rel0 is None:
            rel0, ctl0 = rel, ctl
        curve.append((len([s for s in lib.syms if len(s) > 1]), rel, ctl))
        bar = "█" * int(round(max(0.0, (rel0 - rel) / rel0) * 34))
        print(f"  {len([s for s in lib.syms if len(s) > 1]):>4} {len(lib.syms):>5}"
              f" {rel:>14.2f} {ctl:>14.2f}   {bar}")
        m = lib.best_merge(TRAIN)
        if m is None:
            break
        lib.add(m)
        lib.refit(TRAIN)

    rel_end, ctl_end = curve[-1][1], curve[-1][2]
    print("\n" + "=" * 72)
    print("RESULT")
    print("=" * 72)
    print(f"  related : {rel0:6.2f} -> {rel_end:6.2f} bits/task  ({rel0 / rel_end:.2f}x)")
    print(f"  control : {ctl0:6.2f} -> {ctl_end:6.2f} bits/task  ({ctl0 / ctl_end:.2f}x)")
    learned = [" ".join(s) for s in lib.syms if len(s) > 1]
    print(f"\n  abstractions learned from training: {learned}")
    if rel0 / rel_end > 1.3 and (rel0 / rel_end) > 1.25 * (ctl0 / ctl_end):
        print("\n  ✓ TRANSFER: related solutions compressed; control did NOT.")
        print("    The abstractions are structural, not generic — the particular")
        print("    discovered in training reached novel compositions it never saw.")
    else:
        print("\n  ✗ no clean transfer signal — needs a richer domain or more merges.")


if __name__ == "__main__":
    main()
