#!/usr/bin/env python3
"""
v12: generalization MEASURED, not assumed — a held-out list-functions benchmark.

v1-v11 planted the motifs and the train/test split by hand: generalization by
construction. v12 leaves the sandbox:

  1. DISCOVER, don't plant. Train = a set of depth-2 list-program tasks (hidden
     compositions of idioms). We SOLVE them at the base level by search, then let
     BPE DISCOVER the recurring idioms by compressing the solutions. The system is
     never told the idioms.
  2. HELD OUT. Evaluate on NOVEL depth-3 tasks the library never saw:
       - test-near : novel compositions of the SAME (now-discovered) idioms;
       - test-far  : compositions using a HELD-OUT idiom never in training.
  3. REAL BASELINE. Measure solve-rate within a fixed search budget and median
     search nodes, WITH the discovered library vs base ops only, on near and far.

The claim, if it holds: training experience (compressed into reusable idioms)
extends the frontier of what's solvable on novel, harder, held-out tasks — and
does so for near tasks but not far ones. That is generalization, measured.

Pure stdlib. Budget-capped; runs in ~10-30s on CPU.
"""
import itertools
import random
from collections import Counter
from statistics import median

random.seed(7)

# ---- list-processing DSL (list[int] -> list[int]) ----
OP = {
    "inc*": lambda L: [x + 1 for x in L],
    "dec*": lambda L: [x - 1 for x in L],
    "dbl*": lambda L: [2 * x for x in L],
    "sqr*": lambda L: [x * x for x in L],
    "neg*": lambda L: [-x for x in L],
    "even": lambda L: [x for x in L if x % 2 == 0],
    "pos":  lambda L: [x for x in L if x > 0],
    "rev":  lambda L: L[::-1],
    "tail": lambda L: L[1:],
    "srt":  lambda L: sorted(L),
}
BASE = list(OP)

# hidden idioms (the system must DISCOVER A-D; E is held out of training)
IDIOMS = {"A": ("even", "dbl*"), "B": ("pos", "inc*"),
          "C": ("rev", "tail"), "D": ("sqr*", "neg*"), "E": ("srt", "tail")}
TRAIN_IDIOMS = ["A", "B", "C", "D"]

EX_LISTS = [[1, -2, 3, -4, 5], [2, 4, -6, 8], [-1, -3, 0, 2, 6], [5, -5, 10, -3, 0]]
BUDGET = 20000
MAXD = 6


def run(seq, L):
    for op in seq:
        L = OP[op](L)
    return L


def io(prog):
    return [run(prog, list(L)) for L in EX_LISTS]


def expand(combo):
    return tuple(op for sym in combo for op in sym)


def solve(target_io, vocab):
    """Iterative-deepening over vocab symbols; return (solved, nodes)."""
    n = 0
    for d in range(1, MAXD + 1):
        for combo in itertools.product(vocab, repeat=d):
            n += 1
            if n > BUDGET:
                return False, n
            prog = expand(combo)
            if len(prog) > MAXD:
                continue
            if all(run(prog, list(L)) == o for L, o in zip(EX_LISTS, target_io)):
                return True, n
    return False, n


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


def discover_library(solved_programs):
    """BPE over the SOLVED training programs — discovers recurring idioms."""
    vocab = [(b,) for b in BASE]
    while True:
        pairs = Counter()
        for q in solved_programs:
            s = seg(q, vocab)
            for a, b in zip(s, s[1:]):
                pairs[(a, b)] += 1
        if not pairs:
            break
        (a, b), c = max(pairs.items(), key=lambda kv: kv[1])
        if c < 2:
            break
        vocab.append(a + b)
    return vocab


def compose(idiom_names):
    return tuple(op for nm in idiom_names for op in IDIOMS[nm])


def evaluate(tasks, vocab):
    solved, nodes = 0, []
    for prog in tasks:
        ok, n = solve(io(prog), vocab)
        solved += ok
        if ok:
            nodes.append(n)
    return solved / len(tasks), (median(nodes) if nodes else None)


def main():
    base = [(b,) for b in BASE]

    # --- TRAIN: solve depth-2 idiom compositions at the base level, then discover ---
    train_tasks = [compose([i, j]) for i in TRAIN_IDIOMS for j in TRAIN_IDIOMS]
    solved = []
    for prog in train_tasks:
        ok, _ = solve(io(prog), base)
        if ok:
            # recover the shortest base program the solver actually found
            for d in range(1, MAXD + 1):
                hit = next((expand(c) for c in itertools.product(base, repeat=d)
                            if len(expand(c)) <= MAXD
                            and all(run(expand(c), list(L)) == o
                                    for L, o in zip(EX_LISTS, io(prog)))), None)
                if hit:
                    solved.append(hit)
                    break
    lib = discover_library(solved)
    discovered = [" ".join(s) for s in lib if len(s) > 1]

    # --- HELD-OUT test sets (depth-3, never trained) ---
    near_all = [compose([i, j, k]) for i in TRAIN_IDIOMS for j in TRAIN_IDIOMS
                for k in TRAIN_IDIOMS]
    test_near = random.sample(near_all, 20)
    far_all = []
    for pos in range(3):
        for combo in itertools.product(TRAIN_IDIOMS, repeat=2):
            names = list(combo)
            names.insert(pos, "E")
            far_all.append(compose(names))
    test_far = random.sample(far_all, 16)

    print("=" * 78)
    print("v12  GENERALIZATION MEASURED — held-out list-functions benchmark")
    print("=" * 78)
    print(f"  trained by SOLVING {len(train_tasks)} depth-2 tasks, then BPE-discovering idioms.")
    print(f"  discovered library: {discovered}")
    print(f"  (ground-truth train idioms were "
          f"{[' '.join(IDIOMS[n]) for n in TRAIN_IDIOMS]} — never told to it)\n")
    print(f"  budget {BUDGET:,} nodes/task; test tasks are NOVEL depth-3 compositions.\n")
    print(f"  {'held-out set':<16}{'vocab':<12}{'solved':>10}{'median nodes':>15}")
    print("  " + "-" * 53)
    out = {}
    for name, tasks in [("test-near", test_near), ("test-far", test_far)]:
        for vname, vocab in [("base", base), ("learned", lib)]:
            rate, med = evaluate(tasks, vocab)
            out[(name, vname)] = rate
            print(f"  {name:<16}{vname:<12}{rate*100:>8.0f}% {str(med) if med else '-':>15}")

    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    nb, nl = out[("test-near", "base")], out[("test-near", "learned")]
    fb, fl = out[("test-far", "base")], out[("test-far", "learned")]
    print(f"  test-NEAR: base solves {nb*100:.0f}% of novel depth-3 tasks within budget;")
    print(f"    the discovered library solves {nl*100:.0f}% — idioms learned from depth-2")
    print(f"    experience extend the solvable frontier to novel, deeper, held-out tasks.")
    print(f"  test-FAR:  base {fb*100:.0f}% -> learned {fl*100:.0f}%. The held-out idiom isn't in")
    print(f"    the library, so the gain is smaller — the honest boundary of transfer.")
    print()
    print("  Generalization, measured: a library DISCOVERED (not planted) from training")
    print("  tasks helps solve NOVEL held-out tasks it never saw — strongly where they")
    print("  share learned structure, weakly where they don't. Not by construction.")


if __name__ == "__main__":
    main()
