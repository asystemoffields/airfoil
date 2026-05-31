#!/usr/bin/env python3
"""
v14: the sleep must SUPPRESS — a governed wake-sleep that prunes, and holds.

v13 showed the loop bootstraps (round 0->1) but then BLOATS: ungoverned, the
sleep hoards every recurring fragment, the vocabulary balloons, search branching
rises, and held-out accuracy DECLINES. The loop eats itself.

Biological intelligence is mostly inhibition — pruning synapses in sleep,
suppressing the irrelevant, learning to forget. So the fix isn't a patch; it's
the missing half of the mechanism. v14 governs the sleep with a two-part MDL
razor (v2's idea, now as a *filter*): a macro is kept only if adding it REDUCES
the total description length of everything solved — its reuse savings must exceed
both its own storage cost and the rising cost of a bigger codebook. Junk
composites (used once, or not enough to pay their way) are suppressed.

We run governed vs ungoverned head-to-head on the same tasks and budget.
Pure stdlib.
"""
import itertools
import math
import random
from collections import Counter

random.seed(13)  # same draw as v13 for a fair head-to-head

OP = {
    "inc*": lambda L: [x + 1 for x in L], "dec*": lambda L: [x - 1 for x in L],
    "dbl*": lambda L: [2 * x for x in L], "sqr*": lambda L: [x * x for x in L],
    "neg*": lambda L: [-x for x in L],
    "even": lambda L: [x for x in L if x % 2 == 0],
    "pos":  lambda L: [x for x in L if x > 0],
    "rev":  lambda L: L[::-1], "tail": lambda L: L[1:], "srt": lambda L: sorted(L),
}
BASE = list(OP)
LOGB = math.log2(len(BASE))
IDIOMS = {"A": ("even", "dbl*"), "B": ("pos", "inc*"), "C": ("rev", "tail"),
          "D": ("sqr*", "neg*")}
NAMES = list(IDIOMS)
EX_LISTS = [[1, -2, 3, -4, 5], [2, 4, -6, 8], [-1, -3, 0, 2, 6], [5, -5, 10, -3, 0]]
BUDGET = 15000
MAXD = 6


def run(seq, L):
    for op in seq:
        L = OP[op](L)
    return L


def io(prog):
    return [run(prog, list(L)) for L in EX_LISTS]


def expand(combo):
    return tuple(op for s in combo for op in s)


def solve(target, vocab):
    n = 0
    for d in range(1, MAXD + 1):
        for combo in itertools.product(vocab, repeat=d):
            n += 1
            if n > BUDGET:
                return None
            prog = expand(combo)
            if len(prog) <= MAXD and all(run(prog, list(L)) == o
                                         for L, o in zip(EX_LISTS, target)):
                return prog
    return None


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


def candidates(corpus, vocab):
    pairs = Counter()
    for q in corpus:
        s = seg(q, vocab)
        for a, b in zip(s, s[1:]):
            pairs[(a, b)] += 1
    return pairs


def sleep_ungoverned(corpus):
    """v13: greedily merge the most frequent pair until none recurs. Hoards."""
    vocab = [(b,) for b in BASE]
    while True:
        pairs = candidates(corpus, vocab)
        if not pairs:
            break
        (a, b), c = max(pairs.items(), key=lambda kv: kv[1])
        if c < 2:
            break
        vocab.append(a + b)
    return vocab


def total_mdl(corpus, vocab):
    lib = sum(len(s) * LOGB for s in vocab if len(s) > 1)          # define each macro
    bits = math.log2(len(vocab))
    data = sum(len(seg(q, vocab)) for q in corpus) * bits          # name each symbol used
    return lib + data


def sleep_governed(corpus):
    """Keep a macro ONLY if it reduces total two-part MDL. Suppress the rest."""
    vocab = [(b,) for b in BASE]
    while True:
        cands = [a + b for (a, b), c in candidates(corpus, vocab).items() if c >= 2]
        cur = total_mdl(corpus, vocab)
        best, best_mdl = None, cur
        for m in cands:
            if m in vocab:
                continue
            mdl = total_mdl(corpus, vocab + [m])
            if mdl < best_mdl:
                best, best_mdl = m, mdl
        if best is None:                 # nothing pays its way -> stop (the razor)
            break
        vocab.append(best)
    return vocab


def comp(names):
    return tuple(op for nm in names for op in IDIOMS[nm])


def solve_rate(tasks, vocab):
    return sum(solve(io(p), vocab) is not None for p in tasks) / len(tasks)


def run_loop(sleep_fn, train, held, R=5):
    vocab = [(b,) for b in BASE]
    solved, rows = [], []
    for r in range(R):
        rows.append((len([s for s in vocab if len(s) > 1]), solve_rate(held, vocab)))
        for p in train:
            prog = solve(io(p), vocab)
            if prog is not None:
                solved.append(prog)
        new = sleep_fn(solved)
        if len(new) == len(vocab):
            rows.append((len([s for s in vocab if len(s) > 1]), solve_rate(held, vocab)))
            break
        vocab = new
    return rows, vocab


def main():
    train = [comp([n]) for n in NAMES]
    train += [comp(list(c)) for c in random.sample(list(itertools.product(NAMES, repeat=2)), 10)]
    train += [comp(list(c)) for c in random.sample(list(itertools.product(NAMES, repeat=3)), 12)]
    train += [comp(list(c)) for c in random.sample(list(itertools.product(NAMES, repeat=4)), 10)]
    held = [comp(list(c)) for c in random.sample(list(itertools.product(NAMES, repeat=2)), 8)]
    held += [comp(list(c)) for c in random.sample(list(itertools.product(NAMES, repeat=3)), 10)]
    held += [comp(list(c)) for c in random.sample(list(itertools.product(NAMES, repeat=4)), 10)]

    print("=" * 78)
    print("v14  THE SLEEP MUST SUPPRESS — governed (MDL-pruned) vs ungoverned wake-sleep")
    print("=" * 78)
    print(f"  budget {BUDGET:,} nodes/task; held-out solve-rate & library size per round\n")

    ung_rows, ung_lib = run_loop(sleep_ungoverned, train, held)
    gov_rows, gov_lib = run_loop(sleep_governed, train, held)

    print(f"  {'round':<7}{'UNGOVERNED (v13)':>26}{'GOVERNED (v14)':>24}")
    print(f"  {'':<7}{'lib / held-out':>26}{'lib / held-out':>24}")
    print("  " + "-" * 56)
    for r in range(max(len(ung_rows), len(gov_rows))):
        u = ung_rows[r] if r < len(ung_rows) else ung_rows[-1]
        g = gov_rows[r] if r < len(gov_rows) else gov_rows[-1]
        print(f"  {r:<7}{f'{u[0]} macros / {u[1]*100:.0f}%':>26}{f'{g[0]} macros / {g[1]*100:.0f}%':>24}")

    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    u_final, g_final = ung_rows[-1], gov_rows[-1]
    print(f"  UNGOVERNED: bloats to {u_final[0]} macros, held-out peaks then settles at "
          f"{u_final[1]*100:.0f}%.")
    print(f"  GOVERNED:   keeps {g_final[0]} macros, held-out holds at {g_final[1]*100:.0f}%.")
    print(f"  kept (governed): {[' '.join(s) for s in gov_lib if len(s) > 1]}")
    print()
    print("  The razor at the sleep keeps the library small and the gain stable: a")
    print("  macro survives only if it pays for itself, so the loop stops hoarding the")
    print("  junk composites that bloated v13's search. Same bootstrap, no self-eating.")
    print("  Intelligence as much in what it SUPPRESSES as what it stores — the sleep's")
    print("  job is forgetting as well as learning. (Synaptic pruning, made of MDL.)")


if __name__ == "__main__":
    main()
