#!/usr/bin/env python3
"""
v13: the loop bootstraps — iterative wake-sleep, and a solvable frontier that
expands from experience on a FIXED budget.

v12 was a single solve-then-compress pass. v13 closes it into a loop (DreamCoder's
wake-sleep; AlphaZero's search-then-distill; the brain's consolidation):

  WAKE   solve as many tasks as the current library allows, within a fixed budget.
  SLEEP  compress everything solved so far into reusable idioms (BPE).
  repeat.

The point: round 0 (base ops only) can only reach SHALLOW tasks within budget.
But compressing those solutions discovers the idioms, and with idioms as single
steps, DEEPER tasks fall inside the same budget next round — whose solutions
compress into still-bigger pieces, and so on. Competence compounds from its own
experience, with no increase in compute and no change to the base. We track the
solvable frontier on a HELD-OUT set to show it's generalization, not memorization.

Pure stdlib.
"""
import itertools
import random
from collections import Counter

random.seed(13)

OP = {
    "inc*": lambda L: [x + 1 for x in L], "dec*": lambda L: [x - 1 for x in L],
    "dbl*": lambda L: [2 * x for x in L], "sqr*": lambda L: [x * x for x in L],
    "neg*": lambda L: [-x for x in L],
    "even": lambda L: [x for x in L if x % 2 == 0],
    "pos":  lambda L: [x for x in L if x > 0],
    "rev":  lambda L: L[::-1], "tail": lambda L: L[1:], "srt": lambda L: sorted(L),
}
BASE = list(OP)
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
    """Return (base-op program or None, nodes). Budget-capped IDDFS over vocab."""
    n = 0
    for d in range(1, MAXD + 1):
        for combo in itertools.product(vocab, repeat=d):
            n += 1
            if n > BUDGET:
                return None, n
            prog = expand(combo)
            if len(prog) > MAXD:
                continue
            if all(run(prog, list(L)) == o for L, o in zip(EX_LISTS, target)):
                return prog, n
    return None, n


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


def sleep_compress(solved):
    """BPE over all solved base-programs -> a hierarchical library (idioms, then
    composites of idioms as bigger pieces recur)."""
    vocab = [(b,) for b in BASE]
    while True:
        pairs = Counter()
        for q in solved:
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


def comp(names):
    return tuple(op for nm in names for op in IDIOMS[nm])


def solve_rate(tasks, vocab):
    return sum(solve(io(p), vocab)[0] is not None for p in tasks) / len(tasks)


def main():
    # curriculum: idiom singles + a spread of depths 2-4 (phrase-structured so
    # bigger pieces recur). held-out: novel compositions, depths 2-4.
    train = [comp([n]) for n in NAMES]                                  # depth 1
    train += [comp(list(c)) for c in random.sample(
        list(itertools.product(NAMES, repeat=2)), 10)]                  # depth 2
    train += [comp(list(c)) for c in random.sample(
        list(itertools.product(NAMES, repeat=3)), 12)]                  # depth 3
    train += [comp(list(c)) for c in random.sample(
        list(itertools.product(NAMES, repeat=4)), 10)]                  # depth 4
    held = [comp(list(c)) for c in random.sample(
        list(itertools.product(NAMES, repeat=2)), 8)]
    held += [comp(list(c)) for c in random.sample(
        list(itertools.product(NAMES, repeat=3)), 10)]
    held += [comp(list(c)) for c in random.sample(
        list(itertools.product(NAMES, repeat=4)), 10)]

    print("=" * 78)
    print("v13  THE LOOP BOOTSTRAPS — iterative wake-sleep on a FIXED budget")
    print("=" * 78)
    print(f"  budget {BUDGET:,} nodes/task; {len(train)} train + {len(held)} held-out tasks\n")
    print(f"  {'round':<7}{'library size':>14}{'train solved':>15}{'HELD-OUT solved':>18}")
    print("  " + "-" * 54)

    vocab = [(b,) for b in BASE]
    solved_progs = []
    for r in range(5):
        tr = solve_rate(train, vocab)
        hl = solve_rate(held, vocab)
        nmac = len([s for s in vocab if len(s) > 1])
        print(f"  {r:<7}{f'{len(vocab)} ({nmac} macro)':>14}{tr*100:>13.0f}% {hl*100:>16.0f}%")
        # WAKE: solve train, accumulate solutions
        for p in train:
            prog, _ = solve(io(p), vocab)
            if prog is not None:
                solved_progs.append(prog)
        # SLEEP: recompress everything solved so far
        new = sleep_compress(solved_progs)
        if len(new) == len(vocab):          # converged — no new structure to find
            break
        vocab = new

    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    learned = [" ".join(s) for s in vocab if len(s) > 1]
    print(f"  final library ({len(learned)} pieces, MANY junk composites): {learned[:6]} ...")
    print(f"  ground-truth idioms were {[' '.join(IDIOMS[n]) for n in NAMES]} (never told).")
    print()
    print("  THE BOOTSTRAP IS REAL (round 0 -> 1): just by discovering the idioms from")
    print("  its own solutions, the solvable frontier jumps ~58->83% (train) and")
    print("  ~46->79% (HELD-OUT) on a fixed budget with a frozen base. Competence")
    print("  compounding from experience, and generalizing (held-out rises with train).")
    print()
    print("  BUT IT THEN PLATEAUS AND HELD-OUT DECLINES — an honest, useful failure:")
    print("  the library grows UNCHECKED (junk composites from shortest-equivalent")
    print("  scrambling), and a bloated vocabulary raises the search branching factor")
    print("  (the v3/v6 tax), re-burying some tasks past budget. Ungoverned compression")
    print("  bloats and degrades.")
    print()
    print("  Lesson -> v14: crystallization must be GOVERNED — keep only pieces that pay")
    print("  their way (MDL, v4) and gate them (consensus, v10). The loop bootstraps;")
    print("  unbounded it eats itself. The wake-sleep needs an Occam razor at the sleep.")


if __name__ == "__main__":
    main()
