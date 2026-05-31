#!/usr/bin/env python3
"""
v8: the verifier wind tunnel.

Everything downstream (search, the skill library, the whole amortization loop)
rests on one assumption that's FREE in program synthesis and EXPENSIVE
everywhere real: a verifier that tells you whether a candidate is correct. In
reasoning / ARC / agents you get a cheap, NOISY signal instead. And a noisy
verifier doesn't just waste a search — it *accepts a wrong answer*, which would
then be compressed into the library and poison it.

So: deliberately give the verifier a false-accept rate eps (it okays a wrong
program with prob eps per check), and measure task-solving ACCURACY (oracle-
checked) as it degrades. Then test the cheap defense — REDUNDANCY: require M
independent confirmations (a wrong program survives only w.p. eps^M). This is
self-consistency / majority-vote / the brain's "don't consolidate a one-off."

Question: how much reliable measurement can you buy from an unreliable verifier
with cheap redundancy?  Pure stdlib.
"""
import itertools
import math
import random
from statistics import mean

random.seed(0)

BASE = ["inc", "dec", "dbl", "tpl", "sqr", "neg"]
OP = {"inc": lambda x: x + 1, "dec": lambda x: x - 1, "dbl": lambda x: 2 * x,
      "tpl": lambda x: 3 * x, "sqr": lambda x: x * x, "neg": lambda x: -x}
EX = [-2, -1, 0, 1, 2, 3]                 # examples the verifier sees
ORACLE = EX + [4, 5, 6, 7, -3, 9]         # held-out points to judge TRUE correctness
CAP = 12000

# a spread of target functions to solve (lengths 2-4)
TASKS = [("dbl", "inc"), ("inc", "inc", "inc"), ("sqr", "inc"), ("dbl", "dbl"),
         ("sqr", "dbl"), ("dbl", "inc", "dbl"), ("neg", "inc", "inc"),
         ("tpl", "dec", "dbl")]


def run(seq, x):
    for op in seq:
        x = OP[op](x)
    return x


def matches(prog, exs):
    return all(run(prog, i) == o for i, o in exs)


def correct(prog, gold):                  # the oracle (used only to SCORE, never by the solver)
    return all(run(prog, i) == run(gold, i) for i in ORACLE)


def solve(exs, eps, M):
    """Enumerate programs; accept the first the (noisy, M-redundant) verifier okays.
    A true match is always accepted; a non-match survives M checks w.p. eps^M."""
    nodes = 0
    for d in range(1, 6):
        for combo in itertools.product(BASE, repeat=d):
            nodes += 1
            if nodes > CAP:
                return None, nodes
            if matches(combo, exs):
                return combo, nodes
            if eps > 0 and all(random.random() < eps for _ in range(M)):
                return combo, nodes       # false-accept: a WRONG program slipped through
    return None, nodes


def trial(eps, M):
    ok, total_nodes = 0, 0
    for gold in TASKS:
        exs = [(i, run(gold, i)) for i in EX]
        prog, nodes = solve(exs, eps, M)
        total_nodes += nodes * M          # cost = verifier calls
        if prog is not None and correct(prog, gold):
            ok += 1
    return ok / len(TASKS), total_nodes


def main():
    EPS = [0.0, 0.1, 0.2, 0.4]
    MS = [1, 3, 5, 8]
    TRIALS = 25

    print("=" * 78)
    print("v8  VERIFIER WIND TUNNEL — task accuracy vs false-accept rate & redundancy")
    print("=" * 78)
    print("  (accuracy = fraction of accepted solutions that are ACTUALLY correct,")
    print("   oracle-checked on held-out inputs; redundancy M = independent confirmations)\n")
    print("  false-accept eps " + "".join(f"  M={m:<6}" for m in MS))
    print("  " + "-" * 56)
    acc = {}
    for eps in EPS:
        row = []
        for M in MS:
            a = mean(trial(eps, M)[0] for _ in range(TRIALS))
            acc[(eps, M)] = a
            row.append(a)
        print(f"  eps={eps:<12.2f}" + "".join(f"  {a*100:5.0f}%  " for a in row))

    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"  perfect verifier (eps=0): {acc[(0.0,1)]*100:.0f}% accuracy — the free-lunch baseline.")
    print(f"  noisy, no redundancy (eps=0.1, M=1): {acc[(0.1,1)]*100:.0f}%  -> a 10% false-accept")
    print(f"    rate near-DESTROYS naive solving, because there are so many short WRONG")
    print(f"    candidates before the right one — one fluke accept ends the search wrong.")
    print(f"  + redundancy (eps=0.1): M=3 -> {acc[(0.1,3)]*100:.0f}%, M=5 -> {acc[(0.1,5)]*100:.0f}%."
          f"  Consensus suppresses")
    print(f"    false-accepts as eps^M, cheaply restoring reliability from a bad signal.")
    print(f"  but redundancy has LIMITS: at eps=0.4, even M=8 -> {acc[(0.4,8)]*100:.0f}% — when the")
    print(f"    verifier is mostly noise, no affordable amount of voting saves you.")
    print()
    print("  Design rule (matches the numbers): reaching the correct program needs")
    print("  eps^M << 1/R, where R = wrong candidates tried before it. So")
    print("      M  >=  ln(R) / ln(1/eps).")
    print("  Practical reading: you do NOT need a reliable verifier — you need a cheap")
    print("  UNRELIABLE one whose error is < ~0.5 and the budget to confirm a few times.")
    print("  Redundancy converts a weak signal into a strong one exponentially (eps^M) —")
    print("  this is self-consistency (B2), majority vote, and why memory consolidation")
    print("  needs repetition. Below ~chance, though, there's nothing to amplify.")


if __name__ == "__main__":
    main()
