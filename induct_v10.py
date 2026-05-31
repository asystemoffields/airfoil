#!/usr/bin/env python3
"""
v10: the first end-to-end loop test — does a diverse-verifier gate stop library
poisoning, and how badly does poisoning amplify through reuse?

The whole point of a library is REUSE. That's also its danger: if a weak verifier
lets a WRONG program crystallize as a skill, every future composition that reuses
that skill inherits the error. A one-off mistake becomes a structural one.

Loop:
  1. Crystallize 4 primitive skills (the motifs) by solving each under a verifier.
  2. Solve held-out COMPOSITIONS (depth-2 and depth-3) purely by composing the
     crystallized skills — no new search.
  3. Score skills and compositions against the oracle.

Verifier = partial test suite (v9's systematic-weakness model). The GATE strength
K = how many DIVERSE partial suites must agree before a skill is crystallized
(K=1 = no gate). We measure skill correctness AND composition correctness vs K,
to see (a) whether diversity cleans the library, and (b) how poisoning amplifies
with composition depth.  Pure stdlib.
"""
import itertools
import random
from statistics import mean

random.seed(0)

BASE = ["inc", "dec", "dbl", "tpl", "sqr", "neg"]
OP = {"inc": lambda x: x + 1, "dec": lambda x: x - 1, "dbl": lambda x: 2 * x,
      "tpl": lambda x: 3 * x, "sqr": lambda x: x * x, "neg": lambda x: -x}
EX = [-3, -2, -1, 0, 1, 2, 3, 4]
ORACLE = EX + [5, 6, 7, -4, 8, 9]
CAP = 12000

SKILLS = {"M1": ("dbl", "inc"), "M2": ("inc", "inc", "inc"),
          "M3": ("sqr", "inc"), "M4": ("dbl", "dbl")}
NAMES = list(SKILLS)


def run(seq, x):
    for op in seq:
        x = OP[op](x)
    return x


def solve_on(gold, inputs):
    inputs = sorted(set(inputs))
    n = 0
    for d in range(1, 6):
        for combo in itertools.product(BASE, repeat=d):
            n += 1
            if n > CAP:
                return None
            if all(run(combo, i) == run(gold, i) for i in inputs):
                return combo
    return None


def matches_oracle(prog, gold):
    return prog is not None and all(run(prog, i) == run(gold, i) for i in ORACLE)


def crystallize(gold, K):
    """Accept the first program matching gold on the UNION of K diverse 1-case
    suites (K=1 = a single feeble verifier, no gate)."""
    inputs = set()
    for _ in range(K):
        inputs.update(random.sample(EX, 1))
    return solve_on(gold, inputs)


def trial(K):
    lib = {name: crystallize(gold, K) for name, gold in SKILLS.items()}
    skill_ok = mean(matches_oracle(lib[n], SKILLS[n]) for n in NAMES)

    def comp_ok(combo):
        prog = tuple(op for n in combo for op in lib[n])
        gold = tuple(op for n in combo for op in SKILLS[n])
        return matches_oracle(prog, gold)

    d2 = mean(comp_ok(c) for c in itertools.product(NAMES, repeat=2))
    triples = list(itertools.product(NAMES, repeat=3))
    d3 = mean(comp_ok(c) for c in triples)
    return skill_ok, d2, d3


def main():
    KS = [1, 2, 3, 5]
    TRIALS = 40
    print("=" * 78)
    print("v10  LIBRARY POISONING & THE CONSENSUS GATE (first end-to-end loop test)")
    print("=" * 78)
    print("  gate K = # of diverse partial-suite verifiers that must agree to crystallize")
    print("  a skill. We reuse the crystallized skills to build held-out compositions.\n")
    print(f"  {'gate K':<10}{'skill correct':>15}{'depth-2 comp':>15}{'depth-3 comp':>15}")
    print("  " + "-" * 55)
    res = {}
    for K in KS:
        s, d2, d3 = (mean(x) for x in zip(*(trial(K) for _ in range(TRIALS))))
        res[K] = (s, d2, d3)
        print(f"  K={K:<8}{s*100:>13.0f}% {d2*100:>13.0f}% {d3*100:>13.0f}%")

    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    s1, d2_1, d3_1 = res[1]
    sN, d2N, d3N = res[KS[-1]]
    print(f"  No gate (K=1): skills only {s1*100:.0f}% correct — and look how it AMPLIFIES:")
    print(f"    depth-2 compositions {d2_1*100:.0f}%, depth-3 {d3_1*100:.0f}%. A composition is")
    print(f"    correct only if EVERY reused skill is — so poisoning compounds with depth")
    print(f"    (~skill_acc^depth). One bad primitive corrupts a whole family of reuses.")
    print(f"  Diverse gate (K={KS[-1]}): skills {sN*100:.0f}%, depth-2 {d2N*100:.0f}%, depth-3 {d3N*100:.0f}%.")
    print(f"    Manufacturing verifier independence at the crystallization step keeps the")
    print(f"    library clean — and because reuse amplifies, the gate matters MORE here")
    print(f"    than in one-shot solving: it protects every future composition at once.")
    print()
    print("  The loop closes: a cheap weak signal + diversity at the gate buys a clean,")
    print("  reusable library. Verification isn't just per-task insurance — in a")
    print("  reuse/amortization system it's the thing standing between you and")
    print("  compounding structural error.")


if __name__ == "__main__":
    main()
