#!/usr/bin/env python3
"""
v9: repetition vs DIVERSITY — the verifier caveat made quantitative.

v8 showed redundancy rescues a noisy verifier as eps^M — but it assumed the
errors were INDEPENDENT (noise re-rolled each check). Real verifiers fail
SYSTEMATICALLY: the same flawed check mis-judges the same case the same way every
time. Asking it M times gives the same answer M times, and eps^M collapses to eps.

Model a systematic-but-weak verifier as a PARTIAL TEST SUITE: it knows the true
answer on a subset S of the inputs and accepts any program matching there —
deterministically blind to differences outside S. (This is exactly real life:
a unit-test set, a check, a judge-prompt, each covering some cases and not others.)

- REPETITION: ask the SAME partial suite M times -> identical verdict -> no gain.
- DIVERSITY: M DIFFERENT partial suites (different covered cases). A wrong program
  must fool ALL of them -> must match the true answer on the UNION of their cases.
  As the union grows, only the genuinely-correct program survives.

Same number of checks; the question is whether you spend them on volume or variety.
Pure stdlib.
"""
import itertools
import random
from statistics import mean

random.seed(0)

BASE = ["inc", "dec", "dbl", "tpl", "sqr", "neg"]
OP = {"inc": lambda x: x + 1, "dec": lambda x: x - 1, "dbl": lambda x: 2 * x,
      "tpl": lambda x: 3 * x, "sqr": lambda x: x * x, "neg": lambda x: -x}
EX = [-3, -2, -1, 0, 1, 2, 3, 4]          # full evidence available
ORACLE = EX + [5, 6, 7, -4, 8, 9]
SUBSET = 1                                # each weak verifier covers only 1 case (deliberately feeble)
CAP = 12000

TASKS = [("dbl", "inc"), ("inc", "inc", "inc"), ("sqr", "inc"), ("dbl", "dbl"),
         ("sqr", "dbl"), ("dbl", "inc", "dbl"), ("neg", "inc", "inc"),
         ("tpl", "dec", "dbl")]


def run(seq, x):
    for op in seq:
        x = OP[op](x)
    return x


def solve_on(gold, inputs):
    """Accept the first enumerated program matching gold on the given input set."""
    inputs = sorted(set(inputs))
    nodes = 0
    for d in range(1, 6):
        for combo in itertools.product(BASE, repeat=d):
            nodes += 1
            if nodes > CAP:
                return None
            if all(run(combo, i) == run(gold, i) for i in inputs):
                return combo
    return None


def correct(prog, gold):
    return prog is not None and all(run(prog, i) == run(gold, i) for i in ORACLE)


def accuracy(make_inputs):
    """make_inputs() -> the set of inputs the verifier-policy effectively checks."""
    ok = 0
    for gold in TASKS:
        ok += correct(solve_on(gold, make_inputs()), gold)
    return ok / len(TASKS)


def main():
    MS = [1, 2, 3, 4, 5]
    TRIALS = 30
    print("=" * 78)
    print("v9  REPETITION vs DIVERSITY — rescuing a SYSTEMATICALLY weak verifier")
    print("=" * 78)
    print(f"  each weak verifier = a partial test suite covering {SUBSET} of {len(EX)} cases.")
    print(f"  same total checks (M); spent on repetition vs on variety.\n")
    print(f"  {'M (checks)':<14}" + "".join(f"{m:>9}" for m in MS))
    print("  " + "-" * 60)

    # REPETITION: one fixed subset, asked M times -> same inputs every time.
    rep = []
    for M in MS:
        a = mean(accuracy(lambda: random.sample(EX, SUBSET)) for _ in range(TRIALS))
        rep.append(a)  # M-independent by construction (re-asking the same suite adds nothing)
    print(f"  {'repetition':<14}" + "".join(f"{a*100:7.0f}% " for a in rep))

    # DIVERSITY: M different subsets -> union of their covered cases.
    div = []
    for M in MS:
        def mk():
            s = set()
            for _ in range(M):
                s.update(random.sample(EX, SUBSET))
            return s
        div.append(mean(accuracy(mk) for _ in range(TRIALS)))
    print(f"  {'diversity':<14}" + "".join(f"{a*100:7.0f}% " for a in div))

    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"  repetition stays flat (~{rep[-1]*100:.0f}%): re-asking one systematically-weak")
    print(f"    verifier reproduces its blind spot — no amount of volume corrects it.")
    print(f"  diversity climbs {div[0]*100:.0f}% -> {div[-1]*100:.0f}%: independent blind spots cover")
    print(f"    each other; a wrong program must fool the UNION, which shrinks the gap.")
    print(f"  SAME budget (M checks), opposite payoff. The lever is variety, not volume.")
    print()
    print("  Bridges v8: v8's independent-noise redundancy was the OPTIMISTIC limit")
    print("  (max diversity — every check decorrelated); repetition here is the")
    print("  PESSIMISTIC limit (zero diversity). Reality sits between, and the whole")
    print("  engineering job is manufacturing independence: different lenses, different")
    print("  evidence, different framings — not the same check louder. (Why biology")
    print("  cross-checks across senses, and Monte Carlo uses decorrelated restarts.)")
    print("  Verifier DIVERSITY, not verifier volume, is the ballgame.")


if __name__ == "__main__":
    main()
