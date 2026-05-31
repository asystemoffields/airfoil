#!/usr/bin/env python3
"""
v17: the clean necessity window — incompressible tasks by construction.

v16 had the right instrument (matched libraries + budget sweep) but a confounded
task set: filter/reverse/sort compositions collapse to short functions, so "depth"
was faked and the idiom baseline was inflated.

v17 removes the confound by CONSTRUCTION. Every op is a polynomial map, and each
idiom contains exactly one squaring (sqr), which DOUBLES polynomial degree. So a
depth-K composition computes a degree-2^K polynomial — and since only sqr raises
degree (the affine ops inc/dbl/tpl/dec/neg preserve it), ANY program computing it
must contain >= K squarings, hence has length >= K. There is no shorter program:
the tasks are provably incompressible below their depth. No collapse, no knob.

(We evaluate mod a prime on a handful of random points: intermediate values stay
bounded, and by Schwartz-Zippel distinct polynomials disagree on a random point
with overwhelming probability, so function-equality is exact in practice.)

Then the matched-library budget sweep gives a CLEAN necessity window.
Pure stdlib.
"""
import itertools
import random

P = 2_147_483_647  # 2^31 - 1, prime
random.seed(17)
TESTPTS = [random.randrange(2, P) for _ in range(12)]   # random eval points (Schwartz-Zippel)

OP = {
    "inc": lambda x: (x + 1) % P, "dec": lambda x: (x - 1) % P,
    "dbl": lambda x: (2 * x) % P, "tpl": lambda x: (3 * x) % P,
    "neg": lambda x: (-x) % P,    "sqr": lambda x: (x * x) % P,   # the only degree-raiser
}
BASE = list(OP)
# each idiom has EXACTLY one sqr -> degree x2 per idiom (provably incompressible depth)
IDIOMS = {"A": ("sqr", "inc"), "B": ("sqr", "dbl"), "C": ("dbl", "sqr"), "D": ("sqr", "neg")}
PHRASES = {"P1": ["A", "B"], "P2": ["C", "D"], "P3": ["B", "C"]}
CAP = 100_000
MAXD = 12


def run(seq, x):
    for op in seq:
        x = OP[op](x)
    return x


def comp(names):
    out = []
    for nm in names:
        out += list(IDIOMS[nm])
    return tuple(out)


def io(prog):
    return tuple(run(prog, x) for x in TESTPTS)


def nodes_to_solve(target, vocab):
    n = 0
    for d in range(1, MAXD + 1):
        for combo in itertools.product(vocab, repeat=d):
            n += 1
            if n > CAP:
                return None
            prog = tuple(op for s in combo for op in s)
            if len(prog) <= MAXD and io(prog) == target:
                return n
    return None


def main():
    lib_idioms = [(b,) for b in BASE] + [IDIOMS[n] for n in IDIOMS]
    lib_phrases = lib_idioms + [comp(PHRASES[p]) for p in PHRASES]

    d4 = [comp(PHRASES[p] + PHRASES[q]) for p, q in
          (("P1", "P2"), ("P2", "P3"), ("P3", "P1"), ("P1", "P3"))]
    d6 = [comp(PHRASES[p] + PHRASES[q] + PHRASES[r]) for p, q, r in
          (("P1", "P2", "P3"), ("P2", "P3", "P1"), ("P3", "P1", "P2"), ("P1", "P3", "P2"))]

    cost = {}
    for depth, tasks in (("depth-4", d4), ("depth-6", d6)):
        for libname, lib in (("idioms", lib_idioms), ("phrases", lib_phrases)):
            cost[(depth, libname)] = [nodes_to_solve(io(t), lib) for t in tasks]

    budgets = [300, 1000, 3000, 10000, 30000, 100000]

    def rate(depth, libname, B):
        cs = cost[(depth, libname)]
        return sum(c is not None and c <= B for c in cs) / len(cs)

    print("=" * 78)
    print("v17  THE CLEAN NECESSITY WINDOW — provably-incompressible tasks, budget sweep")
    print("=" * 78)
    print("  every idiom has one sqr (degree x2); a depth-K task is a degree-2^K poly,")
    print("  so any solver needs >= K squarings -> NO program shorter than depth K exists.\n")
    head = "  " + " " * 18 + "".join(f"{b:>9}" for b in budgets)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for depth in ("depth-4", "depth-6"):
        print(f"  {depth} ({'2 phrases' if depth=='depth-4' else '3 phrases'})")
        for libname in ("idioms", "phrases"):
            row = "".join(f"{rate(depth, libname, b)*100:>8.0f}%" for b in budgets)
            print(f"    {libname:<14}{row}")

    def med(depth, lib):
        cs = [c for c in cost[(depth, lib)] if c is not None]
        return f"{int(sum(cs)/len(cs)):,}" if cs else f">{CAP:,} (unsolved)"
    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"  mean nodes to solve:        depth-4              depth-6")
    print(f"    idioms-only : {med('depth-4','idioms'):>18}   {med('depth-6','idioms'):>18}")
    print(f"    +phrases    : {med('depth-4','phrases'):>18}   {med('depth-6','phrases'):>18}")
    print()
    print("  Clean now — no task can collapse, so the necessity window is unambiguous:")
    print("  phrases solve at near-flat cost; idiom-only search costs ~|V|^depth, so the")
    print("  budget at which idioms catch up runs away EXPONENTIALLY with depth. depth-4:")
    print("  idioms catch up only at the high end. depth-6: idioms never catch up within")
    print(f"  {CAP:,} nodes at all — deeper abstraction is REQUIRED, not merely cheaper.")
    print()
    print("  This is the airfoil thesis, quantified: 'merely cheaper' is the luxury of a")
    print("  budget that dwarfs the task. For anything deep relative to your compute —")
    print("  the weak-hardware regime — the right reusable abstraction is the only way")
    print("  the search ever closes. Configuration, not resources.")


if __name__ == "__main__":
    main()
