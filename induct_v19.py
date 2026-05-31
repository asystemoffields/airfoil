#!/usr/bin/env python3
"""
v19 (cycle 1): the EXTERNAL benchmark — real ARC, honest baseline.

The first 18 versions worked on distributions WE designed. v19 points the apparatus
at ARC-AGI (fchollet/ARC, 400 public training tasks): grid -> grid transformations,
a distribution we did not make and cannot tune. This is the honest test — and a
clean win is not expected; ARC is unsolved, and small program-synthesis DSLs solve
only a slice.

Cycle 1 establishes the harness + a baseline: a parameter-free GEOMETRIC grid DSL
(rotate / flip / transpose / crop-to-content / tile / mirror) and a depth-bounded
search that must map EVERY train pair, then is scored on the held-out test grid
(exact match — ARC's own metric). No library, no policy yet — that's the next
cycles. The number this prints is the honest floor to improve on.

Pure stdlib.
"""
import glob
import itertools
import json

TASKS = sorted(glob.glob("/data/arc/data/training/*.json"))
MAXD = 3


def rot90(g):    return tuple(zip(*g[::-1]))
def rot180(g):   return tuple(r[::-1] for r in g[::-1])
def rot270(g):   return tuple(zip(*g))[::-1]
def flip_h(g):   return tuple(r[::-1] for r in g)
def flip_v(g):   return g[::-1]
def transpose(g):return tuple(zip(*g))
def tile_h(g):   return tuple(r + r for r in g)
def tile_v(g):   return g + g
def mirror_h(g): return tuple(r + r[::-1] for r in g)
def mirror_v(g): return g + g[::-1]


def crop(g):
    rows = [i for i, r in enumerate(g) if any(v != 0 for v in r)]
    cols = [j for j in range(len(g[0])) if any(g[i][j] != 0 for i in range(len(g)))]
    if not rows or not cols:
        return g
    return tuple(tuple(g[i][j] for j in range(cols[0], cols[-1] + 1))
                 for i in range(rows[0], rows[-1] + 1))


OPS = {"rot90": rot90, "rot180": rot180, "rot270": rot270, "flip_h": flip_h,
       "flip_v": flip_v, "transpose": transpose, "crop": crop,
       "tile_h": tile_h, "tile_v": tile_v, "mirror_h": mirror_h, "mirror_v": mirror_v}
OPNAMES = list(OPS)


def grid(g):
    return tuple(tuple(r) for r in g)


def apply(prog, g):
    try:
        for op in prog:
            g = grid(OPS[op](g))
        return g
    except Exception:
        return None


def solve(task):
    """Find an op-sequence mapping every train input to its output; None if none."""
    train = [(grid(p["input"]), grid(p["output"])) for p in task["train"]]
    # depth 0 (identity)
    if all(i == o for i, o in train):
        return ()
    for d in range(1, MAXD + 1):
        for prog in itertools.product(OPNAMES, repeat=d):
            if all(apply(prog, i) == o for i, o in train):
                return prog
    return None


def main():
    print("=" * 74)
    print("v19 (cycle 1)  ARC — honest baseline (parameter-free geometric DSL + search)")
    print("=" * 74)
    print(f"  {len(TASKS)} training tasks; DSL = {len(OPNAMES)} ops; search depth <= {MAXD}\n")
    solved_train = 0     # a program consistent with all train pairs found
    solved_test = 0      # ...and it also gets the held-out test grid right (ARC metric)
    hits = []
    for f in TASKS:
        task = json.load(open(f))
        prog = solve(task)
        if prog is None:
            continue
        solved_train += 1
        ok = all(apply(prog, grid(t["input"])) == grid(t["output"]) for t in task["test"])
        if ok:
            solved_test += 1
            hits.append((f.split("/")[-1][:8], " ".join(prog) or "identity"))
    n = len(TASKS)
    print(f"  found a train-consistent program : {solved_train}/{n}  ({solved_train/n*100:.1f}%)")
    print(f"  ...and correct on held-out test  : {solved_test}/{n}  ({solved_test/n*100:.1f}%)  <- ARC score")
    print(f"\n  solved tasks (id: program):")
    for tid, prog in hits[:20]:
        print(f"    {tid}  {prog}")
    print("\n" + "=" * 74)
    print("RESULT")
    print("=" * 74)
    print(f"  Honest baseline: {solved_test}/{n} ({solved_test/n*100:.1f}%) of real ARC training")
    print("  tasks fall to a parameter-free geometric DSL — exactly the small geometric")
    print("  slice you'd expect, and the floor to beat. ARC's hard mass needs recoloring,")
    print("  objects, counting, and parameterized ops — and a LIBRARY of reusable")
    print("  transforms learned across tasks (the airfoil loop), which is the next cycle.")
    print("  This is the adversary the whole project was built to eventually face.")


if __name__ == "__main__":
    main()
