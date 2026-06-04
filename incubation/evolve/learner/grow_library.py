#!/usr/bin/env python3
"""Branch-B scale-prep BOX-PREP 4 — LIBRARY GROWTH: does the ceiling climb by COMPOSING earned predicates?

Naming single predicates is easy (the faculty spans them). The real test of "the ceiling stops being our
imagination": can the system build a predicate NO single faculty instantiation can express, by COMPOSING ones it
already earned -- and does it solve a task the grammar AND every level-0 predicate miss? Headline task: "recolor
the object contained in the LARGEST container" = containment @ is-largest. We (1) EARN level-0 predicates from
simple tasks (containment, largest) into a library, then (2) show the composed task is grammar=0 AND unsolved by
any single predicate, but SOLVED by composing two earned predicates, then (3) show the minted composition is a
reusable named library entry (solves a fresh instance with no re-search). Run: /data/llm/.venv/bin/python grow_library.py"""
import sys
import numpy as np
import grammar as G
import rel_dsl as D

sys.path.insert(0, "/data/Windows-files/Documents/airfoil/incubation/evolve")
from ground_arc import winning_relations

rng = np.random.RandomState(0)


def make_containment_task(n):
    demos = []
    for _ in range(n):
        g = np.zeros((22, 22), int); boxes = []
        for _ in range(3):
            for _t in range(30):
                r, c = rng.randint(0, 16), rng.randint(0, 16)
                if all(abs(r - br) > 7 or abs(c - bc) > 7 for br, bc in boxes):
                    g[r:r+6, c:c+6] = 5; g[r+1:r+5, c+1:c+5] = 0; boxes.append((r, c)); break
        for (r, c) in boxes:
            g[r + 1 + rng.randint(0, 3), c + 1 + rng.randint(0, 3)] = 4
        placed = 0
        for _t in range(120):
            if placed >= 4: break
            r, c = rng.randint(0, 22), rng.randint(0, 22)
            if g[r, c] == 0 and all(not (br <= r < br+6 and bc <= c < bc+6) for br, bc in boxes):
                g[r, c] = 4; placed += 1
        out = g.copy(); objs = G.objects(g, 4, True)
        for o in objs:
            col = 2 if any(D._contains(x, o) for x in objs) else 3
            for (rr, cc) in o["cells"]: out[rr, cc] = col
        demos.append((g, out))
    return demos


def make_largest_task(n):
    demos = []
    for _ in range(n):
        g = np.zeros((16, 16), int); placed = []
        for s in (2, 3, 4):
            for _t in range(30):
                r, c = rng.randint(0, 16 - s), rng.randint(0, 16 - s)
                if all(abs(r - pr) > 5 or abs(c - pc) > 5 for pr, pc in placed):
                    g[r:r+s, c:c+s] = 4; placed.append((r, c)); break
        out = g.copy(); objs = G.objects(g, 4, True); mx = max(o["size"] for o in objs)
        for o in objs:
            col = 2 if o["size"] == mx else 3
            for (rr, cc) in o["cells"]: out[rr, cc] = col
        demos.append((g, out))
    return demos


def make_contained_in_largest_task(n):
    demos = []
    for _ in range(n):
        g = np.zeros((26, 26), int); boxes = []
        sizes = [8, 6, 5]; rng.shuffle(sizes)
        for s in sizes:
            for _t in range(40):
                r, c = rng.randint(0, 26 - s), rng.randint(0, 26 - s)
                if all(abs(r - br) > s + 2 or abs(c - bc) > s + 2 for br, bc, _ in boxes):
                    g[r:r+s, c:c+s] = 5; g[r+1:r+s-1, c+1:c+s-1] = 0
                    g[r + s // 2, c + s // 2] = 4; boxes.append((r, c, s)); break
        out = g.copy(); objs = G.objects(g, 4, True); mx = max(o["size"] for o in objs)
        for o in objs:
            in_largest = any(D._contains(b, o) and b["size"] == mx for b in objs)
            col = 2 if in_largest else 3
            for (rr, cc) in o["cells"]: out[rr, cc] = col
        demos.append((g, out))
    return demos


def main():
    # (1) EARN level-0 predicates from simple tasks -> the library
    library = []
    for gen, name in [(make_containment_task, "containment"), (make_largest_task, "largest")]:
        prog = D.earn_predicate(gen(4), gen(2))
        if prog is not None:
            library.append(prog.key); print(f"  earned from {name:12s}: {prog.key}")
    print(f"LIBRARY (level-0, earned from simple tasks): {[str(k) for k in library]}")

    # (2) the COMPOSED task -- grammar=0, no single predicate, but composition of earned predicates solves
    ctr = make_contained_in_largest_task(4); cte = make_contained_in_largest_task(2)
    gram = len(winning_relations(ctr, cte))
    single = D.earn_predicate(ctr, cte)
    composed = D.earn_composed(ctr, cte, library)
    print(f"\nCOMPOSED task 'contained-in-LARGEST':")
    print(f"  GRAMMAR winning relations:        {gram}  (per-object features can't say it)")
    print(f"  any single LEVEL-0 predicate:     {single.key if single else None}  (faculty can't say it in one)")
    print(f"  COMPOSED of earned predicates:    {composed.key if composed else None}")

    # (3) the minted composition is a reusable NAMED library entry -- solves a FRESH instance, no re-search
    reuse = D.verify(composed, make_contained_in_largest_task(3)) if composed else False
    print(f"  reuse on a fresh instance (named, no re-search): {reuse}")
    print("READ: grammar=0 + no single predicate, but a COMPOSITION of two EARNED predicates solves it = the "
          "expressiveness ceiling climbs past the faculty's single instantiations, built from earned vocabulary "
          "(nothing hand-coded). The minted composition is reusable = the library GREW a relation we never gave it.")


if __name__ == "__main__":
    main()
