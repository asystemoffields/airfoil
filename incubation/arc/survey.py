#!/usr/bin/env python3
"""ARC grounding — survey: how many tasks does the small DSL solve (length<=2)? Defines the arena where the
LLM-proposer-vs-random comparison is meaningful (tasks no DSL program solves give every proposer 0%).
Run with /data/llm/.venv/bin/python."""
import time
import dsl
from itertools import product

f = lambda *a: print(*a, flush=True)


def instantiate(pal):
    """all length-1 op instances given the task palette (color args from present colors that differ)."""
    insts = []
    for name in dsl.OP_NAMES:
        na = dsl.OPS[name][1]
        if na == 0:
            insts.append((name, ()))
        elif na == 1:                                       # fill_holes(color)
            for c in pal:
                if c != 0: insts.append((name, (c,)))
        elif na == 2:                                       # recolor(a,b)
            for a in pal:
                for b in pal:
                    if a != b: insts.append((name, (a, b)))
    return insts


def search(train, max_len=2, cap=6000):
    pal = dsl.palette(train); insts = instantiate(pal)
    for ins in insts:                                       # length 1
        if dsl.solves([ins], train): return [ins]
    tried = 0                                               # length 2 (capped)
    for a, b in product(insts, insts):
        if dsl.solves([a, b], train): return [a, b]
        tried += 1
        if tried >= cap: break
    return None


def main():
    tasks = dsl.load_all(dsl.TRAIN_DIR, n=400)
    f(f"surveying {len(tasks)} training tasks with DSL ({len(dsl.OP_NAMES)} ops), length<=2...")
    solved = []; t0 = time.time()
    for tid, train, test in tasks:
        prog = search(train)
        if prog: solved.append((tid, prog))
    f(f"\nSOLVED {len(solved)}/{len(tasks)} in {time.time()-t0:.0f}s")
    f("examples:")
    from collections import Counter
    opc = Counter()
    for tid, prog in solved:
        opc[tuple(n for n, _ in prog)] += 1
    for shape, c in opc.most_common(15):
        f(f"   {c:3d}x  {' -> '.join(shape)}")
    f("\nfirst 25 solved task ids:")
    f("   " + " ".join(t for t, _ in solved[:25]))
    # persist the solvable set for the LLM-vs-random experiment
    import json
    json.dump([{"tid": t, "prog": [[n, list(a)] for n, a in p]} for t, p in solved],
              open("/data/Windows-files/Documents/airfoil/incubation/arc/solvable.json", "w"))
    f("\nwrote solvable.json")


if __name__ == "__main__":
    main()
