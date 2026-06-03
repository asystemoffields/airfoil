#!/usr/bin/env python3
"""ARC grounding step 2 — a NON-LLM, feedback-guided SEARCH proposer (the toy's value-guided search, now with
a PERFECT world-model = the DSL interpreter). The LLM proposes BLIND (names ops, never executes); this proposer
EXECUTES candidate programs and beams toward the target by grid-distance — exploiting the exact world-model.

Fair budget = EXECUTIONS (programs evaluated on the full train set). All methods capped at B executions; a task
is SOLVED if some evaluated program reproduces ALL train outputs exactly (then we check test generalization).
  random : B random programs (length 1..3, ops uniform)
  enum   : first B programs in BFS order (length-1, then -2, then -3)
  search : best-first over the DSL guided by avg grid-distance-to-target (a heuristic value; learned value = next)
Discovers the length<=3 arena (bigger than the length<=2 enum arena of 23), then compares solve-rate vs budget.
Run with /data/llm/.venv/bin/python."""
import heapq, random, time, json
import numpy as np
import dsl
from proposer_eval import instantiate          # reuse palette-instantiation

f = lambda *a: print(*a, flush=True)
ARC = "/data/Windows-files/Documents/airfoil/incubation/arc"
NTASK = 250
DISCOVER_CAP = 5000
BUDGETS = (50, 200, 1000, 5000)


def gdist(a, b):
    if a is None: return 3.0
    if a.shape != b.shape: return 1.0 + abs(a.size - b.size) / max(a.size, b.size, 1)
    return float((a != b).mean())


def exact(outs, tgts):
    return all(o is not None and o.shape == t.shape and np.array_equal(o, t) for o, t in zip(outs, tgts))


def apply_all(outs, name, args):
    res = []
    for g in outs:
        try: res.append(dsl.OPS[name][0](g, *args))
        except Exception: return None
    return res


def search_solve(train, B, W=8, max_len=3):
    """best-first by avg grid-distance; returns (solved_prog|None, n_exec)."""
    insts = instantiate(dsl.palette(train))
    ins = [gi for gi, _ in train]; tgt = [go for _, go in train]
    start = ([], ins, sum(gdist(a, b) for a, b in zip(ins, tgt)) / len(ins))
    heap = [(start[2], 0, start[0], start[1])]; ctr = 1; nexec = 0
    while heap and nexec < B:
        score, _, prog, outs = heapq.heappop(heap)
        if len(prog) >= max_len: continue
        kids = []
        for inst in insts:
            outs2 = apply_all(outs, inst[0], inst[1]); nexec += 1
            if outs2 is None: continue
            if exact(outs2, tgt): return prog + [inst], nexec
            sc = sum(gdist(a, b) for a, b in zip(outs2, tgt)) / len(outs2)
            kids.append((sc, ctr, prog + [inst], outs2)); ctr += 1
            if nexec >= B: break
        # keep only the W best kids on the frontier (beam) to bound branching
        for k in heapq.nsmallest(W, kids): heapq.heappush(heap, k)
    return None, nexec


def enum_solve(train, B, max_len=3):
    insts = instantiate(dsl.palette(train)); ins = [gi for gi, _ in train]; tgt = [go for _, go in train]
    nexec = 0
    frontier = [([], ins)]
    for L in range(1, max_len + 1):
        nxt = []
        for prog, outs in frontier:
            for inst in insts:
                outs2 = apply_all(outs, inst[0], inst[1]); nexec += 1
                if outs2 is None:
                    if nexec >= B: return None, nexec
                    continue
                if exact(outs2, tgt): return prog + [inst], nexec
                nxt.append((prog + [inst], outs2))
                if nexec >= B: return None, nexec
        frontier = nxt
    return None, nexec


def random_solve(train, B, rng, max_len=3):
    insts = instantiate(dsl.palette(train)); ins = [gi for gi, _ in train]; tgt = [go for _, go in train]
    for _ in range(B):
        L = rng.randint(1, max_len); prog = [insts[rng.randrange(len(insts))] for _ in range(L)]
        outs = ins
        for name, args in prog:
            outs = apply_all(outs, name, args)
            if outs is None: break
        if outs is not None and exact(outs, tgt): return prog, _ + 1
    return None, B


def generalizes(prog, test):
    for gi, go in test:
        o = dsl.apply_prog(gi, prog)
        if o is None or o.shape != go.shape or not np.array_equal(o, go): return False
    return True


def main():
    tasks = dsl.load_all(dsl.TRAIN_DIR, n=NTASK)
    f(f"discovering length<=3 arena over {len(tasks)} tasks (search best-first, cap {DISCOVER_CAP})...")
    t0 = time.time(); arena = []
    for tid, train, test in tasks:
        prog, ne = search_solve(train, DISCOVER_CAP)
        if prog: arena.append((tid, train, test, prog))
    f(f"  arena = {len(arena)} length<=3 solvable tasks (vs 23 at length<=2) in {time.time()-t0:.0f}s")
    from collections import Counter
    lc = Counter(len(p) for _, _, _, p in arena); f(f"  by program length: {dict(sorted(lc.items()))}")

    rng = random.Random(0)
    res = {m: {B: {"tr": 0, "te": 0} for B in BUDGETS} for m in ("random", "enum", "search")}
    tg = time.time()
    for tid, train, test, _ in arena:
        for B in BUDGETS:
            for m, fn in (("random", lambda: random_solve(train, B, rng)),
                          ("enum", lambda: enum_solve(train, B)),
                          ("search", lambda: search_solve(train, B))):
                prog, _ne = fn()
                if prog:
                    res[m][B]["tr"] += 1
                    if generalizes(prog, test): res[m][B]["te"] += 1
    f(f"  compared in {time.time()-tg:.0f}s\n")

    N = len(arena)
    f(f"SOLVE RATE on {N}-task length<=3 arena (train-consistent / test-generalizing) vs EXECUTION budget:")
    f("    method      " + "    ".join(f"B={B:<4d}" for B in BUDGETS))
    for m in ("random", "enum", "search"):
        f(f"    {m:<10s} " + "   ".join(f"{res[m][B]['tr']:2d}/{res[m][B]['te']:2d}" for B in BUDGETS))
    f("\n  (reference: LLM-0.5B blind proposer earlier solved 3/4/4 at its proposal budgets 5/15/40)")
    f("READ: if SEARCH (feedback-guided, NON-LLM) dominates at small B and solves the length>=2 chains that")
    f("enum/random miss within budget, the perfect-world-model + grid-distance value IS the proposer — the")
    f("toy's value-guided-search result, grounded on ARC, with NO LLM. (heuristic value now; LEARNED value next.)")


if __name__ == "__main__":
    main()
