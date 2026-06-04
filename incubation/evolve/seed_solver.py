#!/usr/bin/env python3
"""Gen-0 SEED solver for the DIY-AlphaEvolve campaign.

This is today's "current best non-LLM solver" wrapped in the campaign interface: best-first DSL program
search guided by average grid-distance-to-target, over the existing 33-op grid DSL (a PERFECT world-model =
the interpreter). It collects up to K train-consistent programs and returns the 2 shortest (MDL) as the
2 allowed attempts per test input. NO LLM, NO learned nets yet — those are levers the evolution will add.

Faithful to incubation/arc/search_proposer.py::search_solve; `instantiate` inlined to drop the llama dep.
Import/run with /data/llm/.venv/bin/python."""
import sys, heapq
import numpy as np

ARC = "/data/Windows-files/Documents/airfoil/incubation/arc"
if ARC not in sys.path:
    sys.path.insert(0, ARC)
import dsl

META = {"name": "seed_search_v0",
        "desc": "best-first grid-distance DSL search over 33 ops, MDL pick of 2 attempts, perfect world-model"}


def instantiate(pal):
    """All length-1 op instances; color args drawn from non-zero palette colors (recolor/swap need a!=b)."""
    colors = [c for c in pal if c != 0]
    insts = []
    for name, (_fn, nc) in dsl.OPS.items():
        if nc == 0:
            insts.append((name, ()))
        elif nc == 1:
            for c in colors:
                insts.append((name, (c,)))
        elif nc == 2:
            for a in colors:
                for b in colors:
                    if a != b:
                        insts.append((name, (a, b)))
    return insts


def _gdist(a, b):
    if a is None: return 3.0
    if a.shape != b.shape: return 1.0 + abs(a.size - b.size) / max(a.size, b.size, 1)
    return float((a != b).mean())


def _apply_all(outs, name, args):
    res = []
    for g in outs:
        try:
            res.append(dsl.OPS[name][0](g, *args))
        except Exception:
            return None
    return res


def _exact(outs, tgt):
    return all(o is not None and o.shape == t.shape and np.array_equal(o, t) for o, t in zip(outs, tgt))


def search_collect(train, B, W=25, max_len=3, K=2):
    """Best-first by avg grid-distance; collect up to K train-consistent programs. Returns (progs, n_exec)."""
    insts = instantiate(dsl.palette(train))
    ins = [gi for gi, _ in train]; tgt = [go for _, go in train]
    start = sum(_gdist(a, b) for a, b in zip(ins, tgt)) / len(ins)
    heap = [(start, 0, [], ins)]; ctr = 1; nexec = 0; found = []
    while heap and nexec < B:
        _score, _c, prog, outs = heapq.heappop(heap)
        if len(prog) >= max_len:
            continue
        kids = []
        for inst in insts:
            outs2 = _apply_all(outs, inst[0], inst[1]); nexec += 1
            if outs2 is None:
                if nexec >= B: break
                continue
            if _exact(outs2, tgt):
                found.append(prog + [inst])
                if len(found) >= K:
                    return found, nexec
            else:
                sc = sum(_gdist(a, b) for a, b in zip(outs2, tgt)) / len(outs2)
                kids.append((sc, ctr, prog + [inst], outs2)); ctr += 1
            if nexec >= B: break
        for k in heapq.nsmallest(W, kids):
            heapq.heappush(heap, k)
    return found, nexec


def _plen(p):
    return sum(1 + len(a) for _, a in p)


def solve(train, test_inputs, budget):
    progs, _ = search_collect(train, budget, K=4)
    progs = sorted(progs, key=_plen)[:2]          # MDL: 2 shortest train-consistent programs as the 2 attempts
    attempts = []
    for gi in test_inputs:
        cand = []
        for p in progs:
            o = dsl.apply_prog(gi, p)
            if o is not None:
                cand.append(o)
        attempts.append(cand)
    return attempts
