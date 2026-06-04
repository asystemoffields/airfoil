#!/usr/bin/env python3
"""Vine — LEARNING LOOP gate 2 (genetic-algo gen-2): does the loop compound on REAL-DISTRIBUTION data (RE-ARC),
not just synthetic? The make-or-break for ever leaving the box. Two honest measurements:
  (1) EXPRESSIVENESS FLOOR: of N real RE-ARC relations, how many does the proposer (derive) even SOLVE? (a design
      explorer measured ~2/114 -- if derive solves ~nothing, the loop has no parents to abstract -> the bottleneck is
      the PROPOSER reaching real ARC, NOT the loop, and off-box compute is wasted.)
  (2) COMPOUNDING-ON-REAL: for relations derive solves >=2 same-relation instances of, does cost-per-solve DROP across
      instances (the schema minted from instance 1-2 cheapens instance 3+)?
Run: /data/llm/.venv/bin/python gate2.py [n_relations]"""
import sys, time, random
import numpy as np

# OUR modules FIRST so 'generators' resolves to ours (caches it), THEN load RE-ARC's generators by file path
from derive_grammar import derive
from learn_loop import parse, skeleton, antiunify_prog, has_holes, instantiate_prog

sys.path.append("/data/rearc_code/re-arc")
import importlib.util
_spec = importlib.util.spec_from_file_location("rearc_gen", "/data/rearc_code/re-arc/generators.py")
RG = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(RG)
REARC = {n[len("generate_"):]: getattr(RG, n) for n in dir(RG) if n.startswith("generate_")}


def make_task(gen, k_train=4, k_test=2, diff=(0.1, 0.4)):
    """build a (train, test) ARC-style task from k+m fresh instances of ONE RE-ARC relation."""
    pairs = []
    for _ in range(k_train + k_test):
        ex = gen(*diff)
        pairs.append((np.array(ex["input"], int), np.array(ex["output"], int)))
    return pairs[:k_train], pairs[k_train:]


def main():
    n_rel = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    random.seed(7)
    rels = list(REARC)[:n_rel]
    t0 = time.time()

    # (1) EXPRESSIVENESS FLOOR
    solved_rels = []
    for tid in rels:
        try:
            tr, te = make_task(REARC[tid])
            prog, _n = derive(tr, te, max_steps=2, budget=3000)
            if prog is not None:
                solved_rels.append((tid, prog))
        except Exception:
            continue
    print(f"GATE 2 — RE-ARC real-distribution data [{time.time()-t0:.0f}s]:")
    print(f"  (1) EXPRESSIVENESS FLOOR: derive solves {len(solved_rels)}/{len(rels)} relations "
          f"({100*len(solved_rels)/max(1,len(rels)):.0f}%)")
    for tid, prog in solved_rels[:8]:
        print(f"        {tid}: {prog}")

    # (2) COMPOUNDING-ON-REAL: for each solved relation, stream 10 fresh instances, loop with abstraction+reuse
    if solved_rels:
        comp = []
        for tid, _ in solved_rels:
            gen = REARC[tid]; library = []; by_skel = {}; costs = []
            for _ in range(10):
                tr, te = make_task(gen)
                n_task = 0; solved = False
                for sch in library:
                    r, n = instantiate_prog(sch, tr, te); n_task += n
                    if r is not None: solved = True; break
                if not solved:
                    prog, n = derive(tr, te, max_steps=2, budget=3000); n_task += n
                    if prog is not None:
                        solved = True; P = [parse(x) for x in prog]; sk = skeleton(P)
                        for prior in by_skel.get(sk, []):
                            s = antiunify_prog(prior, P)
                            if s and has_holes(s) and s not in library:
                                t3, e3 = make_task(gen); r3, _ = instantiate_prog(s, t3, e3)
                                if r3 is not None: library.append(s)
                        by_skel.setdefault(sk, []).append(P)
                if solved: costs.append(n_task)
            if len(costs) >= 4:
                first, last = np.median(costs[:2]), np.median(costs[-2:])
                comp.append((tid, len(costs), first, last, len(library)))
        print(f"\n  (2) COMPOUNDING-ON-REAL ({len(comp)} relations with >=4 same-relation solves):")
        for tid, ns, f, l, lib in comp:
            print(f"        {tid}: {ns}/10 solved, cost {f:.0f}->{l:.0f} ({l/max(1,f):.2f}x), library {lib}")
        if comp:
            ratios = [l / max(1, f) for _t, _n, f, l, _lib in comp]
            print(f"     median cost-ratio across compounding relations: {np.median(ratios):.2f}x "
                  f"(<1.0 = compounds on REAL data)")
    print("\nREAD: floor >> 0 AND median cost-ratio < 1 = the loop compounds on real-distribution data -> off-box "
          "scale (step 3) pays. Floor ~0 = the PROPOSER can't reach real ARC -> expressiveness is the wall, not the "
          "loop; compute won't fix it, and gen-3 must be the proposer (recognizer-pruned search / richer bases).")


if __name__ == "__main__":
    main()
