#!/usr/bin/env python3
"""STANDARDIZED comparison: grade every gen-3 inventor against the BEST retrieval solver (gen2_base) as the
shared ablation, on held-out arc1-eval(400). 'Invention beyond retrieval' = solves gen2_base does NOT get.
Also report the combined coverage+invention ceiling (union)."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); EVOLVE = os.path.dirname(HERE); sys.path.insert(0, EVOLVE)
import harness

def solved_set(path, split="arc1-eval", B=4000):
    sol = harness.load_solver(os.path.join(EVOLVE, path))
    for hook in ("reset_library", "reset_state"):
        try:
            if hasattr(sol, hook): getattr(sol, hook)()
        except Exception: pass
    _m, rows = harness.evaluate(sol, harness.load_split(split), B)
    return set(r["task_id"] for r in rows if r["solved"])

base = solved_set("cand/gen2_base.py")
cd   = solved_set("cand/gen3_03_causal-decomposition.py")
co   = solved_set("cand/gen3_01_compositional-synthesis.py")
an   = solved_set("cand/gen3_02_analogical-repurposing.py")

print(f"gen2_base (best retrieval) eval solved: {len(base)}")
print(f"causal-decomp eval solved: {len(cd)}")
print(f"  causal-decomp BEYOND base (invented vs best-retrieval): {len(cd-base)}  {sorted(cd-base)}")
print(f"  union(base, causal-decomp): {len(base|cd)}")
print(f"compositional beyond base: {len(co-base)}  {sorted(co-base)}")
print(f"analogical beyond base: {len(an-base)}  {sorted(an-base)}")
allinv = cd | co | an
print(f"union(base, ALL 3 inventors): {len(base|allinv)}  (invention adds {len(allinv-base)} beyond best retrieval: {sorted(allinv-base)})")
print("=== done ===")
