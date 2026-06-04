#!/usr/bin/env python3
"""Independent full-split verification of the gen-3 inventors through the standardized invention gate.
Adds evolve/ to sys.path itself (the bug that crashed the first attempt was running from runs/)."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
EVOLVE = os.path.dirname(HERE)
sys.path.insert(0, EVOLVE)
import invention_gate as IG, harness

FILES = {
    "compositional": os.path.join(EVOLVE, "cand/gen3_01_compositional-synthesis.py"),
    "analogical":    os.path.join(EVOLVE, "cand/gen3_02_analogical-repurposing.py"),
    "causal-decomp": os.path.join(EVOLVE, "cand/gen3_03_causal-decomposition.py"),
}

for split in ("arc1-eval", "arc1-train"):
    tasks = harness.load_split(split)
    print(f"\n===== {split} (n={len(tasks)}) =====", flush=True)
    for name, f in FILES.items():
        sol = IG.load_solver(f)
        try:
            if hasattr(sol, "reset_library"):
                sol.reset_library()
        except Exception:
            pass
        m = IG.evaluate_invention(sol, tasks, 4000, label=f"{name}/{split}", log=False)
        print(f"  {name:14s} solved={m['solved']:3d} ablated={m['ablated_solved']:3d} "
              f"INVENTED={m['invented']:3d} certified={m.get('certified_invention')}", flush=True)
print("\n=== verify_gen3 complete ===", flush=True)
