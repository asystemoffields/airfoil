#!/usr/bin/env python3
"""Find arc1-train tasks (gen2_base MISSES) that gen4_01 solves via a depth>=2 COMPOSITION.

We use gen4_01's OWN beam (_beam_search) exactly as its _invent calls it, then check which programs
verify on train AND have length>=2 AND actually reproduce the held-out TRAIN test output (gen4_01's
'solve on train'). This gives:
  (a) the count of train tasks gen4_01 composed-solves (depth>=2),
  (b) validation that our independent enumerator (in _diag_composition) finds compositions where they
      exist, and whether the composition generalizes on these (train) tasks' own test pairs.
Run with /data/llm/.venv/bin/python from .../incubation/evolve.
"""
import sys, os, json, time
import numpy as np

HERE = "/data/Windows-files/Documents/airfoil/incubation/evolve"
ARC = "/data/Windows-files/Documents/airfoil/incubation/arc"
for p in (HERE, os.path.join(HERE, "cand"), ARC):
    if p not in sys.path:
        sys.path.insert(0, p)
import dsl
import importlib.util
_spec = importlib.util.spec_from_file_location("g4", os.path.join(HERE, "cand", "gen4_01_relational-depth.py"))
G4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G4)

TRAIN_DIR = "/data/arc/data/training"


def _eq(a, b):
    return a is not None and getattr(a, "shape", None) == getattr(b, "shape", None) and np.array_equal(a, b)


def composed_solve(tid):
    """Return (composed_solved_train: bool, generalizes_to_test: bool, depth, prog_names, n_progs).
    composed_solved_train := gen4_01's beam (its actual budget) returns a verified prog of len>=2.
    We emulate gen4_01's _invent beam exactly."""
    tr, te = dsl.load_task(os.path.join(TRAIN_DIR, tid + ".json"))
    train = [(np.asarray(gi, int), np.asarray(go, int)) for gi, go in tr]
    test_pairs = [(np.asarray(gi, int), np.asarray(go, int)) for gi, go in te]
    alphabet, bg, colors = G4.build_alphabet(train)
    amap = dict(alphabet)
    # gen4_01 uses: beam_budget = max(800, min(budget, 6000)) with budget>=800 -> up to 6000.
    beam_budget = 6000
    beam_progs = G4._beam_search(train, alphabet, beam_budget, max_depth=4, beam_width=14, collect=2)
    comp_progs = [p for p in beam_progs if len(p) >= 2]
    if not comp_progs:
        return False, False, 0, None, len(beam_progs)
    # pick first composition; check it reproduces ALL test outputs (generalization for a TRAIN task)
    best = None
    gen = False
    for p in comp_progs:
        names = [n for n, _ in p]
        ok = True
        for ti, to in test_pairs:
            o = ti
            for _n, fn in p:
                o = fn(o) if o is not None else None
            if not _eq(o, to):
                ok = False
                break
        if ok:
            gen = True
            best = names
            break
    if best is None:
        best = [n for n, _ in comp_progs[0]]
    return True, gen, len(comp_progs[0]), best, len(beam_progs)


def main():
    base = json.load(open(os.path.join(HERE, "runs", "gen2_base_solved.json")))
    missed = sorted(base["arc1_train"]["missed"])
    print(f"scanning {len(missed)} gen2_base-missed arc1-train tasks for gen4_01 depth>=2 composed solves",
          flush=True)
    rows = []
    t0 = time.time()
    composed = 0
    composed_gen = 0
    for i, tid in enumerate(missed):
        try:
            cs, gen, depth, prog, npg = composed_solve(tid)
        except Exception as e:
            rows.append({"task_id": tid, "error": f"{type(e).__name__}: {e}"})
            continue
        if cs:
            composed += 1
            if gen:
                composed_gen += 1
            rows.append({"task_id": tid, "composed_solved_train": True, "generalizes_test": gen,
                         "depth": depth, "prog": prog, "n_beam_progs": npg})
        if (i + 1) % 40 == 0:
            print(f"  [{i+1}/{len(missed)}] composed={composed} composed&generalize={composed_gen} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    out = os.path.join(HERE, "runs", "_diag_train_composed.json")
    json.dump({"n_missed": len(missed), "composed": composed, "composed_generalize": composed_gen,
               "rows": rows}, open(out, "w"))
    print("\n===== TRAIN composed-solve summary =====")
    print(f"gen2_base-missed train tasks scanned: {len(missed)}")
    print(f"gen4_01 depth>=2 COMPOSED-solved (train-consistent, beyond base): {composed}")
    print(f"  of those, composition ALSO generalizes to that task's test pairs: {composed_gen}")
    print("WROTE", out, flush=True)
    print("composed task ids:", [r["task_id"] for r in rows if r.get("composed_solved_train")])


if __name__ == "__main__":
    main()
