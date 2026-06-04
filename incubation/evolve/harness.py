#!/usr/bin/env python3
"""DIY-AlphaEvolve campaign harness — the FITNESS FUNCTION for evolving a non-LLM ARC-AGI-2 solver.

A candidate SOLVER is a self-contained python module exposing:
    META = {"name": str, ...}                       # optional metadata
    solve(train, test_inputs, budget) -> attempts   # REQUIRED
where
    train       : list of (input_grid, output_grid)  numpy int2d
    test_inputs : list of input_grid                  numpy int2d
    budget      : int  (per-task search/exec budget the solver must respect)
    attempts    : list (one entry per test input) of up to 2 candidate output grids (numpy int2d),
                  best-first. ARC allows 2 attempts per test output.

Scoring: ARC 2-attempt rule. A task is SOLVED iff, for EVERY test pair, one of the (<=2) attempts for
that test input matches the held-out output EXACTLY (shape + values). We also report partial credit
(fraction of test outputs hit) for a finer fitness signal.

This module persists a json log for every run (the repo historically saved none). Import/run with
/data/llm/.venv/bin/python."""
import sys, os, json, time
import numpy as np

ARC = "/data/Windows-files/Documents/airfoil/incubation/arc"
if ARC not in sys.path:
    sys.path.insert(0, ARC)
import dsl  # single source of truth for the gen-0 grid world-model + task loader

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")

SPLITS = {
    "arc1-train": "/data/arc/data/training",
    "arc1-eval":  "/data/arc/data/evaluation",
    "arc2-train": "/data/arc-agi-2/data/training",
    "arc2-eval":  "/data/arc-agi-2/data/evaluation",
}


def load_split(name, n=None, ids=None):
    """Return [(task_id, train_pairs, test_pairs), ...] for a named split.
    n: take first n (sorted by id). ids: keep only these task ids (after the n cut)."""
    tasks = dsl.load_all(SPLITS[name], n=n)
    if ids is not None:
        keep = set(ids)
        tasks = [t for t in tasks if t[0] in keep]
    return tasks


def _match(pred, tgt):
    return pred is not None and getattr(pred, "shape", None) == tgt.shape and np.array_equal(pred, tgt)


def score_task(attempts, test):
    """attempts: list per-test-input of up to 2 grids. test: list of (inp, out).
    Returns (solved_all: bool, per_test_hit: list[bool], frac: float)."""
    per = []
    for k, (_gi, go) in enumerate(test):
        cand = attempts[k] if (attempts and k < len(attempts)) else []
        hit = any(_match(c, go) for c in (cand or [])[:2])
        per.append(bool(hit))
    frac = (sum(per) / len(per)) if per else 0.0
    return all(per), per, frac


def evaluate(solver, tasks, budget, log_path=None, label="", verbose=False):
    """Run solver.solve over tasks; return (metrics, rows). Robust to solver exceptions/timeouts (caught)."""
    t0 = time.time(); rows = []; solved = 0; partial = 0.0; errors = 0
    for i, (tid, train, test) in enumerate(tasks):
        test_inputs = [gi for gi, _ in test]
        err = None
        try:
            attempts = solver.solve(train, test_inputs, budget)
        except Exception as e:
            attempts = []; errors += 1; err = f"{type(e).__name__}: {e}"
        ok, per, frac = score_task(attempts, test)
        solved += int(ok); partial += frac
        rows.append({"task_id": tid, "solved": ok, "per_test": per, "frac": round(frac, 3), "err": err})
        if verbose and (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(tasks)}] solved={solved} partial={partial:.1f} ({time.time()-t0:.0f}s)", flush=True)
    dt = time.time() - t0
    n = max(len(tasks), 1)
    metrics = {"label": label, "n": len(tasks), "solved": solved,
               "solve_rate": round(solved / n, 4), "partial_credit": round(partial / n, 4),
               "errors": errors, "budget": budget, "seconds": round(dt, 1)}
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        json.dump({"metrics": metrics, "rows": rows}, open(log_path, "w"))
    return metrics, rows


def load_solver(path):
    """Import a candidate solver module from a file path (used by the evolve loop)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("cand_" + os.path.basename(path).replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    # quick self-test on the seed solver
    import seed_solver
    tasks = load_split("arc2-train", n=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    m, _ = evaluate(seed_solver, tasks, B, label="seed/arc2-train", verbose=True)
    print("SMOKE:", json.dumps(m))
