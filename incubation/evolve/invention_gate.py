#!/usr/bin/env python3
"""INVENTION-vs-RETRIEVAL GATE — the reusable harness that measures whether a solver
INVENTS a causal mechanism per task or RETRIEVES one from a fixed menu of whole-mechanism templates.

THE THESIS this gate operationalizes
------------------------------------
Creativity = (1) UNRESTRICTED grasp of cause-and-effect: induce the INVARIANT causal mechanism that
generates a task's train pairs (cross-example invariance licenses CAUSAL vs merely correlational
induction; the held-out test is the exact intervention that verifies it) AND (2) real-time INVENTION
of that mechanism: not retrieving/ranking from a fixed menu of whole-mechanism templates, but
SYNTHESIZING a new cause->effect rule per task by COMPOSING + ABSTRACTING primitive relations, made
FAST by an experience-prior.

CRITICAL DISTINCTION (the alphabet vs the sentence):
  * PRIMITIVES (dsl.py's 32 ops + grid utils) are the ALPHABET — knowledge, free to use.
  * A MECHANISM is an invented SENTENCE — a novel composition/abstraction/repurposing of primitives.
  * gen-1/2 RETRIEVED whole sentences from a menu (template induction). That is exactly what gen-3
    must transcend: the inventor synthesizes the sentence.

THE CERTIFICATION (what makes this a GATE, not just a scorer)
------------------------------------------------------------
A solve counts as CREATIVE only if it survives an ABLATION that disables invention
(composition / abstraction / repurposing) — i.e. a single whole-template RETRIEVAL cannot produce it.

  INVENTED = (solved by full solver) - (solved by the same solver with invention disabled)

The ablation is the solver's own `solve_ablated`. The gate does NOT trust a number; it requires the
solver to PROVE the solve is unreachable by retrieval alone, by failing it under ablation.

SOLVER CONVENTIONS (a gen-3 inventor module SHOULD expose)
----------------------------------------------------------
    solve(train, test_inputs, budget)          -> attempts          [REQUIRED — same as harness.py]
    solve_ablated(train, test_inputs, budget)   -> attempts          [OPTIONAL — invention DISABLED:
            the SAME solver restricted to single-whole-template retrieval/ranking; NO composition,
            NO abstraction, NO repurposing. If absent: ablated_solved=0 and a flag is raised
            ("cannot certify invention") — we never credit invention we cannot disprove by ablation.]
    reset_library()                             -> None              [OPTIONAL — clears the cross-task
            experience library (module-level state accumulated from PRIOR verified solves this run),
            so a run starts cold. Used by transfer_invention to isolate experience-transfer. If absent:
            transfer reports None with a flag.]

INTEGRITY (hard rules the gate itself obeys, and that solvers are trusted to obey):
  solve()/solve_ablated() learn ONLY from (a) the current task's train pairs, (b) module-level state
  from PRIOR solve() calls this run (verified-correct only), (c) self-generated synthetic data built at
  import. NEVER read ARC task files or test OUTPUTS (test INPUTS only). No network, no LLM at solve
  time. Respect budget. The gate passes solvers ONLY (train, test_inputs, budget) — it physically
  withholds held-out outputs from the solver and scores them itself via harness.score_task.

Reuses harness.load_split / harness.score_task / harness._match. Importable: `import invention_gate`.
Run with /data/llm/.venv/bin/python from /data/Windows-files/Documents/airfoil/incubation/evolve.
"""
import os
import sys
import json
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Single source of truth for split loading + ARC 2-attempt scoring + exact match.
import harness
from harness import load_split, score_task, _match  # re-exported for callers of the gate

RUNS = os.path.join(HERE, "runs")


# ---------------------------------------------------------------------------
# capability probes — what hooks does this solver expose?
# ---------------------------------------------------------------------------
def _has(solver, name):
    return callable(getattr(solver, name, None))


def solver_name(solver):
    meta = getattr(solver, "META", None)
    if isinstance(meta, dict) and meta.get("name"):
        return str(meta["name"])
    return getattr(solver, "__name__", "solver")


def _run_solve(fn, train, test_inputs, budget):
    """Call a solve-like fn defensively. The gate passes ONLY (train, test_inputs, budget) — held-out
    test OUTPUTS are never handed to the solver. Returns (attempts, err_str_or_None)."""
    try:
        attempts = fn(train, test_inputs, budget)
        return attempts, None
    except Exception as e:  # robust to solver exceptions/timeouts, exactly like harness.evaluate
        return [], "%s: %s" % (type(e).__name__, e)


def _reset(solver):
    """Clear cross-task experience library if the solver exposes the documented hook. Returns True iff
    a reset actually happened (so callers know whether a 'cold' run is genuinely cold)."""
    if _has(solver, "reset_library"):
        try:
            solver.reset_library()
            return True
        except Exception:
            return False
    return False


# ---------------------------------------------------------------------------
# THE GATE: full solve vs invention-ablated solve, per task, ARC 2-attempt exact
# ---------------------------------------------------------------------------
def evaluate_invention(solver_module, tasks, budget, label="", log=True, verbose=False):
    """Measure INVENTION on `tasks` for one solver.

    tasks : [(task_id, train_pairs, test_pairs), ...]  (test_pairs hold held-out outputs the gate scores;
            they are NEVER passed to the solver).
    budget: per-task search/exec budget the solver must respect.

    Returns a metrics dict including:
      n               : #tasks
      solved          : #tasks the FULL solver solves (ARC 2-attempt exact on every test pair)
      ablated_solved  : #tasks the invention-DISABLED solver solves (single-whole-template retrieval)
      INVENTED        : solved - ablated_solved  (>=0 clamp) — the creativity number: solves that ONLY
                        the invention path (composition/abstraction/repurposing) produces.
      has_ablation    : whether solver exposes solve_ablated (False => ablated_solved=0, NOT certified)
      certified_invention : bool — True iff has_ablation (we can disprove retrieval) AND INVENTED>0.
      cannot_certify  : True when no solve_ablated hook exists (INVENTED reported but uncertified).
      per_task        : list of tags per task (see below).

    Per-task tag fields:
      task_id, solved, ablated_solved, frac, ablated_frac, err, ablated_err, tag
    where tag is one of:
      "INVENTED"        full solves, ablation FAILS  -> a creative solve (retrieval cannot reach it)
      "retrieved"       full solves AND ablation solves -> reachable by single-template retrieval
      "ablation_only"   ablation solves but full FAILS -> regression/instability (flagged for inspection)
      "unsolved"        neither solves
      "uncertified"     full solves but NO ablation hook -> cannot tell invention from retrieval
    """
    label = label or solver_name(solver_module)
    has_abl = _has(solver_module, "solve_ablated")
    has_reset = _has(solver_module, "reset_library")

    # Fresh library for a clean measurement (experience prior accumulates within this run only).
    _reset(solver_module)

    t0 = time.time()
    per_task = []
    solved = 0
    ablated_solved = 0
    invented = 0
    errors = 0
    ablated_errors = 0

    for i, (tid, train, test) in enumerate(tasks):
        test_inputs = [gi for gi, _ in test]

        # --- FULL solver (invention enabled). Library state persists across tasks this loop. ---
        attempts, err = _run_solve(solver_module.solve, train, test_inputs, budget)
        ok, _per, frac = score_task(attempts, test)
        if err:
            errors += 1

        # --- ABLATED solver (invention disabled): single-whole-template retrieval only. ---
        if has_abl:
            a_attempts, a_err = _run_solve(solver_module.solve_ablated, train, test_inputs, budget)
            a_ok, _aper, a_frac = score_task(a_attempts, test)
            if a_err:
                ablated_errors += 1
        else:
            a_ok, a_frac, a_err = False, 0.0, None

        solved += int(ok)
        ablated_solved += int(a_ok)

        if ok and not a_ok and has_abl:
            tag = "INVENTED"; invented += 1
        elif ok and a_ok:
            tag = "retrieved"
        elif (not ok) and a_ok:
            tag = "ablation_only"
        elif ok and not has_abl:
            tag = "uncertified"
        else:
            tag = "unsolved"

        per_task.append({
            "task_id": tid, "solved": bool(ok), "ablated_solved": bool(a_ok),
            "frac": round(frac, 3), "ablated_frac": round(a_frac, 3),
            "err": err, "ablated_err": a_err, "tag": tag,
        })
        if verbose and (i + 1) % 25 == 0:
            print("  [%d/%d] solved=%d ablated=%d INVENTED=%d (%.0fs)"
                  % (i + 1, len(tasks), solved, ablated_solved, invented, time.time() - t0), flush=True)

    n = len(tasks)
    # INVENTED counts ONLY tasks the full solver solves AND ablation does not (already accumulated when
    # has_abl). When there is no ablation hook we cannot certify any invention -> 0 + flag.
    metrics = {
        "label": label,
        "gate": "invention",
        "n": n,
        "solved": solved,
        "solve_rate": round(solved / n, 4) if n else 0.0,
        "ablated_solved": ablated_solved,
        "ablated_solve_rate": round(ablated_solved / n, 4) if n else 0.0,
        "INVENTED": invented,
        "invention_rate": round(invented / n, 4) if n else 0.0,
        "has_ablation": has_abl,
        "has_reset": has_reset,
        "certified_invention": bool(has_abl and invented > 0),
        "cannot_certify": (not has_abl),
        "errors": errors,
        "ablated_errors": ablated_errors,
        "budget": budget,
        "seconds": round(time.time() - t0, 1),
    }
    if not has_abl:
        metrics["flag"] = ("no solve_ablated hook: ablated_solved forced to 0 and INVENTED uncertified "
                           "(cannot disprove retrieval). Expose solve_ablated to certify invention.")

    result = {"metrics": metrics, "per_task": per_task}
    if log:
        _persist(result, kind="invention", label=label)
    return metrics


# ---------------------------------------------------------------------------
# TRANSFER: how many solves depend on the cross-task experience library?
# ---------------------------------------------------------------------------
def transfer_invention(solver_module, tasks, budget, label="", log=True, verbose=False):
    """Measure EXPERIENCE-TRANSFER for one solver via the documented `reset_library()` hook.

    reused = (solved with the library accumulating across the full run, warm)
             MINUS
             (solved with the cross-task library FORCED EMPTY before every task, cold).

    The cold condition resets the library BEFORE each task, so each task is solved with zero carry-over
    from prior tasks — isolating the experience prior's contribution. A positive `reused` = solves that
    exist ONLY because experience from earlier tasks transferred forward (the fast-by-prior claim).

    Returns a metrics dict:
      warm_solved, cold_solved, reused (=warm-cold, >=0 clamp), reuse_rate,
      has_reset (False => warm/cold/reused = None and a flag is set; transfer cannot be isolated),
      per_task tags ("transfer" = warm solved & cold not; "intrinsic" = both; "lost_when_cold" handled).
    """
    label = label or solver_name(solver_module)
    has_reset = _has(solver_module, "reset_library")
    n = len(tasks)

    if not has_reset:
        metrics = {
            "label": label, "gate": "transfer", "n": n,
            "warm_solved": None, "cold_solved": None, "reused": None, "reuse_rate": None,
            "has_reset": False, "budget": budget, "seconds": 0.0,
            "flag": ("no reset_library hook: cannot force the cross-task library empty, so "
                     "experience-transfer cannot be isolated. Expose reset_library() to measure it."),
        }
        result = {"metrics": metrics, "per_task": []}
        if log:
            _persist(result, kind="transfer", label=label)
        return metrics

    t0 = time.time()

    # ---- WARM pass: library accumulates across the whole run (experience prior active). ----
    _reset(solver_module)
    warm = {}
    warm_solved = 0
    for tid, train, test in tasks:
        test_inputs = [gi for gi, _ in test]
        attempts, _err = _run_solve(solver_module.solve, train, test_inputs, budget)
        ok, _per, _frac = score_task(attempts, test)
        warm[tid] = bool(ok)
        warm_solved += int(ok)

    # ---- COLD pass: reset BEFORE every task => no cross-task carry-over (library forced empty). ----
    cold = {}
    cold_solved = 0
    for tid, train, test in tasks:
        _reset(solver_module)  # forced-empty cross-task library for THIS task
        test_inputs = [gi for gi, _ in test]
        attempts, _err = _run_solve(solver_module.solve, train, test_inputs, budget)
        ok, _per, _frac = score_task(attempts, test)
        cold[tid] = bool(ok)
        cold_solved += int(ok)

    _reset(solver_module)  # leave the module clean for subsequent callers

    per_task = []
    reused = 0
    for tid, _train, _test in tasks:
        w, c = warm.get(tid, False), cold.get(tid, False)
        if w and not c:
            tag = "transfer"; reused += 1          # solved ONLY with accumulated experience
        elif w and c:
            tag = "intrinsic"                       # solvable cold; experience not required
        elif (not w) and c:
            tag = "lost_when_warm"                  # warm run regressed it (library interference; flag)
        else:
            tag = "unsolved"
        per_task.append({"task_id": tid, "warm_solved": w, "cold_solved": c, "tag": tag})

    metrics = {
        "label": label, "gate": "transfer", "n": n,
        "warm_solved": warm_solved, "cold_solved": cold_solved,
        "reused": reused, "reuse_rate": round(reused / n, 4) if n else 0.0,
        "has_reset": True,
        "lost_when_warm": sum(1 for p in per_task if p["tag"] == "lost_when_warm"),
        "budget": budget, "seconds": round(time.time() - t0, 1),
    }
    if verbose:
        print("  TRANSFER %s: warm=%d cold=%d reused=%d (%.0fs)"
              % (label, warm_solved, cold_solved, reused, metrics["seconds"]), flush=True)

    result = {"metrics": metrics, "per_task": per_task}
    if log:
        _persist(result, kind="transfer", label=label)
    return metrics


# ---------------------------------------------------------------------------
# convenience: load + score an arbitrary candidate file through both gates
# ---------------------------------------------------------------------------
def load_solver(path):
    """Import a candidate solver module from a file path (delegates to harness.load_solver)."""
    return harness.load_solver(path)


def score_candidate(path, split="arc1-eval", n=None, budget=3000, ids=None,
                    do_transfer=True, log=True, verbose=False):
    """Load a candidate solver file, run it through the INVENTION gate (and TRANSFER gate if requested)
    on a named split, persist a json log, and return a combined dict:
        {"file","solver","split","n","budget","invention": {...}, "transfer": {...}|None}.
    `split` is a harness SPLITS key (e.g. 'arc1-train','arc1-eval','arc2-train','arc2-eval')."""
    solver = load_solver(path)
    name = solver_name(solver)
    tasks = load_split(split, n=n, ids=ids)
    inv = evaluate_invention(solver, tasks, budget, label=name, log=log, verbose=verbose)
    tr = transfer_invention(solver, tasks, budget, label=name, log=log, verbose=verbose) if do_transfer else None
    combined = {
        "file": os.path.abspath(path), "solver": name, "split": split,
        "n": len(tasks), "budget": budget, "invention": inv, "transfer": tr,
    }
    if log:
        _persist(combined, kind="candidate", label=name)
    return combined


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------
def _safe(label):
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(label))[:64] or "solver"


def _persist(obj, kind, label):
    os.makedirs(RUNS, exist_ok=True)
    fn = "%s_%s_%s.json" % (kind, _safe(label), time.strftime("%Y%m%d-%H%M%S"))
    path = os.path.join(RUNS, fn)
    with open(path, "w") as f:
        json.dump(obj, f)
    obj_metrics = obj.get("metrics", obj)
    if isinstance(obj_metrics, dict):
        obj_metrics["_log"] = path
    return path


# ---------------------------------------------------------------------------
# CLI / self-validation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Usage: invention_gate.py [candidate.py] [split] [n] [budget]
    # With no args: validate on seed_solver (no ablation -> INVENTED uncertified) and on a
    # trivially-ablated param-struct (solve_ablated==solve -> INVENTED==0).
    if len(sys.argv) > 1 and sys.argv[1].endswith(".py"):
        path = sys.argv[1]
        split = sys.argv[2] if len(sys.argv) > 2 else "arc1-eval"
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 40
        budget = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
        out = score_candidate(path, split=split, n=n, budget=budget, verbose=True)
        print(json.dumps({"invention": out["invention"],
                          "transfer": out["transfer"]}, indent=2, default=str))
    else:
        import seed_solver
        tasks = load_split("arc1-train", n=20)
        m = evaluate_invention(seed_solver, tasks, 1200, label="seed_solver")
        print("SEED:", json.dumps(m, default=str))
