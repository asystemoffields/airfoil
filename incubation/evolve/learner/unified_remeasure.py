#!/usr/bin/env python3
"""Branch-B scale-prep — wire the EFFECT faculty into the solver next to the predicate faculty, then RE-MEASURE.

Unified solve = recognizer-guided recolor (earn_predicate + earn_composed, via kaggle_loop.policy_solve) OR the
motor hand (earn_effect: eye-grounded move). Confirms (1) the unified loop handles BOTH a recolor family and a MOVE
family, and (2) the real-data coverage delta on BARC vs the recolor-only baseline (18/226) -- and WHICH tasks the
move gesture adds, so we know which gesture to grow next. Run: /data/llm/.venv/bin/python unified_remeasure.py [N]"""
import sys, time
import numpy as np
from kaggle_loop import policy_solve
from effect_faculty import earn_effect, make_align_task
from grow_library import make_containment_task

sys.path.insert(0, "/data/Windows-files/Documents/airfoil/incubation/evolve")
from ground_arc import winning_relations
from ground_barc import load_barc_tasks


def unified_solve(train, test, library):
    """predicate faculty first (recolor singles+compositions), then motor hand (eye-grounded move)."""
    prog, cost = policy_solve(train, test, library)
    if prog is not None:
        return prog, "recolor"
    try:
        eff = earn_effect(train, test)
    except Exception:
        eff = None
    if eff is not None:
        return eff, "move"
    return None, None


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    t0 = time.time()

    # (1) the unified loop handles BOTH faculties -- a recolor family AND a move family
    print("UNIFIED LOOP (predicate faculty + motor hand) on two families:")
    for fam, gen in [("containment (recolor)", make_containment_task), ("align-to-anchor (MOVE)", make_align_task)]:
        ok = kind = 0; tot = 12
        for _ in range(tot):
            prog, k = unified_solve(gen(4), gen(2), [])
            ok += int(prog is not None); kind = k if prog is not None else kind
        print(f"  {fam:<26}: solved {ok}/{tot}  via '{kind}'")

    # (2) RE-MEASURE real-data coverage on BARC with effects wired in (baseline: recolor-only 18/226)
    rec = mov = tot = 0
    for tid, train, test, meta in load_barc_tasks(limit=N):
        tot += 1
        try:
            prog, kind = unified_solve(train, test, [])
        except Exception:
            continue
        if kind == "recolor": rec += 1
        elif kind == "move": mov += 1
    print(f"\nRE-MEASURE on BARC ARC-Heavy (N={N} -> {tot} tasks) [{time.time()-t0:.0f}s]:")
    print(f"  recolor (baseline): {rec}   + move (NEW from the motor hand): {mov}   = {rec+mov} total")
    print("READ: the unified solver handles both faculties in one loop; the move DELTA on real BARC shows the "
          "marginal value of ONE gesture + which family it unlocks -> grow the NEXT gesture toward the still-unsolved "
          "bulk (measure, then grow).")


if __name__ == "__main__":
    main()
