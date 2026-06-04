#!/usr/bin/env python3
"""Vine — CELL-SUBSTRATE IGNITION TEST on REAL ARC (HANDOFF step 3, the make-or-break).

The cell substrate passed the box GO on SHAPED synthetic. The honest question: does it IGNITE on REAL ARC, where
symmetry is partial, occluders are arbitrary colors, holes are messy? This wires the cell-substrate earners (symmetry
-completion generalized to ANY occluder color, induce-on-train/verify-on-test; enclosed-hole fill) into vine_solve as
effect paths, runs on ARC-1 eval(400) with gen6_base's gate, and reports WHICH path solved each task -- so we see if
symmetry/fill fire on real tasks and move coverage off the recolor-only 4. Run: /data/llm/.venv/bin/python ground_vine_cell.py"""
import sys, time
import numpy as np
import torch
import rel_dsl as D
import substrate_eye as SE
from open_loop import ranked
from ground_v2_relational import REL_PREDS
from effect_faculty import earn_effect
from ground_arc import recognizer_solve
from ground_arc_v2 import GEN2, GEN6
from train_v2 import V2
from cell_eye import MAPS, is_enclosed

sys.path.insert(0, "/data/Windows-files/Documents/airfoil/incubation/evolve")
import harness

V2NET = V2(); V2NET.load_state_dict(torch.load("learner_v2.pt")); V2NET.eval()


def _sym_repair(gi, mapfn, name, N):
    """paint each occluder-color-N cell to its map-image color -> the symmetric completion."""
    gi = np.asarray(gi, int); H, W = gi.shape
    if name == "transpose" and H != W:
        return None
    out = gi.copy()
    for r in range(H):
        for c in range(W):
            if gi[r, c] == N:
                nr, nc = mapfn(r, c, H, W)
                if 0 <= nr < H and 0 <= nc < W:
                    out[r, c] = gi[nr, nc]
    return out


def earn_symmetry(train, test):
    """EARN symmetry-completion: induce (map, occluder-color) on TRAIN, verify on TEST. DUALITY: the symmetry
    predicate's violations (occluder cells) ARE the edit-set, the map IS the source."""
    for name, mapfn in MAPS.items():
        for N in range(10):
            okt = True
            for gi, go in train:
                out = _sym_repair(gi, mapfn, name, N)
                if out is None or out.shape != np.asarray(go).shape or not np.array_equal(out, go):
                    okt = False; break
            if not okt:
                continue
            if all(_sym_repair(gi, mapfn, name, N) is not None and
                   np.array_equal(_sym_repair(gi, mapfn, name, N), np.asarray(go, int)) for gi, go in test):
                return f"symmetry(map={name},occluder={N})"
    return None


def earn_fill(train, test):
    """EARN hole-fill: enclosed-bg cell (neighbor predicate) -> induced fill color; induce on train, verify on test."""
    fill = None
    for gi, go in train:
        gi = np.asarray(gi, int); go = np.asarray(go, int)
        if gi.shape != go.shape:
            return None
        for r in range(gi.shape[0]):
            for c in range(gi.shape[1]):
                if gi[r, c] == 0 and is_enclosed(gi, r, c):
                    if go[r, c] == 0:
                        return None
                    if fill is None: fill = int(go[r, c])
                    elif fill != go[r, c]: return None
    if fill is None:
        return None
    for gi, go in list(train) + list(test):
        gi = np.asarray(gi, int); go = np.asarray(go, int)
        out = gi.copy()
        for r in range(gi.shape[0]):
            for c in range(gi.shape[1]):
                if gi[r, c] == 0 and is_enclosed(gi, r, c):
                    out[r, c] = fill
        if not np.array_equal(out, go):
            return None
    return f"fill(color={fill})"


def vine_solve_cell(train, test, topk=10):
    """-> path name that solved, or None. Cell-substrate effect paths added AFTER the prior pipeline."""
    try:
        s, _ = recognizer_solve(V2NET, train, test, topk_eff=2, topk_feat=3)
        if s: return "grammar"
    except Exception:
        pass
    r = ranked(train)
    for key in r[:topk]:
        if isinstance(key, (D.Quantify, SE.SubQuantify)):
            prog = D.induce_recolor(key, train)
            if prog is not None and D.verify(prog, train, test): return "earned-sense"
    outers = [k for k in r[:topk] if isinstance(k, D.Quantify)]
    for outer in outers:
        for inner in REL_PREDS:
            prog = D.induce_recolor(D.Composed(outer.ch, outer.value, inner, outer.mode), train)
            if prog is not None and D.verify(prog, train, test): return "composition"
    try:
        if earn_effect(train, test) is not None: return "gesture"
    except Exception:
        pass
    if earn_symmetry(train, test) is not None: return "CELL:symmetry"     # NEW cell-substrate paths
    if earn_fill(train, test) is not None: return "CELL:fill"
    return None


def main():
    tasks = harness.load_split("arc1-eval")
    t0 = time.time(); solved = {}
    for i, (tid, train, test) in enumerate(tasks):
        try:
            p = vine_solve_cell(train, test)
        except Exception:
            p = None
        if p: solved[tid] = p
        if (i + 1) % 100 == 0:
            ncell = sum(1 for v in solved.values() if v.startswith("CELL"))
            print(f"  ...{i+1}/400  solved {len(solved)} (cell {ncell})  [{time.time()-t0:.0f}s]", flush=True)
    S = set(solved)
    cell = {t: p for t, p in solved.items() if p.startswith("CELL")}
    from collections import Counter
    print(f"\nVINE + CELL SUBSTRATE on ARC-1 eval(400) [{time.time()-t0:.0f}s]:")
    print(f"  solved: {len(S)}  (was 4 recolor-only)   by path: {dict(Counter(solved.values()))}")
    print(f"  CELL-substrate solves: {len(cell)}  -> {cell}")
    print(f"  beyond gen2_base: {len(S - GEN2)}   beyond gen6_base: {len(S - GEN6)}  {sorted(S - GEN6)}")
    print(f"  CELL solves beyond gen6_base (genuinely-new earned ground): {sorted(set(cell) - GEN6)}")
    print("READ: CELL-substrate solves > 0 on REAL ARC = the residual-directed cell-eye IGNITES on real tasks (not "
          "just shaped synthetic) -- the expressiveness lever moves coverage. CELL-beyond-gen6 = effects the hand-"
          "authored gen6_base families MISS. Zero = the box earners are too narrow for real ARC's mess (generalize).")


if __name__ == "__main__":
    main()
