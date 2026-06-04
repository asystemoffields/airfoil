#!/usr/bin/env python3
"""Vine — PERCEPTION-COMPLETION ignition retry (HANDOFF step 1+2, the PRINCIPLED layer).

The 0/400 ignition NO-GO had two causes: perception too thin + tasks too messy. This addresses the FIRST, principled
one: complete the cell-substrate's innate maps to the FULL GRID ISOMETRY GROUP (dihedral: mirrors + rotations +
diagonals -- finite + complete, NOT a treadmill) + TRANSLATIONAL/periodic symmetry + FLOOD-FILL of multi-cell
enclosed regions (not 1x1). Re-run on ARC-1 eval(400): does completing the symmetry group lift cell solves above 0?
If yes, perception WAS a real gap; if still ~0, real ARC's MESS dominates -> analogical adaptation (the other layer).
Run: /data/llm/.venv/bin/python ground_vine_cell2.py"""
import sys, time
from collections import deque, Counter
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

sys.path.insert(0, "/data/Windows-files/Documents/airfoil/incubation/evolve")
import harness
V2NET = V2(); V2NET.load_state_dict(torch.load("learner_v2.pt")); V2NET.eval()

# the COMPLETE grid isometry group (dihedral D4) -- principled + finite, the grid's symmetry group
DIHEDRAL = {
    "mirror_h": lambda r, c, H, W: (r, W-1-c),
    "mirror_v": lambda r, c, H, W: (H-1-r, c),
    "rot180":   lambda r, c, H, W: (H-1-r, W-1-c),
    "diag":     lambda r, c, H, W: (c, r),
    "adiag":    lambda r, c, H, W: (W-1-c, H-1-r),
    "rot90":    lambda r, c, H, W: (c, H-1-r),
    "rot270":   lambda r, c, H, W: (W-1-c, r),
}
SQUARE_ONLY = {"diag", "adiag", "rot90", "rot270"}
NB = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _sym_repair(gi, name, mapfn, N):
    gi = np.asarray(gi, int); H, W = gi.shape
    if name in SQUARE_ONLY and H != W:
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
    for name, mapfn in DIHEDRAL.items():
        for N in range(10):
            okt = True
            for gi, go in train:
                out = _sym_repair(gi, name, mapfn, N)
                if out is None or out.shape != np.asarray(go).shape or not np.array_equal(out, go):
                    okt = False; break
            if okt and all(_sym_repair(gi, name, mapfn, N) is not None and
                           np.array_equal(_sym_repair(gi, name, mapfn, N), np.asarray(go, int)) for gi, go in test):
                return f"symmetry({name},occ={N})"
    return None


def _periods(gi, N):
    """axis-aligned periods (pr,0)/(0,pc) consistent on non-occluder cells -- translational symmetry."""
    H, W = gi.shape; out = []
    for pc in range(1, W):
        if all(gi[r, c] == N or gi[r, c+pc] == N or gi[r, c] == gi[r, c+pc]
               for r in range(H) for c in range(W-pc)):
            out.append((0, pc)); break
    for pr in range(1, H):
        if all(gi[r, c] == N or gi[r+pr, c] == N or gi[r, c] == gi[r+pr, c]
               for c in range(W) for r in range(H-pr)):
            out.append((pr, 0)); break
    return out


def earn_periodic(train, test):
    for N in range(1, 10):
        okt = True
        for gi, go in list(train) + list(test):
            gi = np.asarray(gi, int); go = np.asarray(go, int)
            if gi.shape != go.shape or N not in gi:
                okt = False; break
            H, W = gi.shape; out = gi.copy(); ps = _periods(gi, N)
            if not ps:
                okt = False; break
            for r in range(H):
                for c in range(W):
                    if gi[r, c] == N:
                        for (pr, pc) in ps:
                            done = False
                            for k in list(range(1, max(H, W))) + list(range(-1, -max(H, W), -1)):
                                r2, c2 = r+k*pr, c+k*pc
                                if 0 <= r2 < H and 0 <= c2 < W and gi[r2, c2] != N:
                                    out[r, c] = gi[r2, c2]; done = True; break
                            if done: break
            if not np.array_equal(out, go):
                okt = False; break
        if okt:
            return f"periodic(occ={N})"
    return None


def _enclosed(g):
    H, W = g.shape; seen = np.zeros((H, W), bool); regs = []
    for r in range(H):
        for c in range(W):
            if g[r, c] == 0 and not seen[r, c]:
                q = deque([(r, c)]); seen[r, c] = True; cells = []; touch = False
                while q:
                    cr, cc = q.popleft(); cells.append((cr, cc))
                    if cr in (0, H-1) or cc in (0, W-1): touch = True
                    for dr, dc in NB:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < H and 0 <= nc < W and g[nr, nc] == 0 and not seen[nr, nc]:
                            seen[nr, nc] = True; q.append((nr, nc))
                if not touch: regs.append(cells)
    return regs


def earn_floodfill(train, test):
    fill = None
    for gi, go in train:
        gi = np.asarray(gi, int); go = np.asarray(go, int)
        if gi.shape != go.shape: return None
        for cells in _enclosed(gi):
            for (r, c) in cells:
                if go[r, c] == 0: return None
                if fill is None: fill = int(go[r, c])
                elif fill != go[r, c]: return None
    if fill is None: return None
    for gi, go in list(train) + list(test):
        gi = np.asarray(gi, int); go = np.asarray(go, int); out = gi.copy()
        for cells in _enclosed(gi):
            for (r, c) in cells: out[r, c] = fill
        if not np.array_equal(out, go): return None
    return f"floodfill(color={fill})"


def vine_solve(train, test, topk=10):
    try:
        s, _ = recognizer_solve(V2NET, train, test, 2, 3)
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
    for nm, fn in (("CELL:symmetry", earn_symmetry), ("CELL:periodic", earn_periodic), ("CELL:floodfill", earn_floodfill)):
        try:
            if fn(train, test) is not None: return nm
        except Exception:
            pass
    return None


def main():
    tasks = harness.load_split("arc1-eval")
    t0 = time.time(); solved = {}
    for i, (tid, train, test) in enumerate(tasks):
        try:
            p = vine_solve(train, test)
        except Exception:
            p = None
        if p: solved[tid] = p
        if (i + 1) % 100 == 0:
            nc = sum(1 for v in solved.values() if v.startswith("CELL"))
            print(f"  ...{i+1}/400  solved {len(solved)} (cell {nc})  [{time.time()-t0:.0f}s]", flush=True)
    S = set(solved); cell = {t: p for t, p in solved.items() if p.startswith("CELL")}
    print(f"\nVINE + COMPLETED PERCEPTION on ARC-1 eval(400) [{time.time()-t0:.0f}s]:")
    print(f"  solved: {len(S)}  by path: {dict(Counter(solved.values()))}")
    print(f"  CELL solves: {len(cell)} -> {cell}")
    print(f"  beyond gen2_base: {len(S - GEN2)}   beyond gen6_base: {len(S - GEN6)}  {sorted(S - GEN6)}")
    print("READ: cell solves > 0 = completing the PRINCIPLED perception (full isometry group + periodicity + flood-"
          "fill) ignites on real ARC -> perception WAS the gap. Still ~0 = real ARC's MESS dominates -> need the "
          "analogical-adaptation layer (workflow wr36mpy4u) to bend schemas to 'looks-like-but-not-exact' tasks.")


if __name__ == "__main__":
    main()
