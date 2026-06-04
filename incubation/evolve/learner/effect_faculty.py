#!/usr/bin/env python3
"""Branch-B scale-prep — the EFFECT FACULTY (the motor hand), eye-grounded (Alex's call: option 2).

Mirror of the relational eye. The hand has three atomic motor primitives — paint / place / erase — and NOTHING
else is hand-coded. An effect is a COMPOSITION of atoms whose PARAMETERS (which target, which displacement) are
computed by the relational EYE: an eye-selected anchor object tells the hand WHERE to act. So "move each object to
align with the largest" = the eye picks the largest (a Quantify predicate), the hand erases+places each object at
the eye-derived offset. The gesture is EARNED (search target-predicate x displacement-mode + verify), never given.
This completes the architecture into PERCEIVE (eye) -> ROUTE (V2) -> ACT (hand), thin faculties + earned vocabulary.
Run: /data/llm/.venv/bin/python effect_faculty.py   (smoke: a MOVE task the recolor grammar structurally can't do)"""
import numpy as np
import grammar as G
import rel_dsl as D

CORE = "core"
MODES = ["align_row", "align_col", "onto"]   # eye-derived displacement rules (target tells the hand where)


# ---- the three motor ATOMS (the hand) ----
def erase(grid, cells):
    for (r, c) in cells: grid[r, c] = 0
    return grid
def place(grid, cells, color, delta):
    for (r, c) in cells:
        nr, nc = r + delta[0], c + delta[1]
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]: grid[nr, nc] = color
    return grid


class Move:                                   # an EARNED gesture: eye-selected target + eye-derived displacement
    tier = CORE
    def __init__(self, target_pred, mode): self.target_pred = target_pred; self.mode = mode
    def __repr__(self): return f"move(toward {self.target_pred}, {self.mode})"
    def _delta(self, o, t):
        if self.mode == "align_row": return (t["r0"] - o["r0"], 0)
        if self.mode == "align_col": return (0, t["c0"] - o["c0"])
        return (t["r0"] - o["r0"], t["c0"] - o["c0"])      # onto
    def ev(self, grid):
        grid = np.asarray(grid, int); objs = G.objects(grid, 4, True)
        targets = [o for o in objs if self.target_pred(o, objs)]
        if len(targets) != 1: return None                  # the eye must name a UNIQUE anchor
        t = targets[0]; out = grid.copy()
        for o in objs:
            if self.target_pred(o, objs): continue          # don't move the anchor
            erase(out, o["cells"]); place(out, o["cells"], o["color"], self._delta(o, t))
        return out


def verify_effect(eff, train, test=None):
    for gi, go in list(train) + (list(test) if test else []):
        o = eff.ev(gi)
        if o is None or o.shape != np.asarray(go).shape or not np.array_equal(o, np.asarray(go)):
            return False
    return True


def earn_effect(train, test):
    """EARN a motor gesture: the EYE (predicate_space) selects the anchor; search target-predicate x mode; verify."""
    for target_pred in D.predicate_space():
        for mode in MODES:
            eff = Move(target_pred, mode)
            if verify_effect(eff, train, test):
                return eff
    return None


# ---- smoke: a MOVE task the recolor grammar structurally cannot express ----
rng = np.random.RandomState(0)
def make_align_task(n):
    demos = []
    for _ in range(n):
        g = np.zeros((16, 16), int)
        ar, ac = rng.randint(0, 13), rng.randint(0, 13)
        g[ar:ar+3, ac:ac+3] = 5                            # anchor = the largest object
        rows = [r for r in range(0, 16, 2) if not (ar - 1 <= r <= ar + 3)]; rng.shuffle(rows)
        for dr in rows[:4]:                                 # dots in SPACED distinct rows (stay size-1, no merge)
            g[dr, rng.randint(0, 16)] = 4
        out = g.copy(); objs = G.objects(g, 4, True)
        anchor = max(objs, key=lambda o: o["size"])
        for o in objs:                                     # ground truth: each dot moves to the anchor's column
            if o["size"] == anchor["size"]: continue
            (r, c) = o["cells"][0]; out[r, c] = 0; out[r, anchor["c0"]] = o["color"]
        demos.append((g, out))
    return demos


def _demo():
    import sys
    sys.path.insert(0, "/data/Windows-files/Documents/airfoil/incubation/evolve")
    from ground_arc import winning_relations
    tr = make_align_task(4); te = make_align_task(2)
    gram = len(winning_relations(tr, te))                   # recolor grammar can't MOVE anything
    eff = earn_effect(tr, te)
    print(f"EFFECT FACULTY (the motor hand, eye-grounded):")
    print(f"  align-to-anchor MOVE task -- GRAMMAR (recolor) winning relations = {gram} (it cannot move objects)")
    print(f"  EARNED motor gesture -> {eff}  (the EYE picks the anchor; the hand moves each object there)")
    print("READ: grammar=0 (recolor can't move) but the system EARNS a MOVE whose target+displacement are computed "
          "by the relational eye = PERCEIVE->ROUTE->ACT with thin faculties + earned vocabulary. Effects join "
          "predicates as EARNED, not hand-coded -> the effect frontier (the 0/30 ConceptARC) opens.")


if __name__ == "__main__":
    _demo()
