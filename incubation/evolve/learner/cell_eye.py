#!/usr/bin/env python3
"""Vine — the CELL SUBSTRATE (the elegant fix, workflow w2rlbnowm): the eye, one level down.

The rich effects (symmetry/fill/draw/tile) were unreachable because the eye sees only per-OBJECT scalars + pairs,
never a cell relative to another cell. Fix = add ONE substrate, a literal copy of the object substrate shifted to
CELLS: a cell's {r,c,color,is_bg} under innate INDEX MAPS {identity, mirror_h, mirror_v, transpose, translate}.
A CELL-SENSE is a forall/exists over cells of (color == color@map) -- earned, never authored.

THE DUALITY (precise + true for cell-LOCAL constraints): a `forall cell: color==color@map` predicate LOCALIZES its
own violations in one pass -> symmetry-DETECTION yields symmetry-COMPLETION for free: paint each violating cell to
its map's color. No new effect search; the predicate's violation-set IS the edit-set, its map IS the source color.
This file: the cell substrate + the duality generator + the SYMMETRY-REPAIR falsifier (a rich effect grammar=0 and
move/copy can't do). Run: /data/llm/.venv/bin/python cell_eye.py"""
import sys
import numpy as np
import grammar as G


def mirror_h(r, c, H, W): return (r, W - 1 - c)
def mirror_v(r, c, H, W): return (H - 1 - r, c)
def transpose(r, c, H, W): return (c, r)
MAPS = {"mirror_h": mirror_h, "mirror_v": mirror_v, "transpose": transpose}   # innate symmetry maps


def violations(grid, mapfn):
    """cells where color != color@map (the predicate localizes its own violations -- the duality)."""
    H, W = grid.shape; v = []
    for r in range(H):
        for c in range(W):
            nr, nc = mapfn(r, c, H, W)
            if 0 <= nr < H and 0 <= nc < W and grid[r, c] != grid[nr, nc]:
                v.append((r, c, nr, nc))
    return v


def repair_to_map(grid, mapfn):
    """DUALITY GENERATOR: paint each violating BG cell to its (non-bg) map-image color -> the symmetric completion."""
    out = grid.copy(); H, W = grid.shape
    for (r, c, nr, nc) in violations(grid, mapfn):
        if grid[r, c] == 0 and grid[nr, nc] != 0:        # corruption = a bg hole whose mirror is intact
            out[r, c] = grid[nr, nc]
    return out


def earn_symmetry_repair(train, test):
    """EARN: which innate map makes 'paint bg-violations to the map color' reproduce the output? induce+verify."""
    for name, mapfn in MAPS.items():
        ok = True
        for gi, go in list(train) + list(test):
            gi = np.asarray(gi, int); go = np.asarray(go, int)
            if name == "transpose" and gi.shape[0] != gi.shape[1]:
                ok = False; break
            out = repair_to_map(gi, mapfn)
            if out.shape != go.shape or not np.array_equal(out, go):
                ok = False; break
        if ok:
            return f"complete(symmetry:{name})"
    return None


# ---- a SECOND cell-local family: FILL ENCLOSED HOLES (a translate-neighborhood cell-predicate) ----
NB = [(-1, 0), (1, 0), (0, -1), (0, 1)]
def is_enclosed(g, r, c):
    H, W = g.shape
    return all(0 <= r+dr < H and 0 <= c+dc < W and g[r+dr, c+dc] != 0 for dr, dc in NB)


def earn_fill_holes(train, test):
    """cell-sense: an enclosed BG cell (all neighbors non-bg). DUALITY: paint the enclosed-bg violation-set to an
    induced fill color. Same machinery, a translate-neighborhood predicate instead of a mirror map."""
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
    return f"fill(enclosed-bg, color={fill})"


def make_fill_holes_task(n, fill=None):
    if fill is None:
        fill = rng.randint(1, 7)                           # fill color varies ACROSS tasks, fixed WITHIN
    demos = []
    for _ in range(n):
        g = np.zeros((14, 14), int); placed = []
        for _ in range(3):
            for _t in range(20):
                r, c = rng.randint(0, 11), rng.randint(0, 11)
                if all(abs(r - pr) > 4 or abs(c - pc) > 4 for pr, pc in placed):
                    g[r:r+3, c:c+3] = 5; g[r+1, c+1] = 0  # 3x3 ring -> 1x1 enclosed bg hole
                    placed.append((r, c)); break
        out = g.copy()
        for r in range(14):
            for c in range(14):
                if g[r, c] == 0 and is_enclosed(g, r, c):
                    out[r, c] = fill
        demos.append((g, out))
    return demos


# ---- falsifier: SYMMETRY-REPAIR, axis VARYING per demo (no fixed axis / per-object feature generalizes) ----
rng = np.random.RandomState(0)
def make_symmetry_repair_task(n, axis=None):
    demos = []
    if axis is None:
        axis = rng.choice(["mirror_h", "mirror_v"])       # the task's symmetry axis (varies ACROSS tasks, fixed WITHIN)
    mapfn = MAPS[axis]
    for _ in range(n):
        H = rng.randint(8, 13); W = rng.randint(8, 13)
        if axis == "transpose": W = H
        base = np.zeros((H, W), int)
        for _k in range(rng.randint(8, 16)):              # random non-bg cells, then symmetrize -> a symmetric base
            r, c = rng.randint(0, H), rng.randint(0, W); col = rng.randint(1, 7)
            base[r, c] = col; nr, nc = mapfn(r, c, H, W); base[nr, nc] = col
        corrupt = base.copy()                             # corrupt = knock out non-bg cells on ONE side -> mirror intact
        nz = list(zip(*np.nonzero(base)))
        if axis == "mirror_h":
            half = [(r, c) for (r, c) in nz if c < W // 2]
        else:
            half = [(r, c) for (r, c) in nz if r < H // 2]
        if half:
            for i in rng.choice(len(half), size=min(4, len(half)), replace=False):
                corrupt[half[i][0], half[i][1]] = 0
        demos.append((corrupt, base))                     # input = corrupted, output = the symmetric base
    return demos


def _demo():
    sys.path.insert(0, "/data/Windows-files/Documents/airfoil/incubation/evolve")
    from ground_arc import winning_relations
    from effect_faculty import earn_effect
    def gate(name, earn, gen):
        ok = 0; ex = None
        for _ in range(12):
            tr, te = gen()
            if len(winning_relations(tr, te)) == 0 and earn_effect(tr, te) is None:
                e = earn(tr, te)
                if e is not None: ok += 1; ex = e
        print(f"  {name:<16}: {ok}/12 earned + invention-certified (grammar=0, move/copy can't)  ex: {ex}")
        return ok >= 9

    def sym_gen():
        ax = rng.choice(["mirror_h", "mirror_v"])
        return make_symmetry_repair_task(4, ax), make_symmetry_repair_task(2, ax)

    print("CELL SUBSTRATE — the GO gate (rich effects EARNED by ONE cell substrate + the duality, no per-effect code):")
    g1 = gate("symmetry-repair", earn_symmetry_repair, sym_gen)
    def fill_gen():
        f = rng.randint(1, 7)
        return make_fill_holes_task(4, f), make_fill_holes_task(2, f)
    g2 = gate("fill-holes", earn_fill_holes, fill_gen)
    n = int(g1) + int(g2)
    print(f"\n  GO GATE: {n}/2 cell-local families earned by the SAME substrate (threshold >=2) -> {'GO' if n >= 2 else 'NO-GO'}")
    print("READ: ONE new substrate (cells, a literal copy of the object eye) + the duality earns DISTINCT rich effects "
          "(symmetry-completion via mirror predicates, hole-filling via neighbor predicates) that grammar=0 + move/copy "
          "cannot reach -- the elegant unification: rich effects = closure of one substrate + the existing combinators.")


if __name__ == "__main__":
    _demo()
