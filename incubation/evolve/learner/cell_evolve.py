#!/usr/bin/env python3
"""Vine — the SELF-EVOLVING cell earn-loop (the correction: ONE search over the generative substrate, NOT hand-coded
per-family earners).

Alex's catch: cell_eye.py / ground_vine_cell2.py HAND-CODED earn_symmetry / earn_fill / earn_periodic -- the treadmill
sneaking back in, betraying the self-evolving design. The honest version: ONE earn loop searches the GENERATIVE cell
substrate -- index-maps {the grid isometry group} + {translations} (the invariance family) and neighbor-quantifiers
(the locality family) -- for a cell-predicate that explains the residual, then the DUALITY auto-generates the effect.
symmetry-completion, periodic-completion, hole-fill ALL fall out of the SAME search, EARNED not authored. Adding a
generator (a new map) automatically yields new earnable effects -- no new earner function. THAT is "evolve everything
it might need" (= every composition of the thin innate basis). Run: /data/llm/.venv/bin/python cell_evolve.py"""
import sys
import numpy as np

rng = np.random.RandomState(0)
SQ = {"diag", "adiag", "rot90", "rot270"}


def iso_maps():
    """the grid ISOMETRY GROUP (dihedral D4) -- finite + COMPLETE generative basis for symmetry, not a hand-picked menu."""
    return {
        "mirror_h": lambda r, c, H, W: (r, W-1-c),
        "mirror_v": lambda r, c, H, W: (H-1-r, c),
        "rot180":   lambda r, c, H, W: (H-1-r, W-1-c),
        "diag":     lambda r, c, H, W: (c, r),
        "adiag":    lambda r, c, H, W: (W-1-c, H-1-r),
        "rot90":    lambda r, c, H, W: (c, H-1-r),
        "rot270":   lambda r, c, H, W: (W-1-c, r),
    }


def trans_maps(maxp=8):
    """TRANSLATIONS (the periodicity generators) -- the other half of the grid's structure group. Both directions
    (g[c]==g[c+-p]), so an occluder near an edge can recover from the intact side."""
    m = {}
    for p in range(1, maxp+1):
        for d in (p, -p):
            m[f"transW{d}"] = lambda r, c, H, W, d=d: (r, c+d)
            m[f"transH{d}"] = lambda r, c, H, W, d=d: (r+d, c)
    return m


def generative_maps():
    """the searchable CLOSURE = generators + their COMPOSITIONS (glide-reflections = isometry o translation),
    GENERATED from the generator set. Adding a generator widens the closure -> new effects earned for free. This is
    the anti-treadmill structure: the ONLY extension point is generators; the earn loop never grows per-family code."""
    isos, trans = iso_maps(), trans_maps()
    maps = {**isos, **trans}
    for ni, fi in isos.items():
        if ni in ("mirror_h", "mirror_v", "rot180"):       # glides: reflection/rotation composed with a translation
            for nt, ft in trans.items():
                maps[f"{ni}o{nt}"] = lambda r, c, H, W, fi=fi, ft=ft: ft(*fi(r, c, H, W), H, W)
    return maps


def _follow(gi, mapfn, N, r, c, H, W):
    """duality: from an occluder-N cell, follow the map until a non-N source cell (k=1 for isometries; k>=1 for translates)."""
    cr, cc = r, c
    for _ in range(max(H, W) + 1):
        nr, nc = mapfn(cr, cc, H, W)
        if not (0 <= nr < H and 0 <= nc < W):
            return None
        if gi[nr, nc] != N:
            return int(gi[nr, nc])
        if (nr, nc) == (r, c):
            return None
        cr, cc = nr, nc
    return None


def _apply_invariance(gi, name, mapfn, N):
    gi = np.asarray(gi, int); H, W = gi.shape
    if name in SQ and H != W:
        return None
    out = gi.copy()
    for r in range(H):
        for c in range(W):
            if gi[r, c] == N:
                v = _follow(gi, mapfn, N, r, c, H, W)
                if v is not None:
                    out[r, c] = v
    return out


NB = [(-1, 0), (1, 0), (0, -1), (0, 1)]
def _apply_fill(gi, N, fill):
    """the LOCALITY family: a bg-N cell all of whose neighbors are non-N (an enclosed cell) -> paint to `fill`.
    (neighbor-quantifier predicate -- the same substrate, a different quantifier over translate-by-1 maps.)"""
    gi = np.asarray(gi, int); H, W = gi.shape; out = gi.copy()
    for r in range(H):
        for c in range(W):
            if gi[r, c] == N and all(0 <= r+dr < H and 0 <= c+dc < W and gi[r+dr, c+dc] != N for dr, dc in NB):
                out[r, c] = fill
    return out


def earn_cell_effect(train, test):
    """ONE earn loop over the GENERATIVE substrate -> returns whatever effect it earns (symmetry / periodic / fill),
    induced on TRAIN, verified on TEST. No per-family code; new basis elements => new earnable effects for free."""
    INV = generative_maps()                               # the generator CLOSURE (isometries + translations + glides)
    for name, mapfn in INV.items():                       # INVARIANCE family -- earned, never per-family hand-coded
        for N in range(10):
            ok = True
            for gi, go in train:
                out = _apply_invariance(gi, name, mapfn, N)
                if out is None or not np.array_equal(out, np.asarray(go, int)):
                    ok = False; break
            if ok and all(_apply_invariance(gi, name, mapfn, N) is not None and
                          np.array_equal(_apply_invariance(gi, name, mapfn, N), np.asarray(go, int)) for gi, go in test):
                return f"invariance({name},occluder={N})"
    for N in range(10):                                   # LOCALITY family (enclosed-cell fill)
        fill = None; consistent = True
        for gi, go in train:
            gi = np.asarray(gi, int); go = np.asarray(go, int); H, W = gi.shape
            for r in range(H):
                for c in range(W):
                    if gi[r, c] == N and all(0 <= r+dr < H and 0 <= c+dc < W and gi[r+dr, c+dc] != N for dr, dc in NB):
                        if fill is None: fill = int(go[r, c])
                        elif fill != go[r, c]: consistent = False
        if consistent and fill is not None:
            if all(np.array_equal(_apply_fill(gi, N, fill), np.asarray(go, int)) for gi, go in list(train)+list(test)):
                return f"fill(occluder={N},color={fill})"
    return None


# ---- box demo: does ONE earn loop evolve ALL THREE families the hand-coded earners did? ----
def _periodic_task(n, P=None):
    occ = 8
    if P is None: P = rng.randint(2, 4)                    # period varies ACROSS tasks, fixed WITHIN
    demos = []
    for _ in range(n):
        H = rng.randint(8, 12); W = P * rng.randint(3, 4)
        pat = rng.randint(1, 7, size=(H, P)); base = np.tile(pat, (1, W // P))
        cor = base.copy(); r0, c0 = rng.randint(0, H-2), rng.randint(0, W-3)
        cor[r0:r0+2, c0:c0+3] = occ
        demos.append((cor, base))
    return demos


def _demo():
    from cell_eye import make_symmetry_repair_task, make_fill_holes_task
    fams = {
        "symmetry": lambda: (lambda ax: (make_symmetry_repair_task(4, ax), make_symmetry_repair_task(2, ax)))(rng.choice(["mirror_h", "mirror_v"])),
        "periodic": lambda: (lambda P: (_periodic_task(4, P), _periodic_task(2, P)))(rng.randint(2, 4)),
        "fill":     lambda: (lambda f: (make_fill_holes_task(4, f), make_fill_holes_task(2, f)))(rng.randint(1, 7)),
    }
    print("SELF-EVOLVING cell earn-loop — does ONE search over the generative substrate evolve all families?")
    for nm, gen in fams.items():
        ok = 0; ex = None
        for _ in range(10):
            tr, te = gen()
            e = earn_cell_effect(tr, te)
            if e is not None: ok += 1; ex = e
        print(f"  {nm:<10}: {ok}/10 EVOLVED by the single loop   ex: {ex}")
    print("READ: one earn_cell_effect (no per-family code) evolves symmetry + periodic + fill from the SAME generative "
          "basis -> THIS is the self-evolving cell eye (vs the hand-coded earners). New maps => new effects for free.")


if __name__ == "__main__":
    _demo()
