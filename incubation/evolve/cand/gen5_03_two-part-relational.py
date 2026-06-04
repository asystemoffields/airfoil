#!/usr/bin/env python3
"""GEN-5 RELATION-INDUCER #3 — TWO-PART / RELATIONAL-PAIR granularity.

THE FRONTIER (CAMPAIGN.md gen-5 DIAGNOSTIC). On real ARC's held-out frontier, composing a FIXED
verb alphabet doesn't engage the tasks at all (of the 366 eval tasks gen2_base misses, ZERO have any
train-consistent program at depth 1-4 in a relational alphabet). EVERY beyond-retrieval win came from
per-task FITTED cause->effect RELATIONS. So the lever is RELATIONS, made systematic and rich — NOT
deeper search over verbs.

THIS MODULE'S GRANULARITY = TWO-PART / RELATIONAL-PAIR — relations BETWEEN parts (the multi-object
frontier). It is a SYSTEMATIC per-task RELATION-INDUCTION engine:
  (1) DECOMPOSE the grid into parts at this granularity: single 'marker' cells, multi-cell objects,
      and gridline-separated PANELS.
  (2) Extract a RICH FEATURE VECTOR per part (the candidate CAUSES): for marker pairs — same-color,
      shared-row, shared-col, on-diagonal, gap-distance, who-is-between; for a key object vs others —
      color, size, hole-count, border-touch, aspect; for panels — which-panel-by-property, and a
      per-cell boolean across panels with a FITTED output-color map.
  (3) INDUCE a feature(s)->EFFECT mapping that holds CONSISTENTLY across ALL train pairs (the
      cross-pair invariance that licenses CAUSAL induction) and EXACT-verify on train. The held-out
      test is the intervention that confirms it.
  (4) A learned FEATURE-RELEVANCE PRIOR (a tiny logistic ranker trained at import on self-generated
      synthetic relational tasks, <90s) ranks WHICH relation-inducers likely drive THIS task's effect,
      so induction stays tractable as the feature space grows.

The induced relations are ones gen2_base's FIXED menu cannot express, e.g.:
  * marker-cell -> stamp a learned template there, KEYED by the marker's color (object-to-marker copy);
  * aligned same-color marker PAIRS -> fill the between-segment with a FITTED color (not the marker's),
    keeping the endpoints (relational connect beyond connect-collinear-same-color);
  * a distinguished KEY object's property -> a recolor/transform applied to all OTHER objects;
  * gridline PANELS compared cell-by-cell with a fitted boolean op + a fitted color map, OR a single
    panel SELECTED by a property (most/least cells, unique, symmetric).

STANDARDIZED GATE (non-negotiable):
    solve_ablated == gen2_base.solve (imported verbatim — the strong retrieval ablation).
    solve         == gen2_base as attempt-1 backstop, THEN this relation-induction as attempt-2.
    invention_gate's INVENTED = solves(full) - solves(ablated) = solves gen2_base MISSES.
Develop/tune ONLY on arc1-train; arc1-eval is held-out — reported, never tuned to.

INTEGRITY. solve()/solve_ablated() learn ONLY from (a) the current task's train pairs, (b) module
state from prior solve() calls this run, (c) self-generated synthetic data at import. NEVER read ARC
task files or test OUTPUTS, no network, no LLM. Budget-respected. Pure python+numpy. Build-time light.
Run/imported with /data/llm/.venv/bin/python from .../incubation/evolve."""
import os
import sys
import time
from collections import deque, Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EVOLVE = os.path.dirname(HERE)
ARC = "/data/Windows-files/Documents/airfoil/incubation/arc"
for p in (EVOLVE, ARC, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

# The strong retrieval baseline IS the standardized ablation. Imported verbatim.
import gen2_base as BASE

META = {"name": "gen5_03_two-part-relational",
        "desc": "gen2_base retrieval backstop (attempt 1) + TWO-PART/RELATIONAL-PAIR relation-induction "
                "(attempt 2): decompose into marker-cells / objects / panels; extract rich cross-part "
                "feature vectors (same-color, aligned, gap, hole-count, border, aspect, panel-property); "
                "induce a feature->effect relation consistent across all train pairs (marker-keyed stamp, "
                "aligned-pair fitted-fill connect, key-object->others, panel compare/select) and "
                "exact-verify; a learned feature-relevance prior ranks which inducers to try. "
                "INVENTED = solves gen2_base cannot."}


# ===========================================================================
# grid helpers
# ===========================================================================
def _eq(a, b):
    return a is not None and getattr(a, "shape", None) == getattr(b, "shape", None) and np.array_equal(a, b)


def _bg(g):
    v, ct = np.unique(g, return_counts=True)
    return int(v[ct.argmax()])


def _bg_train(train):
    c = Counter()
    for gi, _ in train:
        v, ct = np.unique(gi, return_counts=True)
        for vi, ci in zip(v, ct):
            c[int(vi)] += int(ci)
    return c.most_common(1)[0][0] if c else 0


def _components(g, bg=0, diag=False, by_color=False):
    h, w = g.shape
    seen = np.zeros((h, w), bool)
    nb = ([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
          if diag else [(-1, 0), (1, 0), (0, -1), (0, 1)])
    comps = []
    for i in range(h):
        for j in range(w):
            if g[i, j] != bg and not seen[i, j]:
                c0 = g[i, j]
                cells = []
                q = deque([(i, j)])
                seen[i, j] = True
                while q:
                    a, b = q.popleft()
                    cells.append((a, b))
                    for di, dj in nb:
                        x, y = a + di, b + dj
                        if 0 <= x < h and 0 <= y < w and g[x, y] != bg and not seen[x, y]:
                            if (not by_color) or g[x, y] == c0:
                                seen[x, y] = True
                                q.append((x, y))
                comps.append(cells)
    return comps


def _bbox(cells):
    rs = [a for a, _ in cells]
    cs = [b for _, b in cells]
    return min(rs), max(rs), min(cs), max(cs)


def _verify(fn, train):
    for gi, go in train:
        try:
            o = fn(gi)
        except Exception:
            return False
        if not _eq(o, go):
            return False
    return True


def _holes_of(g, cells, bg):
    """Count enclosed bg regions inside an object's bbox (a per-object feature)."""
    r0, r1, c0, c1 = _bbox(cells)
    H, W = r1 - r0 + 1, c1 - c0 + 1
    sub = np.ones((H, W), int)
    for a, b in cells:
        sub[a - r0, b - c0] = 0
    reach = np.zeros((H, W), bool)
    q = deque()
    for i in range(H):
        for j in (0, W - 1):
            if sub[i, j] == 1 and not reach[i, j]:
                reach[i, j] = True
                q.append((i, j))
    for j in range(W):
        for i in (0, H - 1):
            if sub[i, j] == 1 and not reach[i, j]:
                reach[i, j] = True
                q.append((i, j))
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            x, y = i + di, j + dj
            if 0 <= x < H and 0 <= y < W and sub[x, y] == 1 and not reach[x, y]:
                reach[x, y] = True
                q.append((x, y))
    enclosed = (sub == 1) & ~reach
    seen = np.zeros((H, W), bool)
    cnt = 0
    for i in range(H):
        for j in range(W):
            if enclosed[i, j] and not seen[i, j]:
                cnt += 1
                q = deque([(i, j)])
                seen[i, j] = True
                while q:
                    a, b = q.popleft()
                    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        x, y = a + di, b + dj
                        if 0 <= x < H and 0 <= y < W and enclosed[x, y] and not seen[x, y]:
                            seen[x, y] = True
                            q.append((x, y))
    return cnt


# ===========================================================================
# RELATION-INDUCER 1 — MARKER-KEYED STAMP (object-to-marker copy).
# Decompose into single 'marker' cells. The CAUSE = the marker's color (key). The EFFECT = a learned
# stamp template (a relative offset->color footprint) placed at the marker. Templates are induced
# per-color from train (cross-pair invariance: the same color always stamps the same footprint). The
# marker may be kept or overwritten (fitted). Output cells outside any stamp = input.
# ===========================================================================
def fit_marker_stamp(train):
    if any(gi.shape != go.shape for gi, go in train):
        return None
    bg = _bg_train(train)
    # markers = single isolated non-bg cells (8-conn singletons)
    # For each train pair: find singleton markers in input; the output differs only inside small
    # neighborhoods around markers. Learn, per marker-color, the relative footprint of changes.
    # Support radius up to 4 (stamps can be sizeable, e.g. 0ca9ddb6).
    R = 6

    def markers_of(g):
        comps = _components(g, bg=bg, diag=True, by_color=False)
        singles = [c[0] for c in comps if len(c) == 1]
        return singles

    # A stamp paints a set of RELATIVE NON-BG cells around the marker (never paints bg). To learn the
    # footprint WITHOUT contamination from neighbouring stamps, we assign each non-bg OUTPUT cell to its
    # NEAREST marker (Chebyshev/L-inf distance; ties dropped as ambiguous), then collect, per marker-
    # color, the offsets of its own assigned cells. Cross-marker invariance: every marker of a color
    # must induce the SAME footprint, else that color is inconsistent (induction fails). This recovers
    # the small invariant stamp even for colors seen only once.
    per_color = defaultdict(list)        # color -> list of frozenset({(dr,dc): ov})
    for gi, go in train:
        h, w = gi.shape
        singles = markers_of(gi)
        if not singles:
            return None
        feet = {idx: {} for idx in range(len(singles))}
        for a in range(h):
            for b in range(w):
                ov = int(go[a, b])
                if ov == bg:
                    continue
                # nearest marker by L-inf
                best = None
                bestd = None
                tie = False
                for idx, (mr, mc) in enumerate(singles):
                    d = max(abs(a - mr), abs(b - mc))
                    if bestd is None or d < bestd:
                        bestd = d
                        best = idx
                        tie = False
                    elif d == bestd:
                        tie = True
                if best is None or tie or bestd > R:
                    continue
                mr, mc = singles[best]
                feet[best][(a - mr, b - mc)] = ov
        for idx, (mr, mc) in enumerate(singles):
            col = int(gi[mr, mc])
            per_color[col].append(feet[idx])
    # Footprint per color = UNION of observed non-bg offsets across all markers (handles edge-clipping,
    # where some examples see a truncated stamp). Conflict guard: where two observations both cover an
    # offset, they must agree on the color, else the color is inconsistent (induction fails). Final fn
    # is exact-verified on train, discarding any over-generalised footprint.
    footprints = {}
    consistent = True
    for col, lst in per_color.items():
        merged = {}
        for feet in lst:
            for off, ov in feet.items():
                if off in merged:
                    if merged[off] != ov:
                        consistent = False
                        break
                else:
                    merged[off] = ov
            if not consistent:
                break
        if not consistent:
            break
        if merged:
            footprints[col] = merged
    if not consistent or not footprints:
        return None

    def fn(g):
        h, w = g.shape
        out = g.copy()
        comps = _components(g, bg=bg, diag=True, by_color=False)
        singles = [c[0] for c in comps if len(c) == 1]
        if not singles:
            return None
        for (mr, mc) in singles:
            col = int(g[mr, mc])
            foot = footprints.get(col)
            if foot is None:
                return None
            for (dr, dc), ov in foot.items():
                a, b = mr + dr, mc + dc
                if 0 <= a < h and 0 <= b < w:
                    out[a, b] = ov
        return out

    if _verify(fn, train):
        return fn
    return None


# ===========================================================================
# RELATION-INDUCER 2 — ALIGNED-PAIR CONNECT WITH FITTED FILL.
# Decompose into single 'marker' cells. The relational CAUSE = a PAIR of same-color markers sharing a
# row or column (the alignment relation). The EFFECT = fill the strictly-between segment with a FITTED
# color (constant across train, may differ from the markers' color), endpoints kept. Generalizes
# gen2_base's connect_dots (which fills with the marker color and joins ALL collinear, no gap color).
# Also supports the variant where the fill color depends on the pair color (per-color fill map).
# ===========================================================================
def fit_aligned_pair_fill(train):
    if any(gi.shape != go.shape for gi, go in train):
        return None
    bg = _bg_train(train)

    def pairs_of(g):
        """yield (color, (r0,c0), (r1,c1), 'row'|'col') for same-color marker pairs that share a line
        with NOTHING of any non-bg color strictly between (a clean 2-endpoint relation)."""
        out = []
        for c in np.unique(g):
            if c == bg:
                continue
            pts = [tuple(p) for p in np.argwhere(g == c)]
            # group by row and by col; only act on exact pairs in a line (others left to other inducers)
            byr = defaultdict(list)
            byc = defaultdict(list)
            for (r, cc) in pts:
                byr[r].append(cc)
                byc[cc].append(r)
            for r, cols in byr.items():
                cols = sorted(cols)
                for a, b in zip(cols, cols[1:]):
                    out.append((int(c), (r, a), (r, b), "row"))
            for cc, rows in byc.items():
                rows = sorted(rows)
                for a, b in zip(rows, rows[1:]):
                    out.append((int(c), (a, cc), (b, cc), "col"))
        return out

    # Induce the fill color: for each train pair, every between-cell of a detected pair must take a
    # single consistent value. Two hypotheses: GLOBAL constant fill color, or PER-PAIR-COLOR fill map.
    global_fill = set()
    permap = {}          # pair-color -> fill color
    ok_global = True
    ok_permap = True
    saw_any = False
    for gi, go in train:
        prs = pairs_of(gi)
        # an input with no aligned pair is simply identity here (no fill) — that is allowed; it just
        # contributes no evidence about the fill color. Only require SOME pair across the whole train.
        for (c, p0, p1, kind) in prs:
            r0, c0 = p0
            r1, c1 = p1
            cells = []
            if kind == "row":
                for cc in range(min(c0, c1) + 1, max(c0, c1)):
                    cells.append((r0, cc))
            else:
                for rr in range(min(r0, r1) + 1, max(r0, r1)):
                    cells.append((rr, c0))
            if not cells:
                continue
            saw_any = True
            vals = {int(go[a, b]) for a, b in cells}
            if len(vals) != 1:
                ok_global = ok_permap = False
                break
            fv = vals.pop()
            global_fill.add(fv)
            if c in permap and permap[c] != fv:
                ok_permap = False
            permap[c] = fv
        if not (ok_global or ok_permap):
            break
    if not saw_any:
        return None

    candidates = []
    if ok_global and len(global_fill) == 1:
        candidates.append(("global", global_fill.copy().pop(), None))
    if ok_permap and permap:
        candidates.append(("permap", None, dict(permap)))

    for tag, gfill, pmap in candidates:
        def fn(g, tag=tag, gfill=gfill, pmap=pmap):
            out = g.copy()
            for (c, p0, p1, kind) in pairs_of(g):
                if tag == "global":
                    fv = gfill
                else:
                    if c not in pmap:
                        return None
                    fv = pmap[c]
                r0, c0 = p0
                r1, c1 = p1
                if kind == "row":
                    for cc in range(min(c0, c1) + 1, max(c0, c1)):
                        out[r0, cc] = fv
                else:
                    for rr in range(min(r0, r1) + 1, max(r0, r1)):
                        out[rr, c0] = fv
            return out
        if _verify(fn, train):
            return fn
    return None


# ===========================================================================
# RELATION-INDUCER 3 — KEY-OBJECT PROPERTY -> APPLIED TO OTHERS.
# Decompose into multi-cell objects. ONE object is the KEY (distinguished by a relation: it is the
# unique color, or the unique size, or the only one with holes, ...). The EFFECT applies that key's
# property (its color) to all OTHER objects (recolor them to the key's color), or recolors the key to
# a fitted color. Cross-pair invariance: the SAME selection-relation + SAME effect across all pairs.
# ===========================================================================
def _objprops(g, cells, bg):
    r0, r1, c0, c1 = _bbox(cells)
    col = Counter(int(g[a, b]) for a, b in cells).most_common(1)[0][0]
    return {
        "color": col,
        "size": len(cells),
        "holes": _holes_of(g, cells, bg),
        "h": r1 - r0 + 1,
        "w": c1 - c0 + 1,
        "border": int(r0 == 0 or c0 == 0 or r1 == g.shape[0] - 1 or c1 == g.shape[1] - 1),
    }


def fit_key_object_to_others(train):
    if any(gi.shape != go.shape for gi, go in train):
        return None
    bg = _bg_train(train)

    selectors = [
        ("max_size", lambda ps: _argunique(ps, lambda p: p["size"], want="max")),
        ("min_size", lambda ps: _argunique(ps, lambda p: p["size"], want="min")),
        ("most_holes", lambda ps: _argunique(ps, lambda p: p["holes"], want="max")),
        ("unique_color", lambda ps: _unique_by(ps, "color")),
        ("unique_size", lambda ps: _unique_by(ps, "size")),
    ]
    for diag in (False, True):
        for sname, sel in selectors:
            # EFFECT hypothesis A: recolor all OTHERS to the key's color (key kept).
            # EFFECT hypothesis B: recolor all OTHERS to a fitted constant; key kept.
            others_to_keycolor = True
            others_const = set()
            ok = True
            for gi, go in train:
                if np.any((gi != bg) != (go != bg)):
                    ok = False
                    break
                comps = _components(gi, bg=bg, diag=diag)
                if len(comps) < 2:
                    ok = False
                    break
                ps = [_objprops(gi, c, bg) for c in comps]
                ki = sel(ps)
                if ki is None:
                    ok = False
                    break
                keycol = ps[ki]["color"]
                for idx, cells in enumerate(comps):
                    ocs = {int(go[a, b]) for a, b in cells}
                    if len(ocs) != 1:
                        ok = False
                        break
                    oc = ocs.pop()
                    if idx == ki:
                        if oc != ps[idx]["color"]:
                            # key changed -> this simple hypothesis fails
                            ok = False
                            break
                    else:
                        if oc != keycol:
                            others_to_keycolor = False
                        others_const.add(oc)
                if not ok:
                    break
            if not ok:
                continue
            use = None
            if others_to_keycolor:
                use = ("keycolor", None)
            elif len(others_const) == 1:
                use = ("const", others_const.copy().pop())
            if use is None:
                continue

            def fn(g, sel=sel, diag=diag, use=use):
                comps = _components(g, bg=bg, diag=diag)
                if len(comps) < 2:
                    return None
                ps = [_objprops(g, c, bg) for c in comps]
                ki = sel(ps)
                if ki is None:
                    return None
                keycol = ps[ki]["color"]
                out = g.copy()
                for idx, cells in enumerate(comps):
                    if idx == ki:
                        continue
                    nc = keycol if use[0] == "keycolor" else use[1]
                    for a, b in cells:
                        out[a, b] = nc
                return out
            if _verify(fn, train):
                return fn
    return None


def _argunique(ps, keyfn, want="max"):
    """Index of the single object with the strict max/min key; None if tied."""
    if not ps:
        return None
    vals = [keyfn(p) for p in ps]
    target = max(vals) if want == "max" else min(vals)
    idxs = [i for i, v in enumerate(vals) if v == target]
    return idxs[0] if len(idxs) == 1 else None


def _unique_by(ps, attr):
    """Index of the single object whose attr value is unique among all objects; None if not exactly 1."""
    vals = [p[attr] for p in ps]
    cnt = Counter(vals)
    uniq = [i for i, v in enumerate(vals) if cnt[v] == 1]
    return uniq[0] if len(uniq) == 1 else None


# ===========================================================================
# RELATION-INDUCER 4 — PANEL COMPARE / SELECT (gridline-separated panels).
# Decompose by separator lines (or even halves/thirds) into equal panels. Two relations:
#   (A) PER-CELL boolean across panels with a FITTED op + a FITTED 2-color output map. (gen2_base has a
#       fixed panel-logic, but only over a restricted on/off color scheme; here we fit the full map and
#       also support N>2 panels and majority/equal/diff ops.)
#   (B) SELECT one panel by a property (most/least non-bg cells, the unique panel, the symmetric one),
#       output = that panel verbatim.
# ===========================================================================
def _panels(g):
    """Yield lists of equal-shape panels split by a full-line separator OR by even halves/thirds."""
    h, w = g.shape
    res = []
    for ax in (0, 1):
        n = g.shape[ax]
        for c in np.unique(g):
            line = np.all(g == c, axis=1 - ax)
            idx = np.where(line)[0]
            if 0 < len(idx) < n:
                segs = []
                prev = 0
                for i in list(idx) + [n]:
                    if i > prev:
                        seg = g[prev:i, :] if ax == 0 else g[:, prev:i]
                        segs.append(seg)
                    prev = i + 1
                shapes = set(s.shape for s in segs)
                if len(segs) >= 2 and len(shapes) == 1:
                    res.append(segs)
    for k in (2, 3):
        if w % k == 0 and w // k >= 1:
            res.append([g[:, i * w // k:(i + 1) * w // k] for i in range(k)])
        if h % k == 0 and h // k >= 1:
            res.append([g[i * h // k:(i + 1) * h // k, :] for i in range(k)])
    return res


def fit_panel_relation(train):
    bg = _bg_train(train)
    # ---- (A) per-cell boolean across panels with a fitted output color map ----
    g0, o0 = train[0]
    # try to find a panel split whose panel shape == output shape, consistent across train
    # We index split candidates of the first input by (npanels, panel-shape).
    def split_sig(g):
        out = []
        for panels in _panels(g):
            out.append((len(panels), panels[0].shape, panels))
        return out

    fns = []
    # ----- (A) per-cell op + color map -----
    for (npan0, pshape0, _p0) in split_sig(g0):
        if pshape0 != o0.shape:
            continue
        # gather panels per train pair matching this signature
        per = []
        ok = True
        for gi, go in train:
            match = None
            for panels in _panels(gi):
                if len(panels) == npan0 and panels[0].shape == go.shape:
                    match = panels
                    break
            if match is None:
                ok = False
                break
            per.append((match, go))
        if not ok:
            continue
        # induce a per-cell function: stack panels -> tuple of (nonbg?) per panel -> output color.
        # Boolean profile = tuple over panels of (cell != bg). Map profile -> output color, consistent.
        prof_map = {}
        good = True
        for panels, go in per:
            ph, pw = go.shape
            stacks = [p != bg for p in panels]
            for r in range(ph):
                for cc in range(pw):
                    prof = tuple(bool(s[r, cc]) for s in stacks)
                    ov = int(go[r, cc])
                    if prof in prof_map and prof_map[prof] != ov:
                        good = False
                        break
                    prof_map[prof] = ov
                if not good:
                    break
            if not good:
                break
        if good and prof_map:
            def fnA(g, npan0=npan0, prof_map=prof_map, bg=bg):
                match = None
                for panels in _panels(g):
                    if len(panels) == npan0:
                        match = panels
                        break
                if match is None:
                    return None
                ph, pw = match[0].shape
                stacks = [p != bg for p in match]
                out = np.empty((ph, pw), int)
                for r in range(ph):
                    for cc in range(pw):
                        prof = tuple(bool(s[r, cc]) for s in stacks)
                        if prof not in prof_map:
                            return None
                        out[r, cc] = prof_map[prof]
                return out
            if _verify(fnA, train):
                fns.append(fnA)
                break

    # ----- (B) select one panel by a property -----
    if not fns:
        propfns = {
            "most_cells": lambda ps: _argunique([{"v": int((p != bg).sum())} for p in ps],
                                                lambda x: x["v"], "max"),
            "least_cells": lambda ps: _argunique([{"v": int((p != bg).sum())} for p in ps],
                                                 lambda x: x["v"], "min"),
            "most_colors": lambda ps: _argunique([{"v": len(set(p.flatten().tolist()))} for p in ps],
                                                 lambda x: x["v"], "max"),
            "unique": lambda ps: _unique_panel(ps),
        }
        for pname, pf in propfns.items():
            ok = True
            for gi, go in train:
                chosen = None
                for panels in _panels(gi):
                    if panels[0].shape != go.shape:
                        continue
                    ki = pf(panels)
                    if ki is None:
                        continue
                    if np.array_equal(panels[ki], go):
                        chosen = True
                        break
                if not chosen:
                    ok = False
                    break
            if ok:
                def fnB(g, pf=pf):
                    best = None
                    for panels in _panels(g):
                        ki = pf(panels)
                        if ki is None:
                            continue
                        return panels[ki]
                    return None
                if _verify(fnB, train):
                    fns.append(fnB)
                    break
    return fns[0] if fns else None


def _unique_panel(ps):
    sigs = [p.tobytes() for p in ps]
    cnt = Counter(sigs)
    uniq = [i for i, s in enumerate(sigs) if cnt[s] == 1]
    return uniq[0] if len(uniq) == 1 else None


# ===========================================================================
# THE INDUCER REGISTRY + the learned FEATURE-RELEVANCE PRIOR.
# Each inducer decomposes the grid at the two-part granularity, extracts cross-part features, and
# induces a feature->effect relation verified on train. The prior ranks which inducers to try first
# from cheap TASK features (so induction stays tractable). Trained at import on self-gen synthetic
# relational tasks generated by these exact relation kinds (label = which kind generated the pair).
# ===========================================================================
INDUCERS = [
    ("marker_stamp", fit_marker_stamp),
    ("aligned_pair_fill", fit_aligned_pair_fill),
    ("key_object_to_others", fit_key_object_to_others),
    ("panel_relation", fit_panel_relation),
]
INDUCER_NAMES = [n for n, _ in INDUCERS]


def _task_features(train):
    """Cheap task-level features used by the relevance prior to RANK inducers (not for fitting)."""
    bg = _bg_train(train)
    f_same = float(all(gi.shape == go.shape for gi, go in train))
    gi0, go0 = train[0]
    # average #objects, #singleton-markers, #panels-detected, color counts
    nobj = nsing = npan = ncol_in = ncol_out = grow = 0.0
    n = len(train)
    for gi, go in train:
        comps = _components(gi, bg=bg, diag=True)
        nobj += len(comps)
        nsing += sum(1 for c in comps if len(c) == 1)
        npan += 1.0 if _panels(gi) else 0.0
        ncol_in += len({int(v) for v in np.unique(gi)})
        ncol_out += len({int(v) for v in np.unique(go)})
        grow += 1.0 if go.size != gi.size else 0.0
    return np.array([
        1.0, f_same,
        nobj / n, nsing / n, npan / n,
        ncol_in / n, ncol_out / n, (ncol_out - ncol_in) / n,
        grow / n,
        float(nsing / n >= 2.0), float(nobj / n >= 2.0),
    ], float)


FEAT_DIM = len(_task_features([(np.zeros((3, 3), int), np.zeros((3, 3), int))]))


# ---- synthetic relational task generators (one per inducer kind) for the relevance prior ----
def _gen_marker_stamp(rng):
    h = rng.randint(7, 14); w = rng.randint(7, 14)
    g = np.zeros((h, w), int)
    colmap = {}
    ncol = rng.randint(1, 4)
    cols = rng.choice(range(1, 9), ncol, replace=False).tolist()
    for c in cols:
        # a small random footprint
        foot = {(0, 0): c}
        for _ in range(rng.randint(2, 5)):
            dr = rng.randint(-2, 3); dc = rng.randint(-2, 3)
            foot[(dr, dc)] = int(rng.randint(1, 9))
        colmap[c] = foot
    out = g.copy()
    nm = rng.randint(2, 5)
    placed = []
    for _ in range(nm):
        r = rng.randint(2, h - 2); cc = rng.randint(2, w - 2)
        if any(abs(r - pr) < 4 and abs(cc - pc) < 4 for pr, pc in placed):
            continue
        col = int(rng.choice(cols))
        g[r, cc] = col
        placed.append((r, cc))
        for (dr, dc), ov in colmap[col].items():
            a, b = r + dr, cc + dc
            if 0 <= a < h and 0 <= b < w:
                out[a, b] = ov
    if not placed:
        return None
    return g, out


def _gen_aligned_pair_fill(rng):
    h = rng.randint(6, 12); w = rng.randint(6, 12)
    g = np.zeros((h, w), int)
    out = g.copy()
    fill = int(rng.randint(1, 9))
    col = int(rng.randint(1, 9))
    while col == fill:
        col = int(rng.randint(1, 9))
    made = False
    for _ in range(rng.randint(1, 3)):
        if rng.rand() < 0.5:
            r = rng.randint(0, h); a = rng.randint(0, w - 2); b = rng.randint(a + 2, w)
            g[r, a] = col; g[r, b] = col
            out[r, a] = col; out[r, b] = col
            out[r, a + 1:b] = fill
            made = True
        else:
            cc = rng.randint(0, w); a = rng.randint(0, h - 2); b = rng.randint(a + 2, h)
            g[a, cc] = col; g[b, cc] = col
            out[a, cc] = col; out[b, cc] = col
            out[a + 1:b, cc] = fill
            made = True
    if not made:
        return None
    return g, out


def _gen_key_object(rng):
    h = rng.randint(8, 14); w = rng.randint(8, 14)
    g = np.zeros((h, w), int)
    cols = rng.choice(range(1, 9), 3, replace=False).tolist()
    # one big key object, several small
    boxes = []
    def place(sz, col):
        for _ in range(20):
            r = rng.randint(0, h - sz); cc = rng.randint(0, w - sz)
            if all(not (r < br + bs + 1 and br < r + sz + 1 and cc < bc + bs + 1 and bc < cc + sz + 1)
                   for br, bc, bs in boxes):
                g[r:r + sz, cc:cc + sz] = col
                boxes.append((r, cc, sz))
                return True
        return False
    keycol = cols[0]
    if not place(3, keycol):
        return None
    nsm = 0
    for _ in range(rng.randint(2, 4)):
        if place(1, int(rng.choice(cols[1:]))):
            nsm += 1
    if nsm < 1:
        return None
    out = g.copy()
    # recolor others to key color
    out[(g != 0) & (g != keycol)] = keycol
    return g, out


def _gen_panel(rng):
    ph = rng.randint(3, 6); pw = rng.randint(3, 6)
    a = (rng.rand(ph, pw) < 0.4).astype(int)
    b = (rng.rand(ph, pw) < 0.4).astype(int)
    sep = int(rng.randint(1, 9))
    col = int(rng.randint(1, 9))
    while col == sep:
        col = int(rng.randint(1, 9))
    A = a * col; B = b * col
    g = np.concatenate([A, np.full((ph, 1), sep, int), B], axis=1)
    # AND -> output color
    oc = int(rng.randint(1, 9))
    out = ((a & b) * oc).astype(int)
    return g, out


_GENS = {
    "marker_stamp": _gen_marker_stamp,
    "aligned_pair_fill": _gen_aligned_pair_fill,
    "key_object_to_others": _gen_key_object,
    "panel_relation": _gen_panel,
}


def _train_relevance_prior(n_per=180, seed=0):
    """Tiny softmax ranker: task-features -> which inducer kind generated the task. Used only to ORDER
    inducer trials (the verifier supplies correctness). Trained on self-gen relational tasks."""
    rng = np.random.RandomState(seed)
    X = []
    y = []
    idx = {n: k for k, n in enumerate(INDUCER_NAMES)}
    for name in INDUCER_NAMES:
        gen = _GENS[name]
        made = 0
        tries = 0
        while made < n_per and tries < n_per * 6:
            tries += 1
            # 2-4 train pairs sharing the same generative relation = a synthetic task
            npairs = rng.randint(2, 4)
            pairs = []
            for _ in range(npairs):
                r = None
                for _t in range(4):
                    r = gen(rng)
                    if r is not None:
                        break
                if r is None:
                    break
                pairs.append((r[0], r[1]))
            if len(pairs) < 2:
                continue
            try:
                f = _task_features(pairs)
            except Exception:
                continue
            X.append(f)
            y.append(idx[name])
            made += 1
    if not X:
        return np.zeros((len(INDUCER_NAMES), FEAT_DIM)), np.zeros(FEAT_DIM), np.ones(FEAT_DIM)
    X = np.array(X)
    y = np.array(y)
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    W = np.zeros((len(INDUCER_NAMES), FEAT_DIM))
    lr = 0.3
    rng2 = np.random.RandomState(1)
    N = len(Xs)
    for _e in range(8):
        for i in rng2.permutation(N):
            f = Xs[i]; t = y[i]
            logits = W @ f
            logits -= logits.max()
            p = np.exp(logits); p /= p.sum()
            p[t] -= 1.0
            W -= lr * np.outer(p, f)
    return W, mu, sd


_T0 = time.time()
try:
    _W, _MU, _SD = _train_relevance_prior()
except Exception:
    _W, _MU, _SD = np.zeros((len(INDUCER_NAMES), FEAT_DIM)), np.zeros(FEAT_DIM), np.ones(FEAT_DIM)
_BUILD_SEC = time.time() - _T0


def _rank_inducers(train):
    try:
        f = _task_features(train)
        fs = (f - _MU) / _SD
        scores = _W @ fs
        order = sorted(range(len(INDUCER_NAMES)), key=lambda k: -scores[k])
        return [INDUCERS[k] for k in order]
    except Exception:
        return list(INDUCERS)


# ===========================================================================
# IN-RUN EXPERIENCE: remember which inducer kind verified, to try it first next time (a learned, not
# hand-coded, order prior). Verified-only; stores no grids.
# ===========================================================================
_HITS = Counter()


def reset_library():
    """Documented hook: clear cross-task experience so a run starts cold (gate uses this to isolate
    transfer). Also resets the imported base library so the whole solver is genuinely cold."""
    _HITS.clear()
    if hasattr(BASE, "reset_library"):
        try:
            BASE.reset_library()
        except Exception:
            pass
    else:
        try:
            BASE._LIB.__init__()
        except Exception:
            pass


# ===========================================================================
# INVENTION ENTRYPOINT — induce two-part relations; return up to 2 candidate outputs per test input.
# ===========================================================================
def _invent(train, test_inputs, budget):
    train = [(np.asarray(gi, int), np.asarray(go, int)) for gi, go in train]
    ordered = _rank_inducers(train)
    # experience prior: inducers that verified before this run go first (ties keep ranker order)
    ordered = sorted(ordered, key=lambda nf: -_HITS.get(nf[0], 0))

    fitted = []
    t0 = time.time()
    for name, fitter in ordered:
        if time.time() - t0 > 6.0:
            break
        try:
            fn = fitter(train)
        except Exception:
            fn = None
        if fn is not None and _verify(fn, train):
            fitted.append((name, fn))
            _HITS[name] += 1

    attempts = []
    for gi in test_inputs:
        gi = np.asarray(gi, int)
        cand = []
        for _name, fn in fitted:
            try:
                o = fn(gi)
            except Exception:
                o = None
            if o is not None and getattr(o, "ndim", None) == 2 and o.size > 0:
                o = np.asarray(o, int)
                if not any(_eq(o, c) for c in cand):
                    cand.append(o)
            if len(cand) >= 2:
                break
        attempts.append(cand[:2])
    return attempts


# ===========================================================================
# STANDARDIZED GATE WIRING
#   solve_ablated == gen2_base.solve  (the strong retrieval ablation, imported verbatim)
#   solve         == gen2_base attempt-1 backstop, THEN two-part relation-induction attempt-2
# ===========================================================================
def solve_ablated(train, test_inputs, budget):
    """EXACTLY gen2_base.solve — the standardized strong-retrieval ablation. INVENTED = solved - this."""
    return BASE.solve(train, test_inputs, budget)


def solve(train, test_inputs, budget):
    """gen2_base as attempt-1 backstop; relation-induction fills the remaining attempt slot. Reserve
    attempt-1 for the strong retrieval backstop and attempt-2 for the INVENTION so a relation is never
    crowded out by a second base guess. If invention has nothing, base keeps both slots."""
    # attempt 1: the strong retrieval baseline (never regress below it)
    try:
        base_attempts = BASE.solve(train, test_inputs, budget)
    except Exception:
        base_attempts = []
    if not isinstance(base_attempts, list):
        base_attempts = []
    norm = []
    for k in range(len(test_inputs)):
        a = base_attempts[k] if k < len(base_attempts) else []
        if a is None:
            a = []
        norm.append([np.asarray(x, int) for x in a if x is not None][:2])

    # attempt 2: two-part relation-induction invention
    try:
        inv = _invent(train, test_inputs, max(800, budget))
    except Exception:
        inv = [[] for _ in test_inputs]

    merged = []
    for k in range(len(test_inputs)):
        b = list(norm[k])
        iv = [o for o in (inv[k] if k < len(inv) else []) if o is not None]
        cand = []
        if b:
            cand.append(b[0])                       # attempt 1 = base backstop
        for o in iv:                                # attempt 2 = first invention not already present
            if not any(_eq(o, c) for c in cand):
                cand.append(o)
                break
        if len(cand) < 2:                           # backfill: extra base guess, then extra invention
            for o in (b[1:] + iv):
                if not any(_eq(o, c) for c in cand):
                    cand.append(o)
                if len(cand) >= 2:
                    break
        merged.append(cand[:2])
    return merged


# self-generated synthetic sanity at import (validates each inducer runs on its own generated task)
def _selftest():
    rng = np.random.RandomState(0)
    for name, fitter in INDUCERS:
        gen = _GENS[name]
        pairs = []
        for _ in range(3):
            r = gen(rng)
            if r is not None:
                pairs.append(r)
        if len(pairs) >= 2:
            try:
                fitter(pairs)
            except Exception:
                pass


_selftest()
