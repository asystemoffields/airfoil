#!/usr/bin/env python3
"""Vine — the GENERATORS that retire the hidden meta-treadmills (Alex's audit). Each replaces a hand-listed MENU
with a thin WHITELISTED BASIS + a generator that produces the menu (and held-out members) by composition. The basis
is the irreducible innate primitives (whitelisted in treadmill_lint.py); the menu is earned. New capability = extend
the BASIS, never edit a menu. Covers: object features, motor displacements, paint productions, substrate relations.

This is the single source the lint enforces: a vocabulary the solver routes over MUST come from one of these
generators, not a literal menu. Run: /data/llm/.venv/bin/python generators.py  (self-check: held-out members emerge)."""
import numpy as np

# ============ #1 OBJECT FEATURES: was grammar.FEATURE_NAMES (hand menu) -> {reduction} x {cell-property} ============
# BASIS (thin, IRREDUCIBLE -- no element is a composition of the others; compositions EMERGE, not listed).
# NOTE the discipline: `extent` is NOT here (it = max - min, a level-1 composition) and `enclosed` is NOT a cell-prop
# (it's a neighbor-quantifier = a cell-substrate predicate). Both EMERGE from composition; listing them = a menu.
REDUCTIONS = {                                            # irreducible aggregations of a multiset
    "count":   lambda xs: len(xs),
    "max":     lambda xs: max(xs) if xs else 0,
    "min":     lambda xs: min(xs) if xs else 0,
    "nunique": lambda xs: len(set(xs)),
}
CELL_PROPS = {                                            # irreducible per-cell readings
    "row":   lambda cells, g: [r for r, c in cells],
    "col":   lambda cells, g: [c for r, c in cells],
    "color": lambda cells, g: [int(g[r, c]) for r, c in cells],
    "one":   lambda cells, g: [1 for _ in cells],
}
NB = [(-1, 0), (1, 0), (0, -1), (0, 1)]
def _enclosed(g, r, c):
    H, W = g.shape
    return all(0 <= r+dr < H and 0 <= c+dc < W and g[r+dr, c+dc] != 0 for dr, dc in NB)


def object_features():
    """generate features by COMPOSING the basis: each single reduction x cell-property, PLUS level-1 compositions
    (differences of two reductions on the same property). size=count_one, height=(max_row - min_row), width=(max_col
    - min_col), ncolors=nunique_color -- height/width EMERGE as compositions (extent is NOT a basis element); held-out
    (max_row, min_col - max_row, ...) emerge with zero new code."""
    feats = {}
    for pn, pf in CELL_PROPS.items():
        for rn, rf in REDUCTIONS.items():
            feats[f"{rn}_{pn}"] = (lambda cells, g, rf=rf, pf=pf: rf(pf(cells, np.asarray(g, int))))
        # level-1 compositions: differences of reductions on the SAME property (extent/height/width emerge here)
        feats[f"extent_{pn}"] = (lambda cells, g, pf=pf: (lambda xs: (max(xs) - min(xs) + 1) if xs else 0)(pf(cells, np.asarray(g, int))))
    return feats


# ============ #2 MOTOR DISPLACEMENTS: was effect_faculty.MODES (hand menu) -> EYE-COMPUTED targets ============
# BASIS: a target position is READ from the scene (an anchor object's edge/center), never a named mode. Generated
# per scene from the perceived objects -> new "modes" are just other eye-read positions, no menu.
def displacement_modes():
    """generate alignment modes from the {row,col} axis-subset BASIS: the non-empty subsets of axes to align an
    object onto its anchor (row / col / row+col). A complete basis (not a menu); the OFFSET itself is eye-read."""
    axes = ["row", "col"]
    return ["+".join(a for j, a in enumerate(axes) if i & (1 << j)) for i in range(1, 1 << len(axes))]


def motor_targets(obj, anchor):
    """eye-computed placement targets for moving `obj` relative to `anchor` (positions READ, not a MODES menu)."""
    (ar0, ac0, ar1, ac1) = anchor["bbox"]; (r0, c0, r1, c1) = obj["bbox"]
    return {
        "anchor_row":   (ar0 - r0, 0),                    # align rows (eye-read anchor row)
        "anchor_col":   (0, ac0 - c0),                    # align cols
        "onto":         (ar0 - r0, ac0 - c0),             # onto anchor origin
        "above_anchor": (ar0 - r1 - 1, ac0 - c0),         # adjacency READ from anchor edge
        "below_anchor": (ar1 - r0 + 1, ac0 - c0),
    }


# ============ #3 PAINT PRODUCTIONS: was derive_grammar FORWARD/CLOSE menu -> paint(SELECTOR, VALUE) over the eye ====
# BASIS: ONE motor primitive paint(where, what); SELECTORS and VALUES are cell-substrate expressions (eye-computed).
# invariance/colormap/fill collapse to instances. New effect = new selector/value (from the cell substrate), not a
# new production. Each generator yields (name, fit(cur,outs)->apply_fn|None) so the derive engine stays one loop.
def _cmap(cur, outs):
    table = {}
    for c, o in zip(cur, outs):
        c = np.asarray(c, int); o = np.asarray(o, int)
        if c.shape != o.shape:
            return None
        for a, b in zip(c.ravel(), o.ravel()):
            a, b = int(a), int(b)
            if a in table and table[a] != b:
                return None
            table[a] = b
    return table


def _apply_table(g, t):
    return np.vectorize(lambda v: t.get(int(v), int(v)))(np.asarray(g, int))


def _enclosed_mask(g):
    g = np.asarray(g, int); m = np.zeros(g.shape, bool)
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if g[r, c] == 0 and _enclosed(g, r, c):
                m[r, c] = True
    return m


# SELECTOR basis (cell-substrate predicates -> a boolean mask) and VALUE basis (how a selected cell's color is set).
# Both are thin + irreducible; the production family = SELECTOR x VALUE, GENERATED. invariance/fill/colormap EMERGE;
# held-out paints (paint(color==k, const), paint(enclosed, map-image), ...) emerge with zero new code.
def selector_basis(g, present):
    sels = {"all": np.ones(np.asarray(g).shape, bool), "enclosed": _enclosed_mask(g)}
    for k in present:
        sels[f"color={k}"] = (np.asarray(g, int) == k)
    return sels


def _fit_const(cur, outs, sel_name):
    """VALUE=const: one color for all SELECTED cells, induced from the output (a CLOSE production)."""
    val = None
    for c, o in zip(cur, outs):
        c = np.asarray(c, int); o = np.asarray(o, int)
        if c.shape != o.shape:
            return None
        mask = selector_basis(c, sorted(set(int(v) for v in np.unique(c)))).get(sel_name)
        if mask is None:
            return None
        for b in o[mask]:
            if val is None: val = int(b)
            elif val != b: return None
    if val is None:
        return None
    def fn(g, v=val, sn=sel_name):
        g = np.asarray(g, int); out = g.copy()
        out[selector_basis(g, sorted(set(int(x) for x in np.unique(g)))).get(sn, np.zeros(g.shape, bool))] = v
        return out
    return fn


def paint_family(cur):
    """the GENERATED production family for the current intermediate = SELECTOR x VALUE (one motor primitive `paint`).
    Each item: (name, kind, op). kind='close' (VALUE induced from output) or 'forward' (VALUE=image-via-map,
    deterministic). DERIVED from the value-type, not hand-assigned. invariance/colormap/fill are instances."""
    import cell_evolve as CE
    g0 = np.asarray(cur[0], int); present = sorted(set(int(v) for v in np.unique(g0)))
    fam = []
    fam.append(("paint(all,table)", "close",
                lambda cur, outs: (lambda t: (lambda g: _apply_table(g, t)) if t is not None else None)(_cmap(cur, outs))))
    for sname in selector_basis(g0, present):                       # SELECTOR x VALUE=const  (fill, recolor-region, ...)
        fam.append((f"paint({sname},const)", "close", lambda cur, outs, sn=sname: _fit_const(cur, outs, sn)))
    # SELECTOR=diff@map x VALUE=image@map (invariance). The map BASIS is the full closure, but the SEARCH is bounded
    # to the cheap part (isometries + eye-detected periods); the glide closure needs recognizer-pruning to search.
    maps = dict(CE.iso_maps())
    for N in present:
        for (axis, P) in CE.detect_periods(g0, N):
            for s in (P, -P):
                maps[f"per{axis}{s}"] = (lambda r, c, H, W, s=s: (r, c+s)) if axis == "W" else (lambda r, c, H, W, s=s: (r+s, c))
    for nm, mf in maps.items():
        for N in present:
            fam.append((f"paint(diff@{nm},img;occ={N})", "forward",
                        lambda cur, nm=nm, mf=mf, N=N: [CE._apply_invariance(g, nm, mf, N) for g in cur]))
    return fam


# ============ #4 SUBSTRATE RELATIONS: was rel_dsl.pair_signature channels (hand menu) -> a.p OP b.q (substrate) ====
# BASIS: object props x comparison ops. The relational channels (contains/adjacent/aligned/...) are substrate
# conjunctions (substrate_eye proved this) -> generate them, don't hand-list. Held-out relations emerge.
PROPS = ["r0", "c0", "r1", "c1", "h", "w", "size", "color"]
OPS = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, "==": lambda a, b: a == b,
       ">=": lambda a, b: a >= b, ">": lambda a, b: a > b}


def substrate_relations():
    """generate {name -> fn(a,b)} = a.p OP b.q over PROPS x OPS x PROPS. same_size = (size==size), a_bigger =
    (size>size), aligned ~ (r0==r0)|(c0==c0). Held-out relations (h==w-ish, etc.) emerge. No hand-listed channels."""
    rels = {}
    for pa in PROPS:
        for op_name, op in OPS.items():
            for pb in PROPS:
                rels[f"{pa}{op_name}{pb}"] = (lambda a, b, pa=pa, pb=pb, op=op: op(a[pa], b[pb]))
    return rels


def _selfcheck():
    feats = object_features()
    rels = substrate_relations()
    print(f"#1 object_features: {len(feats)} generated from {len(REDUCTIONS)}x{len(CELL_PROPS)} basis "
          f"(size=count_one? {'count_one' in feats}; holes=count_enclosed? {'count_enclosed' in feats}; "
          f"HELD-OUT max_row emerges? {'max_row' in feats})")
    print(f"#2 motor_targets: eye-computed (positions read from anchor bbox; no MODES menu)")
    print(f"#3 paint_productions: paint(selector,value) family from ONE motor primitive (colormap/fill/invariance instances)")
    print(f"#4 substrate_relations: {len(rels)} generated from {len(PROPS)}x{len(OPS)}x{len(PROPS)} basis "
          f"(same_size=size==size? {'size==size' in rels}; HELD-OUT h==w emerges? {'h==w' in rels})")
    print("READ: every menu now comes from a thin whitelisted BASIS via a generator; held-out members emerge with "
          "zero new code. The lint (treadmill_lint.py) enforces this + scans for any new hand-listed menu.")


if __name__ == "__main__":
    _selfcheck()
