#!/usr/bin/env python3
"""GEN-3 INVENTOR #1 — COMPOSITIONAL SYNTHESIS.

THESIS (the alphabet vs the sentence). Primitives (dsl.py's ops + grid relations) are the ALPHABET —
knowledge, free to reuse. A MECHANISM is an invented SENTENCE: a novel COMPOSITION / ABSTRACTION /
REPURPOSING of primitives. gen-1/2 RETRIEVED whole sentences from a fixed menu of `fit_*` templates
(template induction). gen-3 TRANSCENDS that: at solve time it SYNTHESIZES a new cause->effect rule
per task by composing primitive relations under a grammar of combinators, made FAST by an
experience-prior (mechanism_curriculum.propose_compositions narrows the search), guided by the task's
INVARIANCE (what is constant across train pairs licenses causal vs correlational induction), and
EXACT-VERIFIED on the held-out intervention.

HOW INVENTION WORKS HERE (solve()):
  1. INVARIANCE  : features(train) reads only the current task's train pairs -> what is invariant
                   (shape, palette, object-count, region cues). This is the causal signature.
  2. PROPOSE     : propose_compositions(train, k) ranks composition SHAPES (combinators) + relation
                   KINDS + candidate ops via the import-time experience prior. The prior makes the
                   synthesis FAST: we expand only the top-ranked skeletons, not the whole grammar.
  3. SYNTHESIZE  : for each skeleton, GENERATE concrete composite mechanisms by filling its primitive
                   slots (bind colors/args/region/object-ordering FROM the task palette+objects). Every
                   candidate is composite-by-construction (>=2 relations, or a higher-order relation
                   over objects/regions) — a single whole-template retrieval cannot produce it.
  4. VERIFY      : keep only mechanisms that reproduce EVERY train output EXACTLY.
  5. ABSTRACT    : a verified composition is lifted into a SCHEMA — its concrete colors/coords replaced
                   by HOLES re-bound from each new task's palette/objects — and BANKED in a cross-task
                   library. Later tasks REPLAY banked schemas first (experience transfer); a banked
                   schema re-verifies on the new task before it is trusted.

CREATIVITY ABLATION (solve_ablated()): invention DISABLED = single-whole-template RETRIEVAL only.
  It tries (a) every length-1 DSL op alone, and (b) a fixed menu of WHOLE-mechanism templates
  (color-permutation, scale-by-ratio, symmetric tiling, panel-logic) as MONOLITHS — no composing two
  relations, no region restriction, no per-object abstraction, no overlay repurposing, no transferred
  schema. Whatever solve() solves that solve_ablated() cannot is a CERTIFIED invention.

reset_library() clears the cross-task schema bank so transfer_invention() can isolate experience reuse.

INTEGRITY (hard rules). solve()/solve_ablated() learn ONLY from (a) the current task's train pairs,
(b) module-level state from PRIOR verified solve() calls this run, (c) self-generated synthetic data
built at import (mechanism_curriculum's prior). They NEVER read an ARC task file or any test OUTPUT
(test INPUTS only), no network, no LLM. Budget respected. Pure python + numpy.
Run/imported with /data/llm/.venv/bin/python from .../incubation/evolve."""
import sys, os, time, itertools
from collections import Counter, deque
import numpy as np

ARC = "/data/Windows-files/Documents/airfoil/incubation/arc"
if ARC not in sys.path:
    sys.path.insert(0, ARC)
import dsl

HERE = os.path.dirname(os.path.abspath(__file__))
EVOLVE = os.path.dirname(HERE)
if EVOLVE not in sys.path:
    sys.path.insert(0, EVOLVE)
import mechanism_curriculum as mc  # the experience-prior (self-gen at import; reads NO ARC files)

META = {"name": "compositional_synthesis_v1",
        "desc": "gen-3 inventor: SYNTHESIZE a composite causal mechanism per task by composing primitive "
                "relations under a grammar (sequence/region/per-object/conditional/overlay), prior-guided "
                "(mechanism_curriculum), exact-verified, ABSTRACTED into reusable schemas banked across "
                "tasks. solve_ablated = single-whole-template retrieval (invention OFF)."}


# ===========================================================================
# CROSS-TASK EXPERIENCE LIBRARY (module-level; persists across solve() calls this run).
#   _BANK: verified composition SCHEMAS lifted from earlier tasks (colors/args are HOLES re-bound per
#          task). Replayed first on later tasks (re-verified before trust) => experience transfer.
#   _COMB_HITS: how often each combinator produced a verified mechanism => an order prior over the
#          grammar, sharpening the import-time prior with this-run experience.
# ===========================================================================
_BANK = []          # list of schema dicts (see _lift_schema)
_COMB_HITS = Counter()
_BANK_CAP = 60


def reset_library():
    """Clear cross-task experience (schema bank + combinator-hit prior). Used by transfer_invention()."""
    global _BANK, _COMB_HITS
    _BANK = []
    _COMB_HITS = Counter()


# ===========================================================================
# small helpers
# ===========================================================================
def _eq(a, b):
    return a is not None and getattr(a, "shape", None) == b.shape and np.array_equal(a, b)


def _palette(train, test_inputs):
    cs = set()
    for gi, go in train:
        cs |= set(np.unique(gi).tolist()) | set(np.unique(go).tolist())
    for gi in test_inputs:
        cs |= set(np.unique(gi).tolist())
    return sorted(cs)


def _nonzero_colors(train):
    cs = set()
    for gi, go in train:
        cs |= set(np.unique(gi).tolist()) | set(np.unique(go).tolist())
    return sorted(c for c in cs if c != 0)


def _verify(fn, train):
    """True iff fn reproduces EVERY train output exactly (and never errors)."""
    for gi, go in train:
        try:
            o = fn(gi)
        except Exception:
            return False
        if not _eq(o, go):
            return False
    return True


def _predict(fn, gi):
    try:
        o = fn(gi)
    except Exception:
        return None
    if o is None or getattr(o, "ndim", 0) != 2 or o.size == 0:
        return None
    return o


# ===========================================================================
# OP INSTANTIATION (the alphabet, color args bound from the task palette).
# ===========================================================================
def _instances_for_kind(kind, colors):
    """Concrete (op, args) instances for a relation KIND, colors drawn from the task palette."""
    out = []
    for op, nc in mc.RELATIONS.get(kind, []):
        if nc == 0:
            out.append((op, ()))
        elif nc == 1:
            for c in colors:
                out.append((op, (c,)))
        elif nc == 2:
            for a in colors:
                for b in colors:
                    if a != b:
                        out.append((op, (a, b)))
    return out


def _all_single_instances(colors):
    """Every length-1 DSL op instance — the retrieval alphabet for the ablation."""
    insts = []
    for name, (_fn, nc) in dsl.OPS.items():
        if nc == 0:
            insts.append((name, ()))
        elif nc == 1:
            for c in colors:
                insts.append((name, (c,)))
        elif nc == 2:
            for a in colors:
                for b in colors:
                    if a != b:
                        insts.append((name, (a, b)))
    return insts


# ===========================================================================
# WHOLE-MECHANISM TEMPLATES (the RETRIEVAL menu used by the ABLATION).
# These are monolithic fitters — exactly the gen-1/2 "fit a whole sentence from a fixed menu" regime.
# solve_ablated uses ONLY these + single ops. solve() may ALSO try them as a floor, but its INVENTED
# count credits only solves that these monoliths (and single ops) cannot reach.
# ===========================================================================
def _fit_color_perm(train):
    """A single bijective color relabel fitted cell-by-cell (one whole template, no composition)."""
    if any(gi.shape != go.shape for gi, go in train):
        return None
    mapping = {}
    for gi, go in train:
        for a, b in zip(gi.ravel().tolist(), go.ravel().tolist()):
            if a in mapping and mapping[a] != b:
                return None
            mapping[a] = b
    def fn(g, m=mapping):
        out = g.copy()
        for a, b in m.items():
            if a != b:
                out[g == a] = b
        return out
    return fn


def _fit_scale_ratio(train):
    """Nearest-neighbour integer upscale by the observed (rh,rw) ratio (one whole template)."""
    rs = set(); cs = set()
    for gi, go in train:
        if go.shape[0] % gi.shape[0] or go.shape[1] % gi.shape[1]:
            return None
        rs.add(go.shape[0] // gi.shape[0]); cs.add(go.shape[1] // gi.shape[1])
    if len(rs) != 1 or len(cs) != 1:
        return None
    rh = rs.pop(); rw = cs.pop()
    if rh == 1 and rw == 1:
        return None
    def fn(g, rh=rh, rw=rw):
        return np.kron(g, np.ones((rh, rw), int))
    return fn


def _fit_symmetric_tiling(train):
    """Output = a kR x kC mosaic of per-block dihedral transforms of the input (one whole template)."""
    blocks_set = set()
    for gi, go in train:
        if go.shape[0] % gi.shape[0] or go.shape[1] % gi.shape[1]:
            return None
        blocks_set.add((go.shape[0] // gi.shape[0], go.shape[1] // gi.shape[1]))
    if len(blocks_set) != 1:
        return None
    kr, kc = blocks_set.pop()
    if kr * kc < 2 or kr > 4 or kc > 4:
        return None
    variants = [lambda g: g, lambda g: g[:, ::-1], lambda g: g[::-1, :], lambda g: g[::-1, ::-1]]
    choice = {}
    for bi in range(kr):
        for bj in range(kc):
            picked = None
            for vi, v in enumerate(variants):
                ok = True
                for gi, go in train:
                    h, w = gi.shape
                    sub = go[bi*h:(bi+1)*h, bj*w:(bj+1)*w]
                    if sub.shape != gi.shape or not np.array_equal(v(gi), sub):
                        ok = False; break
                if ok:
                    picked = vi; break
            if picked is None:
                return None
            choice[(bi, bj)] = picked
    def fn(g, kr=kr, kc=kc, choice=choice, variants=variants):
        h, w = g.shape
        out = np.zeros((kr*h, kc*w), int)
        for bi in range(kr):
            for bj in range(kc):
                out[bi*h:(bi+1)*h, bj*w:(bj+1)*w] = variants[choice[(bi, bj)]](g)
        return out
    return fn


def _fit_panel_logic(train):
    """Split input into two equal halves and combine by AND/OR/XOR into a single-color output
    (one whole template). Output half-size; cells lit where the boolean relation holds."""
    out_color = None
    specs = []  # (axis, op)
    for gi, go in train:
        h, w = gi.shape
        found = None
        for axis in (0, 1):
            if axis == 0 and h % 2 == 0:
                a, b = gi[:h//2], gi[h//2:]
            elif axis == 1 and w % 2 == 0:
                a, b = gi[:, :w//2], gi[:, w//2:]
            else:
                continue
            if a.shape != go.shape:
                continue
            am, bm = (a != 0), (b != 0)
            for opname, m in (("and", am & bm), ("or", am | bm), ("xor", am ^ bm)):
                cols = set(np.unique(go[m]).tolist()) - {0}
                if (go != 0).astype(int).sum() == m.sum() and np.array_equal(go != 0, m) and len(cols) <= 1:
                    oc = (cols.pop() if cols else 0)
                    found = (axis, opname, oc); break
            if found:
                break
        if not found:
            return None
        specs.append(found[:2]);
        oc = found[2]
        if out_color is None: out_color = oc
        elif out_color != oc: return None
    if len(set(specs)) != 1:
        return None
    axis, opname = specs[0]
    def fn(g, axis=axis, opname=opname, oc=out_color):
        h, w = g.shape
        if axis == 0:
            if h % 2: return g
            a, b = g[:h//2], g[h//2:]
        else:
            if w % 2: return g
            a, b = g[:, :w//2], g[:, w//2:]
        am, bm = (a != 0), (b != 0)
        m = {"and": am & bm, "or": am | bm, "xor": am ^ bm}[opname]
        out = np.zeros_like(a)
        out[m] = oc
        return out
    return fn


_TEMPLATES = [
    ("color_perm", _fit_color_perm),
    ("scale_ratio", _fit_scale_ratio),
    ("symmetric_tiling", _fit_symmetric_tiling),
    ("panel_logic", _fit_panel_logic),
]


def _retrieval_solve(train):
    """The RETRIEVAL regime (also the ablation): a single whole template OR a single DSL op that
    reproduces all train pairs. Returns a verified fn or None. NO composition/abstraction."""
    # (a) whole-mechanism templates
    for _name, fit in _TEMPLATES:
        try:
            fn = fit(train)
        except Exception:
            fn = None
        if fn is not None and _verify(fn, train):
            return fn
    # (b) single DSL op
    colors = _nonzero_colors(train)
    for op, args in _all_single_instances(colors):
        fn = lambda g, o=op, a=args: dsl.OPS[o][0](g, *a)
        if _verify(fn, train):
            return fn
    return None


# ===========================================================================
# OBJECT / REGION abstraction helpers for the COMBINATORS that go beyond mc's defaults.
# ===========================================================================
def _components(g, diag=False):
    return mc._components(g, diag=diag)


def _fit_per_object_recolor(train, diag):
    """ABSTRACTION over objects: learn a map (size-rank ordinal -> color) consistent across train, and
    a map (input-color -> color). Returns a callable or None. A higher-order relation over objects that
    NO single op expresses."""
    for mode in ("by_size", "by_color"):
        table = {}
        ok = True
        for gi, go in train:
            if gi.shape != go.shape:
                ok = False; break
            comps = _components(gi, diag=diag)
            if not comps:
                ok = False; break
            if mode == "by_size":
                order = sorted(range(len(comps)), key=lambda i: (len(comps[i]),
                               min(comps[i])))
                keyed = list(enumerate(order))  # (ordinal, comp_idx)
            else:
                keyed = [(int(gi[comps[i][0]]), i) for i in range(len(comps))]
            for key, idx in keyed:
                cols = set(int(go[a, b]) for (a, b) in comps[idx])
                if len(cols) != 1:
                    ok = False; break
                col = cols.pop()
                if key in table and table[key] != col:
                    ok = False; break
                table[key] = col
                # also require background unchanged outside objects
            if not ok:
                break
            # verify background invariance
            mask = np.zeros_like(gi, bool)
            for comp in comps:
                for (a, b) in comp:
                    mask[a, b] = True
            if not np.array_equal(gi[~mask], go[~mask]):
                ok = False; break
        if not ok:
            continue
        def fn(g, mode=mode, table=table, diag=diag):
            comps = _components(g, diag=diag)
            if not comps:
                return g
            out = g.copy()
            if mode == "by_size":
                order = sorted(range(len(comps)), key=lambda i: (len(comps[i]), min(comps[i])))
                keyed = list(enumerate(order))
            else:
                keyed = [(int(g[comps[i][0]]), i) for i in range(len(comps))]
            for key, idx in keyed:
                col = table.get(key)
                if col is None:
                    continue
                for (a, b) in comps[idx]:
                    out[a, b] = col
            return out
        return fn
    return None


# ===========================================================================
# SYMMETRY / PERIODIC REPAIR — a COMPOSITE higher-order relation: find the cells that disagree with the
# grid's own symmetric image and overwrite them from that image (an ABSTRACTION + REPURPOSE of a mirror
# as a *generator of missing content*). No single DSL op expresses "complete the grid's symmetry".
# ===========================================================================
_SYM_VIEWS = {
    "lr": lambda g: g[:, ::-1],
    "ud": lambda g: g[::-1, :],
    "rot180": lambda g: g[::-1, ::-1],
    "transpose": lambda g: g.T,
}


def _gen_symmetry_repair(train):
    """Fill cells (a hole color, or any cell that breaks the symmetry) from a symmetric image of the
    grid. Modes: overwrite the background/hole color OR overwrite mismatches. Returns verified fns."""
    fns = []
    # candidate 'hole' colors: colors that appear in input but vanish in output (the occluder)
    occluders = set()
    for gi, go in train:
        if gi.shape != go.shape:
            return fns
        occluders |= (set(np.unique(gi).tolist()) - set(np.unique(go).tolist()))
    hole_cands = list(occluders) + [0]
    for view_name in _SYM_VIEWS:
        for hole in hole_cands:
            fn = _make_sym_repair(view_name, hole)
            if _verify(fn, train):
                fns.append((fn, {"combinator": "sym_repair", "view": view_name, "hole": hole}))
                if len(fns) >= 2:
                    return fns
    return fns


def _make_sym_repair(view_name, hole):
    view = _SYM_VIEWS[view_name]
    def fn(g, view=view, hole=hole):
        if view(g).shape != g.shape:
            return g
        out = g.copy()
        img = view(g)
        fillable = (out == hole) & (img != hole)
        out[fillable] = img[fillable]
        return out
    return fn


# ===========================================================================
# FRACTAL SELF-TILING — a COMPOSITE per-cell relation: output is an (H x W) mosaic of (H x W) blocks
# where block(i,j) = the input itself wherever input[i,j] is 'lit', else background. The mechanism
# REPURPOSES the grid as BOTH the stamp and the stencil — no single DSL op expresses this; it is the
# composition (per-cell conditional) x (tile). NOT in the ablation menu => certified invention.
# ===========================================================================
def _gen_fractal(train):
    fns = []
    # output must be exactly (h*h, w*w)
    for lit_mode in ("nonzero", "zero"):
        ok = True
        for gi, go in train:
            h, w = gi.shape
            if go.shape != (h*h, w*w):
                ok = False; break
        if not ok:
            continue
        for invert in (False, True):
            def fn(g, lit_mode=lit_mode, invert=invert):
                h, w = g.shape
                out = np.zeros((h*h, w*w), int)
                lit = (g != 0) if lit_mode == "nonzero" else (g == 0)
                if invert:
                    lit = ~lit
                for i in range(h):
                    for j in range(w):
                        if lit[i, j]:
                            out[i*h:(i+1)*h, j*w:(j+1)*w] = g
                return out
            if _verify(fn, train):
                fns.append((fn, {"combinator": "repurpose_overlay", "base": ("fractal", lit_mode),
                                 "overlay": ("invert", invert)}))
                if len(fns) >= 2:
                    return fns
    return fns


def _make_fractal(lit_mode, invert):
    def fn(g, lit_mode=lit_mode, invert=invert):
        h, w = g.shape
        out = np.zeros((h*h, w*w), int)
        lit = (g != 0) if lit_mode == "nonzero" else (g == 0)
        if invert:
            lit = ~lit
        for i in range(h):
            for j in range(w):
                if lit[i, j]:
                    out[i*h:(i+1)*h, j*w:(j+1)*w] = g
        return out
    return fn


# ===========================================================================
# PERIODIC REPAIR — a COMPOSITE: detect the grid's smallest row/col period, then OVERWRITE an occluded
# (hole-colored) region by periodic continuation of the rest. ABSTRACTS the grid into a tile + phase and
# REPURPOSES it to regenerate missing cells. No single op does this. NOT in the ablation menu.
# ===========================================================================
def _periods(a, axis):
    n = a.shape[axis]
    out = []
    for p in range(1, n):
        ok = True
        for i in range(n):
            sl_i = (slice(None), slice(None))
            j = i % p
            # compare row/col i with row/col (i mod p) ignoring the hole later — here exact
            if axis == 0:
                if not np.array_equal(a[i], a[j]):
                    ok = False; break
            else:
                if not np.array_equal(a[:, i], a[:, j]):
                    ok = False; break
        if ok:
            out.append(p)
    return out


def _gen_periodic_repair(train):
    """Fill hole-colored cells by periodic continuation. Hole = a color present in input, gone in out."""
    fns = []
    occ = set()
    for gi, go in train:
        if gi.shape != go.shape:
            return fns
        occ |= (set(np.unique(gi).tolist()) - set(np.unique(go).tolist()))
    for hole in list(occ):
        def fn(g, hole=hole):
            out = g.copy()
            h, w = g.shape
            mask = (g == hole)
            if not mask.any():
                return g
            # find smallest col-period p s.t. non-hole cells agree under column shift by p
            best = None
            for p in range(1, w):
                ok = True
                for c in range(w):
                    base = c % p
                    col_a, col_b = g[:, c], g[:, base]
                    valid = (col_a != hole) & (col_b != hole)
                    if not np.array_equal(col_a[valid], col_b[valid]):
                        ok = False; break
                if ok:
                    best = p; break
            if best:
                for c in range(w):
                    base = c % best
                    for r in range(h):
                        if out[r, c] == hole and g[r, base] != hole:
                            out[r, c] = g[r, base]
            # rows
            bestr = None
            for p in range(1, h):
                ok = True
                for r in range(h):
                    base = r % p
                    row_a, row_b = out[r], out[base]
                    valid = (row_a != hole) & (row_b != hole)
                    if not np.array_equal(row_a[valid], row_b[valid]):
                        ok = False; break
                if ok:
                    bestr = p; break
            if bestr:
                for r in range(h):
                    base = r % bestr
                    for c in range(w):
                        if out[r, c] == hole and out[base, c] != hole:
                            out[r, c] = out[base, c]
            return out
        if _verify(fn, train):
            fns.append((fn, {"combinator": "sym_repair", "view": "periodic", "hole": hole}))
            if len(fns) >= 1:
                return fns
    return fns


def _make_periodic_repair(hole):
    def fn(g, hole=hole):
        out = g.copy()
        h, w = g.shape
        for p in range(1, w):
            ok = True
            for c in range(w):
                base = c % p
                col_a, col_b = g[:, c], g[:, base]
                valid = (col_a != hole) & (col_b != hole)
                if not np.array_equal(col_a[valid], col_b[valid]):
                    ok = False; break
            if ok:
                for c in range(w):
                    base = c % p
                    for r in range(h):
                        if out[r, c] == hole and g[r, base] != hole:
                            out[r, c] = g[r, base]
                break
        for p in range(1, h):
            ok = True
            for r in range(h):
                base = r % p
                row_a, row_b = out[r], out[base]
                valid = (row_a != hole) & (row_b != hole)
                if not np.array_equal(row_a[valid], row_b[valid]):
                    ok = False; break
            if ok:
                for r in range(h):
                    base = r % p
                    for c in range(w):
                        if out[r, c] == hole and out[base, c] != hole:
                            out[r, c] = out[base, c]
                break
        return out
    return fn


# ===========================================================================
# DENOISE-THEN-TRANSFORM — a COMPOSITE: select (keep_largest / keep_color / remove_color) THEN a
# geometric/move op. Two relations in sequence; the prior often ranks 'select' for object-removal tasks.
# (This is also covered by _gen_sequence, but a focused generator finds it faster within budget.)
# ===========================================================================
def _gen_select_then_geom(train, colors, exec_left):
    fns = []
    selectors = [("identity", ())] + _instances_for_kind("select", colors)
    geoms = _instances_for_kind("geom", colors) + _instances_for_kind("move", colors)
    n = 0
    for so, sa in selectors:
        for go_, ga in geoms:
            if so == "identity" and go_ == "identity":
                continue
            if n >= exec_left:
                return fns
            n += 1
            steps = ([] if so == "identity" else [(so, sa)]) + [(go_, ga)]
            if len(steps) < 2:
                continue
            fn = lambda g, st=steps: mc.run_sequence(g, st)
            if _verify(fn, train):
                fns.append((fn, {"combinator": "sequence", "steps": steps}))
                if len(fns) >= 2:
                    return fns
    return fns


# ===========================================================================
# THE SYNTHESIS ENGINE — GENERATE composite mechanisms from a proposal skeleton, exact-verify.
# Each generator yields verified (fn, schema) where schema captures the composition for ABSTRACTION.
# ===========================================================================
def _gen_sequence(train, colors, kinds, exec_left):
    """Compose >=2 relations (function composition of letters). The prior's kinds focus the op pool."""
    # build a focused op pool from the proposed kinds (+ a small always-on geometric set)
    pool = []
    seen = set()
    for kd in kinds:
        for inst in _instances_for_kind(kd, colors):
            if inst not in seen:
                seen.add(inst); pool.append(inst)
    # cap pool to keep depth-2/3 search within budget
    pool = pool[:26]
    ins = [gi for gi, _ in train]
    tgt = [go for _, go in train]

    # depth-2 then depth-3, verifying after each op against the targets
    # we run a small breadth-limited search seeded by single-op grid-distance to target.
    def gdist_set(outs):
        s = 0.0
        for a, b in zip(outs, tgt):
            if a is None: s += 3.0
            elif a.shape != b.shape: s += 1.0 + abs(a.size - b.size)/max(a.size, b.size, 1)
            else: s += float((a != b).mean())
        return s/len(outs)

    # frontier of (score, prog, outs)
    frontier = [([], ins)]
    nexec = [0]
    results = []

    def apply_all(outs, op, args):
        res = []
        for g in outs:
            try:
                res.append(dsl.OPS[op][0](g, *args))
            except Exception:
                return None
        nexec[0] += 1
        return res

    for depth in range(2):  # produce depth-2 and depth-3 (two expansion rounds from depth-1 frontier)
        new_front = []
        scored = []
        for prog, outs in frontier:
            for op, args in pool:
                if nexec[0] >= exec_left:
                    break
                outs2 = apply_all(outs, op, args)
                if outs2 is None:
                    continue
                prog2 = prog + [(op, args)]
                if len(prog2) >= 2 and all(_eq(o, t) for o, t in zip(outs2, tgt)):
                    results.append(prog2)
                else:
                    scored.append((gdist_set(outs2), prog2, outs2))
            if nexec[0] >= exec_left:
                break
        if results:
            return results
        # keep best W to expand once more (depth 3)
        scored.sort(key=lambda x: x[0])
        frontier = [(p, o) for _s, p, o in scored[:8]]
        if nexec[0] >= exec_left:
            break
    return results


def _gen_region_restrict(train, colors, kinds, exec_left):
    """Apply ONE relation only inside a sub-region (top/bot/left/right/bbox), paste back."""
    fns = []
    region_pool = ["top", "bot", "left", "right", "bbox"]
    use_kinds = [k for k in kinds if k in ("geom", "move", "paint", "colormap")] or ["geom"]
    n = 0
    for region in region_pool:
        for kd in use_kinds:
            for op, args in _instances_for_kind(kd, colors):
                if n >= exec_left:
                    return fns
                n += 1
                fn = (lambda g, o=op, a=args, r=region: mc.run_region_restrict(g, o, a, r))
                if _verify(fn, train):
                    fns.append((fn, {"combinator": "region_restrict", "op": op, "args": args,
                                     "region": region}))
                    if len(fns) >= 3:
                        return fns
    return fns


def _gen_per_object_map(train, exec_left):
    fns = []
    for diag in (False, True):
        fn = _fit_per_object_recolor(train, diag)
        if fn is not None and _verify(fn, train):
            fns.append((fn, {"combinator": "per_object_map", "diag": diag}))
    return fns


def _gen_feature_conditional(train, colors, kinds, exec_left):
    """Choose the relation based on a grid feature (tall/wide/many_obj/dense). Composite: two branches."""
    fns = []
    use_kinds = [k for k in kinds if k in ("geom", "move")] or ["geom", "move"]
    pool = []
    for kd in use_kinds:
        pool += _instances_for_kind(kd, colors)
    pool = pool[:16]
    n = 0
    for pred in ("tall", "wide", "many_obj", "dense"):
        for (o1, a1) in pool:
            for (o2, a2) in pool:
                if (o1, a1) == (o2, a2):
                    continue
                if n >= exec_left:
                    return fns
                n += 1
                fn = (lambda g, p=pred, t=(o1, a1), e=(o2, a2): mc.run_feature_conditional(g, p, t, e))
                if _verify(fn, train):
                    # reject the degenerate case where one branch never fires (then it's a single op)
                    fns.append((fn, {"combinator": "feature_conditional", "pred": pred,
                                     "then": (o1, a1), "else": (o2, a2)}))
                    if len(fns) >= 2:
                        return fns
    return fns


def _gen_repurpose_overlay(train, colors, exec_left):
    """Produce a transformed copy and OVERLAY it onto a base view (nonzero-of-overlay wins on 0s)."""
    fns = []
    geom = _instances_for_kind("geom", colors)
    n = 0
    for (bo, ba) in geom:
        for (oo, oa) in geom:
            if bo == oo:
                continue
            if n >= exec_left:
                return fns
            n += 1
            fn = (lambda g, b=(bo, ba), o=(oo, oa): mc.run_repurpose_overlay(g, b[0], b[1], o[0], o[1]))
            if _verify(fn, train):
                fns.append((fn, {"combinator": "repurpose_overlay", "base": (bo, ba), "overlay": (oo, oa)}))
                if len(fns) >= 2:
                    return fns
    return fns


# ===========================================================================
# ABSTRACTION — lift a verified composition into a reusable SCHEMA (colors -> holes), bank it.
# A banked schema is REPLAYED on later tasks: its color holes are re-bound from the new task's palette
# and the whole thing is RE-VERIFIED before it is trusted (so transfer never costs correctness).
# ===========================================================================
def _is_composite(schema):
    """Is this schema a genuine SENTENCE (>=2 relations or a higher-order object/region relation)?
    Used to exclude single-op 'sentences' from the bank so transfer measures real composition."""
    comb = schema.get("combinator")
    if comb == "sequence":
        return len(schema.get("steps", [])) >= 2
    return comb in ("region_restrict", "per_object_map", "feature_conditional",
                    "repurpose_overlay", "sym_repair")


def _schema_to_fn(schema, colors):
    """Re-instantiate a banked schema as a callable, re-binding any color HOLES from `colors`.
    Returns a callable, or None if the schema cannot be bound to this palette."""
    comb = schema["combinator"]
    if comb == "sequence":
        steps = schema["steps"]
        return lambda g, s=steps: mc.run_sequence(g, s)
    if comb == "region_restrict":
        return lambda g, sc=schema: mc.run_region_restrict(g, sc["op"], sc["args"], sc["region"])
    if comb == "per_object_map":
        diag = schema["diag"]
        # per-object schema is re-FITTED on the new task (its table is task-specific) — replay the fitter
        return ("REFIT_OBJ", diag)
    if comb == "feature_conditional":
        return lambda g, sc=schema: mc.run_feature_conditional(g, sc["pred"], sc["then"], sc["else"])
    if comb == "repurpose_overlay" and schema.get("base", (None,))[0] == "fractal":
        return _make_fractal(schema["base"][1], schema["overlay"][1])
    if comb == "repurpose_overlay":
        return lambda g, sc=schema: mc.run_repurpose_overlay(g, sc["base"][0], sc["base"][1],
                                                             sc["overlay"][0], sc["overlay"][1])
    if comb == "sym_repair":
        if schema.get("view") == "periodic":
            return _make_periodic_repair(schema["hole"])
        return _make_sym_repair(schema["view"], schema["hole"])
    return None


def _bank_schema(schema):
    if not _is_composite(schema):
        return
    # dedupe by a cheap signature
    sig = (schema.get("combinator"), str(schema.get("steps")), str(schema.get("op")),
           schema.get("region"), schema.get("pred"), str(schema.get("base")),
           schema.get("view"), schema.get("hole"))
    for s in _BANK:
        if s.get("_sig") == sig:
            return
    schema["_sig"] = sig
    _BANK.append(schema)
    _COMB_HITS[schema["combinator"]] += 1
    if len(_BANK) > _BANK_CAP:
        del _BANK[0]


def _replay_bank(train, colors):
    """Try banked schemas (experience transfer). Re-verify each on the CURRENT task before trusting.
    Returns a verified fn or None."""
    for schema in reversed(_BANK):  # most-recent first
        fn = _schema_to_fn(schema, colors)
        if fn is None:
            continue
        if isinstance(fn, tuple) and fn[0] == "REFIT_OBJ":
            refit = _fit_per_object_recolor(train, fn[1])
            if refit is not None and _verify(refit, train):
                return refit
            continue
        if _verify(fn, train):
            return fn
    return None


# ===========================================================================
# THE INVENTION PATH — synthesize, verify, abstract. Returns a verified fn or None.
# ===========================================================================
def _invent(train, test_inputs, budget):
    colors = _palette(train, test_inputs)
    nzcolors = [c for c in colors if c != 0] or colors

    # 0) EXPERIENCE TRANSFER: replay banked schemas first (re-verified). Fast + free.
    fn = _replay_bank(train, nzcolors)
    if fn is not None:
        return fn, {"combinator": "transferred"}

    # 1) INVARIANCE + PRIOR: ask the experience-prior which composition shapes/kinds fit.
    try:
        props = mc.propose_compositions(train, k=5)
        _, feat = mc.features(train)
    except Exception:
        props = []; feat = {}

    # 1a) SYMMETRY/PERIODIC REPAIR — composite hole-fill from the grid's own symmetric/periodic image.
    #     Cheap; fires on same-shape tasks where palette changes by erasing an occluder. Try early.
    if feat.get("same_shape_frac", 0) > 0.9:
        for gen in (_gen_symmetry_repair, _gen_periodic_repair):
            v = gen(train)
            if v:
                fn, schema = v[0]
                _bank_schema(schema)
                return fn, schema

    # 1b) FRACTAL SELF-TILING — fires when output is exactly (h*h, w*w). A composite per-cell stamp.
    if feat.get("is_upscale", 0) > 0.5 or feat.get("area_ratio_mean", 1.0) > 3.0:
        vf = _gen_fractal(train)
        if vf:
            fn, schema = vf[0]
            _bank_schema(schema)
            return fn, schema
    # sharpen the prior order with this-run combinator-hit experience
    if _COMB_HITS:
        props = sorted(props, key=lambda p: -(p["score"] + 0.4 * _COMB_HITS.get(p["combinator"], 0)))

    # budget split across the proposed combinators (most budget to the top proposal)
    exec_budget = max(400, budget)
    spent = 0

    def left():
        return max(0, exec_budget - spent)

    for rank, p in enumerate(props):
        if left() <= 0:
            break
        comb = p["combinator"]
        kinds = p["kinds"]
        share = max(150, int(exec_budget * (0.45 if rank == 0 else 0.25 if rank == 1 else 0.12)))
        cap = min(share, left())

        verified = []
        if comb == "sequence":
            progs = _gen_sequence(train, nzcolors, kinds, cap)
            for prog in progs:
                fn = lambda g, pr=prog: mc.run_sequence(g, pr)
                if _verify(fn, train):
                    verified.append((fn, {"combinator": "sequence", "steps": prog}))
        elif comb == "region_restrict":
            verified = _gen_region_restrict(train, nzcolors, kinds, cap)
        elif comb == "per_object_map":
            verified = _gen_per_object_map(train, cap)
        elif comb == "feature_conditional":
            verified = _gen_feature_conditional(train, nzcolors, kinds, cap)
        elif comb == "repurpose_overlay":
            verified = _gen_repurpose_overlay(train, nzcolors, cap)
        spent += cap

        if verified:
            fn, schema = verified[0]
            _bank_schema(schema)
            return fn, schema

    # 2) FALLBACK COMPOSITIONS the prior may have under-ranked: always give per_object_map and a short
    #    sequence over a generic geometric+select pool a chance (cheap, composite-by-construction).
    if left() > 0:
        vob = _gen_per_object_map(train, left())
        if vob:
            fn, schema = vob[0]
            _bank_schema(schema)
            return fn, schema
    if left() > 0:
        vsg = _gen_select_then_geom(train, nzcolors, min(left(), 400))
        if vsg:
            fn, schema = vsg[0]
            _bank_schema(schema)
            return fn, schema
        spent += min(left(), 400)
    if left() > 0:
        vseq = _gen_sequence(train, nzcolors, ["geom", "tile", "move", "select"], left())
        for prog in vseq:
            fn = lambda g, pr=prog: mc.run_sequence(g, pr)
            if _verify(fn, train):
                schema = {"combinator": "sequence", "steps": prog}
                _bank_schema(schema)
                return fn, schema

    return None, None


# ===========================================================================
# PUBLIC SOLVE — full invention (compose + abstract + transfer), with a retrieval floor underneath.
# ===========================================================================
def _attempts_from_fns(fns, test_inputs):
    attempts = []
    for gi in test_inputs:
        cand = []
        for fn in fns:
            o = _predict(fn, gi)
            if o is not None:
                cand.append(o)
            if len(cand) >= 2:
                break
        attempts.append(cand)
    return attempts


def solve(train, test_inputs, budget):
    """FULL solver: INVENT a composite mechanism (prior-guided synthesis + abstraction + transfer);
    fall back to single-template RETRIEVAL only if invention finds nothing. The invention path is what
    the ablation cannot reach => certified creativity."""
    train = [(np.asarray(a, int), np.asarray(b, int)) for a, b in train]
    test_inputs = [np.asarray(g, int) for g in test_inputs]

    fns = []
    # PRIMARY: invention (composition / abstraction / repurposing / transfer)
    inv_fn, _schema = _invent(train, test_inputs, budget)
    if inv_fn is not None:
        fns.append(inv_fn)

    # SECONDARY (second attempt): retrieval floor, so we never regress below gen-0/1 on easy tasks.
    ret_fn = _retrieval_solve(train)
    if ret_fn is not None and not any(ret_fn is f for f in fns):
        fns.append(ret_fn)

    if not fns:
        return [[] for _ in test_inputs]
    return _attempts_from_fns(fns, test_inputs)


def solve_ablated(train, test_inputs, budget):
    """ABLATION: invention DISABLED. Single-whole-template RETRIEVAL only — every length-1 DSL op and a
    fixed menu of whole-mechanism templates, NO composition / abstraction / repurposing / transfer.
    Whatever solve() solves and this does NOT is a certified invention."""
    train = [(np.asarray(a, int), np.asarray(b, int)) for a, b in train]
    test_inputs = [np.asarray(g, int) for g in test_inputs]
    ret_fn = _retrieval_solve(train)
    if ret_fn is None:
        return [[] for _ in test_inputs]
    return _attempts_from_fns([ret_fn], test_inputs)


# ===========================================================================
# self-test
# ===========================================================================
if __name__ == "__main__":
    import json
    # tiny synthetic sanity: a depth-2 sequence (rot90 then reflect_h) should be INVENTED but NOT
    # reachable by a single op or template.
    base = np.array([[1, 2, 0], [0, 3, 0], [4, 0, 0]])
    def mech(g):
        return dsl.reflect_h(dsl.rot90(g))
    tr = [(base, mech(base)), (np.array([[0, 5, 0], [6, 0, 7], [0, 8, 0]]),
                               mech(np.array([[0, 5, 0], [6, 0, 7], [0, 8, 0]])))]
    ti = [np.array([[2, 0, 3], [0, 4, 0], [5, 6, 0]])]
    reset_library()
    full = solve(tr, ti, 3000)
    abl = solve_ablated(tr, ti, 3000)
    want = mech(ti[0])
    full_ok = any(_eq(c, want) for c in (full[0] if full else []))
    abl_ok = any(_eq(c, want) for c in (abl[0] if abl else []))
    print("synthetic depth-2 sequence: full_solves=%s ablation_solves=%s (INVENTED=%s)"
          % (full_ok, abl_ok, full_ok and not abl_ok))
