#!/usr/bin/env python3
"""Gen-2 CONSOLIDATED BASE solver for the DIY-AlphaEvolve ARC-AGI-1 campaign.

WHAT THIS IS. The strongest gen-1 generalizing FLOOR, merged. It takes gen1_05 (param-struct,
30 dev / 28 eval) as the SPINE — its parametric/structural concept bank, its seed DSL fallback,
its in-session concept-order experience prior — and GRAFTS the distinct generalizing concept
families from the other five gen-1 candidates that solved dev-UNION tasks the spine misses:

  GRAFTED FAMILIES (each verified to flip a specific union task the spine missed):
    * local-rule / local-rule-plain   (from exp-library): output[i,j] = f(3x3 neighborhood),
                                       as a learned lookup table, dihedral-INVARIANT and oriented.
                                       -> flips 3618c87e 4258a5f9 50cb2852 54d9e175 6f8cd79b
    * object-recolor-by-size/rank      (from concept-induct + exp + linker): recolor each 4/8-conn
                                       component solid by its size or size-rank, learned from train.
                                       -> flips 08ed6ac7 67385a82 6e82a1ae
    * crop-window family               (from concept-induct): fixed margins / content bbox /
                                       largest|smallest OBJECT bbox (window or cut), 4&8-conn,
                                       by-color & color-agnostic. -> flips 1f85a75f 5bd6f4ac 23b5c85d
    * connect-the-dots                 (from baked-proposer): same-color collinear pairs joined by a
                                       line of that color (row & col rays). -> flips 22168020 22eb0ac0
    * row/col dedup                    (from baked-proposer): collapse consecutive duplicate rows/cols.
                                       -> flips 746b3537

  CREATIVITY MACHINERY (kept WIRED for gen-2 to make pay, even where it read ~0 last gen):
    * EXPERIENCE LIBRARY (_LIB): concept-hit counts re-order which concepts are tried first on
      later tasks (experience prior), AND a store of standalone verified concept closures that are
      REPLAYED (transferred) on later tasks after re-verify.
    * ANTIUNIFICATION MACROS: recurring arg-free DSL op-subsequences mined from verified search
      programs become length-1 "super-ops" spliced into later search (compression / reuse).
    * NOVEL LINKING: compose a REMEMBERED concept with a FRESH-induced concept (both orders),
      and compose a fitted concept AFTER a cheap geometric pre-op; only exact-verified links survive.
    * SELF-GEN CURRICULUM (import time, <~2s): a light op-relevance ranker trained on synthetic
      (input,output,op) transitions re-orders the DSL search op vocabulary (baked-from-data prior).
      make_curriculum() is exported so later generations can train bigger models.

  MDL ORDERING: cheapest/simplest concepts and shortest programs first; an experience discount
  promotes concepts/ops that have already paid off this run. Up to 2 attempts/test input, best-first.

INTEGRITY (hard rules): solve() learns ONLY from (a) the current task's train pairs, (b) module-level
state accumulated from PRIOR solve() calls this run (verified-correct only), (c) self-generated
synthetic data built at import. It NEVER reads any ARC task file or test OUTPUT, no network, no LLM at
solve time. Respects budget. Pure python + numpy. Run/imported with /data/llm/.venv/bin/python."""
import sys, heapq, time
from collections import deque, Counter, defaultdict
import numpy as np

ARC = "/data/Windows-files/Documents/airfoil/incubation/arc"
if ARC not in sys.path:
    sys.path.insert(0, ARC)
import dsl

META = {"name": "gen2_base_v1",
        "desc": "consolidated UNION floor: param-struct spine + grafted local-rule / object-recolor / "
                "crop-window / connect-dots / dedup families; experience library + antiunification macros "
                "+ novel linking + self-gen curriculum ranker; seed DSL fallback"}


# ===========================================================================
# IN-SESSION EXPERIENCE LIBRARY (module-level; persists across solve() calls in a run).
#   concept_hits : name -> #times that concept produced a VERIFIED rule (an experience prior on order)
#   closures     : standalone verified (tag, fn) concepts, REPLAYED/transferred onto later tasks
#   macros       : recurring arg-free DSL op-subsequences mined from verified programs (super-ops)
#   op_hits      : DSL op name -> #verified programs using it (re-orders search op vocab)
# Only verified-correct info ever enters the library; no grids/outputs/files are stored.
# ===========================================================================
class _Library:
    def __init__(self):
        self.concept_hits = Counter()
        self.closures = []          # list of (tag, fn) replayable concept closures
        self.closure_tags = set()
        self.macro_src = {}         # op-name subsequence -> set(task_id)
        self.macros = {}            # macro_name -> op-name subsequence (arg-free only)
        self.op_hits = Counter()
        self.solved_progs = []
        self.audit = []             # (task_id, fired_label, was_library_reuse)

    def bump(self, name):
        self.concept_hits[name] += 1

    def remember_closure(self, tag, fn):
        if tag not in self.closure_tags:
            self.closure_tags.add(tag)
            self.closures.append((tag, fn))

    def ingest_program(self, task_id, prog):
        """prog: list of (op_name, args). Mine recurring arg-free op subsequences -> macros (MDL razor:
        promote a subsequence only once it recurs across >=2 distinct tasks)."""
        self.solved_progs.append((task_id, prog))
        names = [op for op, _ in prog]
        if len(names) < 2:
            return
        seqs = set()
        for L in range(2, len(names) + 1):
            for s in range(0, len(names) - L + 1):
                seqs.add(tuple(names[s:s + L]))
        for seq in seqs:
            self.macro_src.setdefault(seq, set()).add(task_id)
        for seq, srcs in self.macro_src.items():
            if len(srcs) >= 2 and all(seq_op in dsl.OPS and dsl.OPS[seq_op][1] == 0 for seq_op in seq):
                self.macros["MACRO[" + ">".join(seq) + "]"] = seq

    def macro_ops(self):
        out = []
        for mname, seq in self.macros.items():
            def make(seq=seq):
                def fn(g):
                    for op in seq:
                        g = dsl.OPS[op][0](g)
                    return g
                return fn
            out.append((mname, make()))
        return out


_LIB = _Library()


# ===========================================================================
# small grid helpers
# ===========================================================================
def _eq(a, b):
    return a is not None and getattr(a, "shape", None) == b.shape and np.array_equal(a, b)


def _bg_color(grids):
    c = Counter()
    for g in grids:
        v, ct = np.unique(g, return_counts=True)
        for vi, ci in zip(v, ct):
            c[int(vi)] += int(ci)
    return c.most_common(1)[0][0] if c else 0


def _bg(g):
    v, ct = np.unique(g, return_counts=True)
    return int(v[ct.argmax()])


def _verify(fn, train):
    """True iff fn reproduces EVERY train output exactly (and runs without error)."""
    for gi, go in train:
        try:
            o = fn(gi)
        except Exception:
            return False
        if not _eq(o, go):
            return False
    return True


def _components(g, bg=0, diag=False, by_color=False):
    h, w = g.shape
    seen = np.zeros((h, w), bool)
    if diag:
        nb = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        nb = [(-1, 0), (1, 0), (0, -1), (0, 1)]
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


# ###########################################################################
# SPINE CONCEPTS  (verbatim families from gen1_05 param-struct)
# ###########################################################################

# --- CONCEPT: color permutation (bijection fitted from train cells) ---
def fit_color_perm(train):
    if any(gi.shape != go.shape for gi, go in train):
        return None
    mp = {}
    for gi, go in train:
        a = gi.reshape(-1); b = go.reshape(-1)
        for x, y in zip(a.tolist(), b.tolist()):
            if x in mp and mp[x] != y:
                return None
            mp[x] = y
    if all(k == v for k, v in mp.items()):
        return None

    def fn(g):
        out = g.copy()
        for k, v in mp.items():
            if k != v:
                out[g == k] = v
        return out
    return fn


# --- CONCEPT: panel split + inter-panel logic ---
def _separators(g):
    out = []
    h, w = g.shape
    for ax in (0, 1):
        n = g.shape[ax]
        for c in np.unique(g):
            line = np.all(g == c, axis=1 - ax)
            idx = np.where(line)[0]
            if 0 < len(idx) < n:
                out.append((ax, int(c), idx))
    return out


def _split_panels(g):
    cands = []
    h, w = g.shape
    for ax, c, idx in _separators(g):
        segs = []
        prev = 0
        bounds = list(idx) + [g.shape[ax]]
        for i in bounds:
            if i > prev:
                seg = g[prev:i, :] if ax == 0 else g[:, prev:i]
                segs.append(seg)
            prev = i + 1
        shapes = set(s.shape for s in segs)
        if len(segs) >= 2 and len(shapes) == 1:
            cands.append((segs, ax, c))
    if w % 2 == 0:
        cands.append(([g[:, :w // 2], g[:, w // 2:]], 1, None))
    if h % 2 == 0:
        cands.append(([g[:h // 2, :], g[h // 2:, :]], 0, None))
    if w % 3 == 0:
        cands.append(([g[:, :w // 3], g[:, w // 3:2 * w // 3], g[:, 2 * w // 3:]], 1, None))
    if h % 3 == 0:
        cands.append(([g[:h // 3, :], g[h // 3:2 * h // 3, :], g[2 * h // 3:, :]], 0, None))
    return cands


def _combine(panels, mode, bg):
    masks = [(p != bg) for p in panels]
    if mode == "and":
        m = masks[0].copy()
        for x in masks[1:]:
            m &= x
    elif mode == "or":
        m = masks[0].copy()
        for x in masks[1:]:
            m |= x
    elif mode == "xor":
        m = masks[0].copy()
        for x in masks[1:]:
            m ^= x
    elif mode == "diff":
        m = masks[0].copy()
        for x in masks[1:]:
            m &= ~x
    elif mode == "nand":
        m = masks[0].copy()
        for x in masks[1:]:
            m &= x
        m = ~m
    elif mode == "nor":
        m = masks[0].copy()
        for x in masks[1:]:
            m |= x
        m = ~m
    else:
        return None
    return m


def fit_panel_logic(train):
    if not train:
        return None
    bg = _bg_color([gi for gi, _ in train])
    strategies = []
    g0 = train[0][0]
    for si, (panels0, ax0, sep0) in enumerate(_split_panels(g0)):
        pshape = panels0[0].shape
        npan = len(panels0)
        if pshape != train[0][1].shape:
            continue
        ok = True
        per_train = []
        for gi, go in train:
            found = None
            for panels, ax, sep in _split_panels(gi):
                if len(panels) == npan and panels[0].shape == go.shape and ax == ax0 and sep == sep0:
                    found = panels
                    break
            if found is None:
                ok = False
                break
            per_train.append(found)
        if ok:
            strategies.append(per_train)
    for per_train in strategies:
        for mode in ("and", "or", "xor", "diff", "nand", "nor"):
            on_color = None
            off_color = None
            good = True
            for (gi, go), panels in zip(train, per_train):
                m = _combine(panels, mode, bg)
                if m is None or m.shape != go.shape:
                    good = False
                    break
                on_vals = set(np.unique(go[m]).tolist()) if m.any() else set()
                off_vals = set(np.unique(go[~m]).tolist()) if (~m).any() else set()
                if len(on_vals) > 1 or len(off_vals) > 1:
                    good = False
                    break
                oc = next(iter(on_vals)) if on_vals else None
                fc = next(iter(off_vals)) if off_vals else None
                if oc is not None:
                    if on_color is None:
                        on_color = oc
                    elif on_color != oc:
                        good = False
                        break
                if fc is not None:
                    if off_color is None:
                        off_color = fc
                    elif off_color != fc:
                        good = False
                        break
            if not good or on_color is None:
                continue
            if off_color is None:
                off_color = 0
            oc_, fc_, mode_, npan_, ax_, sep_ = on_color, off_color, mode, len(per_train[0]), None, None
            for panels, ax, sep in _split_panels(train[0][0]):
                if len(panels) == npan_ and panels[0].shape == train[0][1].shape:
                    ax_, sep_ = ax, sep
                    break

            def make(oc=oc_, fc=fc_, mode=mode_, npan=npan_, ax=ax_, sep=sep_, bg=bg):
                def fn(g):
                    chosen = None
                    for panels, a, s in _split_panels(g):
                        if len(panels) == npan and a == ax and s == sep:
                            chosen = panels
                            break
                    if chosen is None:
                        return None
                    m = _combine(chosen, mode, bg)
                    if m is None:
                        return None
                    out = np.full(m.shape, fc, int)
                    out[m] = oc
                    return out
                return fn
            fn = make()
            if _verify(fn, train):
                return fn
    return None


# --- CONCEPT: symmetric tiling (mosaic of fitted per-block transforms) ---
_TRANSFORMS = {
    "id":   lambda g: g,
    "mh":   lambda g: g[:, ::-1],
    "mv":   lambda g: g[::-1, :],
    "r180": lambda g: np.rot90(g, 2),
    "r90":  lambda g: np.rot90(g, 1),
    "r270": lambda g: np.rot90(g, 3),
    "tp":   lambda g: g.T,
    "tp2":  lambda g: g[::-1, ::-1].T,
}


def fit_symmetric_tiling(train):
    gi0, go0 = train[0]
    hi, wi = gi0.shape
    ho, wo = go0.shape
    if hi == 0 or wi == 0 or ho % hi or wo % wi:
        return None
    kr, kc = ho // hi, wo // wi
    if kr * kc < 2 or kr > 4 or kc > 4:
        return None
    square = (hi == wi)
    names = list(_TRANSFORMS.keys()) if square else ["id", "mh", "mv", "r180"]
    for gi, go in train:
        if gi.shape[0] == 0 or gi.shape[1] == 0:
            return None
        if go.shape[0] != kr * gi.shape[0] or go.shape[1] != kc * gi.shape[1]:
            return None
    grid_choice = [[None] * kc for _ in range(kr)]
    for bi in range(kr):
        for bj in range(kc):
            chosen = None
            for nm in names:
                ok = True
                for gi, go in train:
                    h, w = gi.shape
                    block = go[bi * h:(bi + 1) * h, bj * w:(bj + 1) * w]
                    t = _TRANSFORMS[nm](gi)
                    if t.shape != block.shape or not np.array_equal(t, block):
                        ok = False
                        break
                if ok:
                    chosen = nm
                    break
            if chosen is None:
                return None
            grid_choice[bi][bj] = chosen

    def fn(g, gc=grid_choice, kr=kr, kc=kc):
        rows = []
        for bi in range(kr):
            row = [_TRANSFORMS[gc[bi][bj]](g) for bj in range(kc)]
            rows.append(np.concatenate(row, axis=1))
        return np.concatenate(rows, axis=0)
    return fn


# --- CONCEPT: fractal self-tiling ---
def fit_fractal(train):
    gi0, go0 = train[0]
    hi, wi = gi0.shape
    ho, wo = go0.shape
    if hi == 0 or wi == 0 or ho != hi * hi or wo != wi * wi:
        return None
    bg = _bg_color([gi for gi, _ in train])
    for invert in (False, True):
        def fn(g, bg=bg, invert=invert):
            h, w = g.shape
            out = np.full((h * h, w * w), bg, int)
            for i in range(h):
                for j in range(w):
                    lit = (g[i, j] != bg)
                    if invert:
                        lit = not lit
                    if lit:
                        out[i * h:(i + 1) * h, j * w:(j + 1) * w] = g
            return out
        if _verify(fn, train):
            return fn
    return None


# --- CONCEPT: periodic / symmetry repair (occlusion fill) ---
def _detect_occluder(train):
    cand = None
    for gi, go in train:
        if gi.shape != go.shape:
            return None
        diff = (gi != go)
        if not diff.any():
            continue
        holes = set(np.unique(gi[diff]).tolist())
        if len(holes) != 1:
            return None
        h = next(iter(holes))
        if cand is None:
            cand = h
        elif cand != h:
            return None
    return cand


def _fill_periodic(g, hole):
    h, w = g.shape
    known = (g != hole)
    if known.all():
        return g.copy()

    def row_period():
        for p in range(1, h):
            ok = True
            for i in range(h - p):
                for j in range(w):
                    if known[i, j] and known[i + p, j] and g[i, j] != g[i + p, j]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return p
        return h

    def col_period():
        for p in range(1, w):
            ok = True
            for j in range(w - p):
                for i in range(h):
                    if known[i, j] and known[i, j + p] and g[i, j] != g[i, j + p]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return p
        return w

    pr = row_period()
    pc = col_period()
    out = g.copy()
    for i in range(h):
        for j in range(w):
            if known[i, j]:
                continue
            val = None
            for ii in range(i % pr, h, pr):
                for jj in range(j % pc, w, pc):
                    if known[ii, jj]:
                        val = g[ii, jj]
                        break
                if val is not None:
                    break
            if val is None:
                for d in range(1, max(h, w)):
                    for (di, dj) in ((d, d), (-d, -d), (d, -d), (-d, d)):
                        ii, jj = i + di, j + dj
                        if 0 <= ii < h and 0 <= jj < w and known[ii, jj]:
                            val = g[ii, jj]
                            break
                    if val is not None:
                        break
            if val is not None:
                out[i, j] = val
    return out


def _fill_diagonal(g, hole):
    h, w = g.shape
    known = (g != hole)
    out = g.copy()
    for axis in ("main", "anti"):
        key = {}
        for i in range(h):
            for j in range(w):
                if known[i, j]:
                    k = (i - j) if axis == "main" else (i + j)
                    key.setdefault(k % max(h, w), set()).add(int(g[i, j]))
        if all(len(v) <= 1 for v in key.values()):
            tmp = g.copy()
            for i in range(h):
                for j in range(w):
                    if not known[i, j]:
                        k = (i - j) if axis == "main" else (i + j)
                        vs = key.get(k % max(h, w))
                        if vs:
                            tmp[i, j] = next(iter(vs))
            if (tmp != hole).all():
                return tmp
    return out


def fit_periodic_repair(train):
    hole = _detect_occluder(train)
    if hole is None:
        return None

    def fn_full(g, hole=hole):
        o = _fill_periodic(g, hole)
        if (o == hole).any():
            o2 = _fill_diagonal(g, hole)
            if not (o2 == hole).any():
                return o2
        return o
    if _verify(fn_full, train):
        return fn_full

    def fn_crop(g, hole=hole):
        nz = np.argwhere(g == hole)
        if nz.size == 0:
            return None
        (r0, c0), (r1, c1) = nz.min(0), nz.max(0) + 1
        full = _fill_periodic(g, hole)
        return full[r0:r1, c0:c1]
    if _verify(fn_crop, train):
        return fn_crop
    return None


def fit_symmetry_repair(train):
    hole = _detect_occluder(train)
    if hole is None:
        return None
    syms = [
        lambda g: g[:, ::-1],
        lambda g: g[::-1, :],
        lambda g: g[::-1, ::-1],
        lambda g: g.T if g.shape[0] == g.shape[1] else None,
        lambda g: np.rot90(g, 1) if g.shape[0] == g.shape[1] else None,
    ]

    def repair(g, hole=hole):
        out = g.copy()
        for _ in range(6):
            changed = False
            for s in syms:
                m = s(out)
                if m is None or m.shape != out.shape:
                    continue
                fillable = (out == hole) & (m != hole)
                if fillable.any():
                    out[fillable] = m[fillable]
                    changed = True
            if not changed:
                break
        return out

    def fn_full(g):
        return repair(g)
    if _verify(fn_full, train):
        return fn_full

    def fn_crop(g, hole=hole):
        nz = np.argwhere(g == hole)
        if nz.size == 0:
            return None
        (r0, c0), (r1, c1) = nz.min(0), nz.max(0) + 1
        return repair(g)[r0:r1, c0:c1]
    if _verify(fn_crop, train):
        return fn_crop
    return None


# --- CONCEPT: scale by integer ratio (nearest-neighbour upscale) ---
def fit_scale_ratio(train):
    gi0, go0 = train[0]
    hi, wi = gi0.shape
    ho, wo = go0.shape
    if hi == 0 or wi == 0 or ho % hi or wo % wi:
        return None
    rh, rw = ho // hi, wo // wi
    if rh == 1 and rw == 1:
        return None
    if max(rh, rw) > 10:
        return None

    def fn(g, rh=rh, rw=rw):
        return np.kron(g, np.ones((rh, rw), int))
    return fn


# --- CONCEPT: periodic tiling of input to larger output ---
def fit_periodic_tiling(train):
    gi0, go0 = train[0]
    hi, wi = gi0.shape
    ho, wo = go0.shape
    if hi == 0 or wi == 0 or ho % hi or wo % wi:
        return None
    rh, rw = ho // hi, wo // wi
    if rh * rw < 2:
        return None

    def fn(g, rh=rh, rw=rw):
        return np.tile(g, (rh, rw))
    return fn


# ###########################################################################
# GRAFTED CONCEPTS  (distinct generalizing families from the other 5 candidates)
# ###########################################################################

# --- GRAFT: local-rule (3x3 neighborhood lookup), dihedral-INVARIANT (from exp-library) ---
def _patch_mats(a, k=1):
    h, w = a.shape
    pad = np.full((h + 2 * k, w + 2 * k), -1, int)
    pad[k:k + h, k:k + w] = a
    return pad


def _canon8(m):
    best = None
    r = m
    for _ in range(4):
        for t in (r, r[:, ::-1]):
            key = t.tobytes()
            if best is None or key < best:
                best = key
        r = np.rot90(r)
    return best


def fit_local_rule(train):
    """output[i,j] = f(canonicalized 3x3 neighborhood of input). Rotation/reflection invariant -> fewer
    effective params, generalizes to unseen-orientation patches. Unknown patches copy input."""
    if any(a.shape != b.shape for a, b in train):
        return None
    mp = {}
    for a, b in train:
        pad = _patch_mats(a, 1)
        h, w = a.shape
        for i in range(h):
            for j in range(w):
                key = _canon8(pad[i:i + 3, j:j + 3])
                y = int(b[i, j])
                if key in mp and mp[key] != y:
                    return None
                mp[key] = y

    def fn(a, mp=mp):
        out = a.copy()
        pad = _patch_mats(a, 1)
        h, w = a.shape
        for i in range(h):
            for j in range(w):
                key = _canon8(pad[i:i + 3, j:j + 3])
                if key in mp:
                    out[i, j] = mp[key]
        return out
    return fn


def fit_local_rule_plain(train):
    """Oriented (no symmetry canonicalization) 3x3 lookup. Catches direction-sensitive local rules."""
    if any(a.shape != b.shape for a, b in train):
        return None
    mp = {}
    for a, b in train:
        h, w = a.shape
        pad = np.full((h + 2, w + 2), -1, int); pad[1:1 + h, 1:1 + w] = a
        for i in range(h):
            for j in range(w):
                key = pad[i:i + 3, j:j + 3].tobytes()
                y = int(b[i, j])
                if key in mp and mp[key] != y:
                    return None
                mp[key] = y

    def fn(a, mp=mp):
        h, w = a.shape
        out = a.copy()
        pad = np.full((h + 2, w + 2), -1, int); pad[1:1 + h, 1:1 + w] = a
        for i in range(h):
            for j in range(w):
                key = pad[i:i + 3, j:j + 3].tobytes()
                if key in mp:
                    out[i, j] = mp[key]
        return out
    return fn


# --- GRAFT: object-recolor by size / size-rank (from concept-induct + exp + linker) ---
def fit_object_recolor(train):
    """Recolor each 4/8-conn object solid by a key (cell-count size, or its asc/desc size rank) ->
    output color, learned consistently across pairs. Footprint preserved; bg unchanged."""
    if not all(i.shape == o.shape for i, o in train):
        return None
    # bgmode 0 = fixed background color 0 (objects = nonzero); "mc" = most-common color as background.
    for bgmode in (0, "mc"):
        for diag in (True, False):
            rules = {}
            for keytype in ("size", "rank_desc", "rank_asc"):
                mapping = {}
                ok = True
                for i, o in train:
                    bg = 0 if bgmode == 0 else _bg(i)
                    if np.any((i != bg) != (o != bg)):  # footprint preserved w.r.t. this background
                        ok = False
                        break
                    comps = _components(i, bg=bg, diag=diag)
                    if not comps:
                        ok = False
                        break
                    sizes = sorted({len(c) for c in comps})
                    for comp in comps:
                        ocolors = {int(o[a, b]) for a, b in comp}
                        if len(ocolors) != 1:
                            ok = False
                            break
                        oc = ocolors.pop()
                        if keytype == "size":
                            key = len(comp)
                        elif keytype == "rank_asc":
                            key = sizes.index(len(comp))
                        else:
                            key = len(sizes) - 1 - sizes.index(len(comp))
                        if key in mapping and mapping[key] != oc:
                            ok = False
                            break
                        mapping[key] = oc
                    if not ok:
                        break
                if ok and mapping:
                    rules[keytype] = mapping
            for keytype in ("size", "rank_desc", "rank_asc"):
                if keytype not in rules:
                    continue
                mapping = rules[keytype]

                def fn(g, _kt=keytype, _map=mapping, _diag=diag, _bgmode=bgmode):
                    bg = 0 if _bgmode == 0 else _bg(g)
                    comps = _components(g, bg=bg, diag=_diag)
                    if not comps:
                        return None
                    sizes = sorted({len(c) for c in comps})
                    out = g.copy()
                    for comp in comps:
                        if _kt == "size":
                            key = len(comp)
                        elif _kt == "rank_asc":
                            key = sizes.index(len(comp))
                        else:
                            key = len(sizes) - 1 - sizes.index(len(comp))
                        if key not in _map:
                            return None
                        for a, b in comp:
                            out[a, b] = _map[key]
                    return out
                if _verify(fn, train):
                    return fn
    return None


# --- GRAFT: crop-window family (from concept-induct, plus by-color object crop from obj-rel) ---
def _crop_rules(train):
    i0, o0 = train[0]

    def margins(i, o):
        H, W = i.shape; h, w = o.shape
        if h > H or w > W:
            return None
        for r in range(H - h + 1):
            for c in range(W - w + 1):
                if np.array_equal(i[r:r + h, c:c + w], o):
                    return (r, c, H - (r + h), W - (c + w))
        return None
    m0 = margins(i0, o0)
    if m0 is not None:
        t, l, b, r = m0

        def fn_m(g, _t=t, _l=l, _b=b, _r=r):
            H, W = g.shape
            if _t + _b >= H or _l + _r >= W:
                return None
            return g[_t:H - _b, _l:W - _r]
        yield ("margin", fn_m)

    def fn_bbox(g):
        bg = _bg(g)
        nz = np.argwhere(g != bg)
        if nz.size == 0:
            return None
        (r0, c0), (r1, c1) = nz.min(0), nz.max(0) + 1
        return g[r0:r1, c0:c1]
    yield ("content_bbox", fn_bbox)

    def _obj_bbox(g, which, window, diag, by_color):
        bg = _bg(g)
        comps = _components(g, bg=bg, diag=diag, by_color=by_color)
        if not comps:
            return None
        comp = max(comps, key=len) if which == "max" else min(comps, key=len)
        rs = [a for a, _ in comp]; cs = [b for _, b in comp]
        r0, r1, c0, c1 = min(rs), max(rs) + 1, min(cs), max(cs) + 1
        if window:
            return g[r0:r1, c0:c1]
        out = np.full((r1 - r0, c1 - c0), bg, int)
        for a, b in comp:
            out[a - r0, b - c0] = g[a, b]
        return out
    # orthogonal (4-conn) before diagonal (8-conn): 8-conn merges touching objects and tends to OVERFIT;
    # 4-conn generalizes more often, so we prefer it as the first train-fitting crop rule.
    for which in ("max", "min"):
        for window in (True, False):
            for diag in (False, True):
                for by_color in (False, True):
                    yield (f"obj_{which}_{'win' if window else 'cut'}_{'d' if diag else 'o'}_{'c' if by_color else 'a'}",
                           lambda g, _w=which, _wd=window, _dg=diag, _bc=by_color: _obj_bbox(g, _w, _wd, _dg, _bc))


def fit_crop(train):
    """Return up to 2 DISTINCT train-fitting crop functions. Many crop rules fit the train pairs but
    disambiguate only on test (e.g. 4-conn vs 8-conn object selection); returning two lets attempt-1 and
    attempt-2 cover the disambiguation instead of greedily committing to a possibly-overfit first match."""
    fns = []
    test_in = [i for i, _ in train]
    sigs = []
    for tag, fn in _crop_rules(train):
        try:
            if not all(_eq(fn(i), o) for i, o in train):
                continue
            # behavioral signature on train inputs (to dedup rules that act identically here)
            sig = tuple(fn(i).tobytes() for i in test_in)
        except Exception:
            continue
        if sig in sigs:
            continue
        sigs.append(sig)
        fns.append(fn)
        if len(fns) >= 2:
            break
    if not fns:
        return None
    return fns if len(fns) > 1 else fns[0]


# --- GRAFT: connect-the-dots (from baked-proposer) ---
def fit_connect_dots(train):
    if any(gi.shape != go.shape for gi, go in train):
        return None

    def fn(g):
        out = g.copy()
        for c in np.unique(g):
            if c == 0:
                continue
            pts = np.argwhere(g == c)
            byr = defaultdict(list); byc = defaultdict(list)
            for r, cc in pts:
                byr[r].append(cc); byc[cc].append(r)
            for r, cols in byr.items():
                cols = sorted(cols)
                for a, b in zip(cols, cols[1:]):
                    out[r, a:b + 1] = c
            for cc, rows in byc.items():
                rows = sorted(rows)
                for a, b in zip(rows, rows[1:]):
                    out[a:b + 1, cc] = c
        return out
    if _verify(fn, train):
        return fn
    return None


# --- GRAFT: row/col dedup (from baked-proposer) ---
def fit_dedup(train):
    for gi, go in train:
        if go.shape == gi.shape or go.shape[0] > gi.shape[0] or go.shape[1] > gi.shape[1]:
            return None

    def fn(g):
        if g.shape[0] == 0 or g.shape[1] == 0:
            return None
        rows = [g[0]]
        for r in g[1:]:
            if not np.array_equal(r, rows[-1]):
                rows.append(r)
        g2 = np.array(rows)
        cols = [g2[:, 0]]
        for c in range(1, g2.shape[1]):
            if not np.array_equal(g2[:, c], cols[-1]):
                cols.append(g2[:, c])
        return np.array(cols).T
    if _verify(fn, train):
        return fn
    return None


# --- GRAFT: symmetry overlay completion (no-occluder variant; fills bg=0 cells from mirrors) ---
def fit_sym_overlay(train):
    if any(gi.shape != go.shape for gi, go in train):
        return None

    def fn(g):
        out = g.copy()
        mirs = [g[:, ::-1], g[::-1, :], g[::-1, ::-1]]
        if g.shape[0] == g.shape[1]:
            mirs += [g.T]
        for m in mirs:
            z = out == 0
            out[z] = m[z]
        return out
    if _verify(fn, train):
        return fn
    return None


# ---------------------------------------------------------------------------
# CONCEPT REGISTRY (the store) — name -> fitter(train) -> fn or None.
# Spine concepts first (cheap structural), then grafts, MDL-ish order.
# ---------------------------------------------------------------------------
CONCEPTS = [
    # --- spine ---
    ("color_perm", fit_color_perm),
    ("scale_ratio", fit_scale_ratio),
    ("periodic_tiling", fit_periodic_tiling),
    ("symmetric_tiling", fit_symmetric_tiling),
    ("fractal", fit_fractal),
    ("panel_logic", fit_panel_logic),
    ("periodic_repair", fit_periodic_repair),
    ("symmetry_repair", fit_symmetry_repair),
    # --- grafts ---
    ("crop", fit_crop),
    ("dedup", fit_dedup),
    ("connect_dots", fit_connect_dots),
    ("object_recolor", fit_object_recolor),
    ("sym_overlay", fit_sym_overlay),
    ("local_rule", fit_local_rule),
    ("local_rule_plain", fit_local_rule_plain),
]

# cheap geometric pre-ops for the LINKER (functional repurposing of DSL geometry before a fitted concept)
_PRE = [
    ("none", lambda g: g),
    ("crop_content", dsl.crop_content),
    ("reflect_h", dsl.reflect_h),
    ("reflect_v", dsl.reflect_v),
    ("rot90", dsl.rot90),
    ("transpose", dsl.transpose),
]


def _concept_order():
    """Experience prior: concepts that verified before this run go first (MDL stays the tiebreak)."""
    hits = _LIB.concept_hits
    return sorted(CONCEPTS, key=lambda nc: -hits.get(nc[0], 0))


def _try_concepts(train):
    """Verified rule-fns from: (1) direct concept fit, (2) REPLAYED library closures (transfer),
    (3) LINKER compositions [geom pre-op -> concept] and [remembered concept <-> fresh concept].
    Returns list of (tag, fn) best-first by experience prior then registry order."""
    rules = []
    fresh = []  # standalone fresh concept fns, for novel linking + library transfer
    # NOTE: every verifying concept reproduces the train outputs EXACTLY (that is the definition of
    # verify), so concepts are NOT deduped by train behavior — they only differ on the held-out TEST,
    # which the attempt-build loop dedups via _eq. Deduping rules by train output would collapse every
    # correct concept into one and is a bug we explicitly avoid.

    # (1) direct fit of each concept on the raw train pairs (a fitter may return one fn OR a list of fns)
    for name, fitter in _concept_order():
        try:
            res = fitter(train)
        except Exception:
            res = None
        if res is None:
            continue
        fns = res if isinstance(res, list) else [res]
        for k, fn in enumerate(fns):
            if fn is not None and _verify(fn, train):
                tag = name if k == 0 else "%s#%d" % (name, k)
                rules.append((tag, fn))
                fresh.append((tag, fn))

    # (2) REPLAY library closures (transfer from prior solved tasks), re-verified on this task
    if _LIB.closures:
        have = {_base_name(t) for t, _ in rules}
        for tag, fn in _LIB.closures:
            if _base_name(tag) in have:
                continue
            if _verify(fn, train):
                rules.append(("lib:" + tag, fn))

    # (3a) NOVEL LINKING: compose a REMEMBERED concept with a FRESH-induced concept (both orders).
    # Skip self-links (a concept composed with itself that still verifies is just the idempotent concept).
    if _LIB.closures and fresh:
        for ltag, lf in _LIB.closures[:40]:
            for ftag, ff in fresh:
                if _base_name(ltag) == _base_name(ftag):
                    continue

                def c1(g, _a=lf, _b=ff):
                    x = _a(g)
                    return None if x is None else _b(x)

                def c2(g, _a=ff, _b=lf):
                    x = _a(g)
                    return None if x is None else _b(x)
                if _verify(c1, train):
                    rules.append(("link:%s>%s" % (ltag, ftag), c1))
                if _verify(c2, train):
                    rules.append(("link:%s>%s" % (ftag, ltag), c2))

    # (3b) LINKER: geometric pre-op then a fitted concept (only if nothing direct fired)
    if not rules:
        for pname, pre in _PRE[1:]:
            try:
                pre_train = [(pre(gi), go) for gi, go in train]
            except Exception:
                continue
            for name, fitter in _concept_order():
                try:
                    fn = fitter(pre_train)
                except Exception:
                    fn = None
                if fn is None:
                    continue

                def composed(g, pre=pre, fn=fn):
                    return fn(pre(g))
                if _verify(composed, train):
                    rules.append((pname + "+" + name, composed))
                    break
            if rules:
                break
    return rules, fresh


# ###########################################################################
# SELF-GEN CURRICULUM + light baked op-relevance ranker  (from baked-proposer)
# Trains an op-ranker on synthetic (input,output,op) transitions at IMPORT; the ranker re-orders
# the DSL search op vocabulary so the beam explores likely-relevant ops first under tight budget.
# make_curriculum() is EXPORTED for later generations to train bigger models (incl. Kaggle).
# ###########################################################################
def _comp_count(g):
    if g.size > 400:
        return 0
    return len(_components(g, bg=0, diag=False))


def transition_features(gi, go):
    hi, wi = gi.shape; ho, wo = go.shape
    si, so = gi.size, go.size
    ci = set(np.unique(gi).tolist()); co = set(np.unique(go).tolist())
    rr = ho / hi if hi else 1.0
    cr = wo / wi if wi else 1.0
    sr = so / si if si else 1.0
    pix_changed = float((gi != go).mean()) if gi.shape == go.shape else 1.0
    return np.array([
        1.0, float(gi.shape == go.shape),
        float(rr), float(cr), float(sr),
        float(rr == 2.0), float(cr == 2.0), float(rr == 0.5), float(cr == 0.5),
        float(ho > hi or wo > wi), float(ho < hi or wo < wi),
        float(len(ci)), float(len(co)), float(len(co - ci)), float(len(ci - co)),
        float((gi != 0).mean()), float((go != 0).mean()), pix_changed,
        float(hi), float(wi), float(hi == wi), float(_comp_count(gi)),
    ], float)


def task_features(train):
    return np.mean([transition_features(gi, go) for gi, go in train], 0)


FEAT_DIM = len(transition_features(np.zeros((2, 2), int), np.zeros((2, 2), int)))


def _rand_grid(rng):
    h = rng.randint(2, 9); w = rng.randint(2, 9)
    ncol = rng.randint(2, 6)
    if rng.rand() < 0.5:
        g = np.zeros((h, w), int)
        k = rng.randint(1, max(2, (h * w) // 3))
        for _ in range(k):
            g[rng.randint(h), rng.randint(w)] = rng.randint(1, ncol)
    else:
        g = rng.randint(0, ncol, (h, w))
    return g


def make_curriculum(n=2500, seed=0):
    """EXPORTED self-gen curriculum: random DSL programs applied to random seeded grids -> synthetic
    (input,output,program) tasks; yields (transition_feature_vector, op_name) labels. The experience
    'baked from data' signal: features of a change -> the op that makes that change."""
    rng = np.random.RandomState(seed)
    names = dsl.OP_NAMES
    samples = []
    for _ in range(n):
        g = _rand_grid(rng)
        pal = [c for c in np.unique(g).tolist() if c != 0] or [1]
        L = rng.randint(1, 4)
        cur = g
        for _step in range(L):
            name = names[rng.randint(len(names))]
            fn, nc = dsl.OPS[name]
            if nc == 0:
                args = ()
            elif nc == 1:
                args = (pal[rng.randint(len(pal))],)
            else:
                if len(pal) >= 2:
                    a, b = rng.choice(pal, 2, replace=False)
                    args = (int(a), int(b))
                else:
                    args = (pal[0], (pal[0] % 9) + 1)
            try:
                nxt = fn(cur, *args)
            except Exception:
                break
            if nxt is None or nxt.size == 0 or nxt.size > 1600:
                break
            if not (nxt.shape == cur.shape and np.array_equal(nxt, cur)):
                samples.append((transition_features(cur, nxt), name))
            cur = nxt
            pal = [c for c in np.unique(cur).tolist() if c != 0] or pal
    return samples


def _train_ranker(samples, epochs=4, lr=0.4, seed=1):
    names = dsl.OP_NAMES
    idx = {n: k for k, n in enumerate(names)}
    W = np.zeros((len(names), FEAT_DIM))
    if not samples:
        return W, idx, np.zeros(FEAT_DIM), np.ones(FEAT_DIM)
    X = np.array([f for f, _ in samples])
    y = np.array([idx[n] for _, n in samples])
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    rng = np.random.RandomState(seed)
    N = len(Xs)
    for _e in range(epochs):
        order = rng.permutation(N)
        for i in order:
            f = Xs[i]; t = y[i]
            logits = W @ f
            logits -= logits.max()
            p = np.exp(logits); p /= p.sum()
            p[t] -= 1.0
            W -= lr * np.outer(p, f)
    return W, idx, mu, sd


_T0 = time.time()
_CURRICULUM = make_curriculum(n=2500, seed=0)
_W, _OPIDX, _MU, _SD = _train_ranker(_CURRICULUM, epochs=4)
_BUILD_SEC = time.time() - _T0


def rank_ops(train, topn=None):
    f = task_features(train)
    fs = (f - _MU) / _SD
    scores = _W @ fs
    names = dsl.OP_NAMES
    order = sorted(range(len(names)), key=lambda k: -scores[k])
    ranked = [names[k] for k in order]
    return ranked[:topn] if topn else ranked


# ###########################################################################
# SEED DSL SEARCH  (never regress below the gen-0 seed) + ranker order + library macros
# ###########################################################################
def _gdist(a, b):
    if a is None:
        return 3.0
    if a.shape != b.shape:
        return 1.0 + abs(a.size - b.size) / max(a.size, b.size, 1)
    return float((a != b).mean())


def _instantiate(train, macro_ops):
    """Length-1 op instances; op NAMES ordered by the baked ranker (+ in-session op hits). Library
    macros (arg-free super-ops) are spliced in first (high-priority reuse of proven subsequences)."""
    pal = dsl.palette(train)
    colors = [c for c in pal if c != 0]
    ranked = rank_ops(train)
    if _LIB.op_hits:
        ranked = sorted(ranked, key=lambda n: -_LIB.op_hits.get(n, 0))
    insts = []
    for mname, fn in macro_ops:
        insts.append((mname, (), fn))
    for name in ranked:
        if name not in dsl.OPS:
            continue
        nc = dsl.OPS[name][1]
        if nc == 0:
            insts.append((name, (), None))
        elif nc == 1:
            for c in colors:
                insts.append((name, (c,), None))
        elif nc == 2:
            for a in colors:
                for b in colors:
                    if a != b:
                        insts.append((name, (a, b), None))
    return insts


def _apply_inst(g, inst):
    name, args, fn = inst
    if fn is not None:
        return fn(g)
    return dsl.OPS[name][0](g, *args)


def _apply_all(outs, inst):
    res = []
    for g in outs:
        try:
            res.append(_apply_inst(g, inst))
        except Exception:
            return None
    return res


def _exact(outs, tgt):
    return all(o is not None and o.shape == t.shape and np.array_equal(o, t) for o, t in zip(outs, tgt))


def _search_collect(train, B, W=25, max_len=3, K=4):
    macro_ops = _LIB.macro_ops()
    insts = _instantiate(train, macro_ops)
    ins = [gi for gi, _ in train]; tgt = [go for _, go in train]
    start = sum(_gdist(a, b) for a, b in zip(ins, tgt)) / len(ins)
    heap = [(start, 0, [], ins)]; ctr = 1; nexec = 0; found = []
    while heap and nexec < B:
        _score, _c, prog, outs = heapq.heappop(heap)
        if len(prog) >= max_len:
            continue
        kids = []
        for inst in insts:
            outs2 = _apply_all(outs, inst); nexec += 1
            if outs2 is None:
                if nexec >= B:
                    break
                continue
            if _exact(outs2, tgt):
                found.append(prog + [inst])
                if len(found) >= K:
                    return found, nexec
            else:
                sc = sum(_gdist(a, b) for a, b in zip(outs2, tgt)) / len(outs2)
                kids.append((sc, ctr, prog + [inst], outs2)); ctr += 1
            if nexec >= B:
                break
        for k in heapq.nsmallest(W, kids):
            heapq.heappush(heap, k)
    return found, nexec


def _plen(p):
    return sum(1 + len(a) for name, a, *_ in p)


def _prog_run(gi, prog):
    g = gi
    try:
        for inst in prog:
            g = _apply_inst(g, inst)
        return g
    except Exception:
        return None


def _seed_attempts(train, test_inputs, budget):
    progs, _ = _search_collect(train, budget, K=4)
    progs = sorted(progs, key=_plen)
    # ingest the shortest arg-free program for macro mining; bump op hits (in-session experience)
    if progs:
        best = min(progs, key=_plen)
        plain = [(n, a) for (n, a, *_) in best if not n.startswith("MACRO[")]
        if len(plain) == len(best) and all(n in dsl.OPS for n, _ in plain):
            _LIB.ingest_program("p%d" % _TASK_COUNTER[0], plain)
        for n, a, *_ in best:
            if not n.startswith("MACRO["):
                _LIB.op_hits[n] += 1
    progs = progs[:2]
    attempts = []
    for gi in test_inputs:
        cand = []
        for p in progs:
            o = _prog_run(gi, p)
            if o is not None:
                cand.append(o)
        attempts.append(cand)
    return attempts


# ###########################################################################
# PUBLIC ENTRYPOINT
# ###########################################################################
_TASK_COUNTER = [0]


# Overfit-prone concepts: a 3x3 lookup / mirror-overlay can MEMORIZE train yet miss test. When ONLY these
# fit, we must NOT short-circuit the seed fallback — the overfit rule may take at most ONE attempt slot,
# the seed search backfills the other so a correct program is never crowded out (MDL/coverage discipline).
_OVERFIT_PRONE = {"local_rule", "local_rule_plain", "sym_overlay"}


def _base_name(tag):
    return tag.split("+")[-1].split(">")[-1].replace("lib:", "").replace("link:", "").split("#")[0]


def _components_of(tag):
    """All concept base-names referenced by a tag (a composition like link:a>b or pre+concept)."""
    body = tag.replace("link:", "").replace("lib:", "")
    parts = []
    for chunk in body.replace("+", ">").split(">"):
        parts.append(chunk.split("#")[0])
    return parts


def _is_prone(tag):
    """A rule is overfit-prone if ANY of its components is in _OVERFIT_PRONE (a link ending in a trusted
    concept but starting from a memorizing 3x3-lookup is still overfit-prone)."""
    return any(p in _OVERFIT_PRONE for p in _components_of(tag))


def solve(train, test_inputs, budget):
    train = [(np.asarray(gi, int), np.asarray(go, int)) for gi, go in train]
    tid = "t%d" % _TASK_COUNTER[0]; _TASK_COUNTER[0] += 1

    # 1) parametric/structural concept store + linker + library replay/compose (exact-verified)
    rules, fresh = _try_concepts(train)
    if rules:
        # record experience: bump verified concept counts + remember standalone fresh closures (transfer)
        for name, fn in rules:
            _LIB.bump(_base_name(name))
        for ftag, ff in fresh:
            # only transfer GENERALIZING concepts: overfit-prone 3x3-lookup/overlay closures memorize
            # train and would replay spuriously, so they stay task-local and never enter the library.
            if not _is_prone(ftag):
                _LIB.remember_closure(ftag, ff)
                _LIB.audit.append((tid, ftag, False))
        for name, _ in rules:
            if name.startswith("lib:") or name.startswith("link:"):
                _LIB.audit.append((tid, name, True))  # a TRANSFER/LINK fired (creativity signal)

        # order concept rules so TRUSTED (low-overfit) fire before OVERFIT-PRONE ones for the 2 slots
        trusted = [(n, f) for n, f in rules if not _is_prone(n)]
        prone = [(n, f) for n, f in rules if _is_prone(n)]
        ordered = trusted + prone

        attempts = []
        for gi in test_inputs:
            cand = []
            for _name, fn in ordered:
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

        # Trust the fast path ONLY when a TRUSTED concept fired and every test got a candidate. When the
        # only fitters are overfit-prone, fall through to merge with the seed search (never crowd it out).
        if trusted and all(len(a) >= 1 for a in attempts):
            return attempts

        # MERGE PATH: keep overfit-prone concept as attempt-1, seed search backfills the remaining slot.
        seed = _seed_attempts(train, test_inputs, budget)
        merged = []
        for k, gi in enumerate(test_inputs):
            cand = list(attempts[k]) if k < len(attempts) else []
            for o in (seed[k] if k < len(seed) else []):
                if o is None:
                    continue
                if any(_eq(o, c) for c in cand):
                    continue
                cand.append(o)
                if len(cand) >= 2:
                    break
            merged.append(cand[:2])
        return merged

    # 2) seed DSL fallback (never regress below the gen-0 seed) + macro/op-hit ingestion
    return _seed_attempts(train, test_inputs, budget)
