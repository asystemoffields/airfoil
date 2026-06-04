#!/usr/bin/env python3
"""Gen-1 candidate #6 — BAKED-IN-FROM-DATA proposer + self-generated curriculum.

Facet thesis: creativity = a CONCEPT STORE filled by EXPERIENCE + a LINKER that recombines concepts,
filtered by exact verify. This candidate operationalises the BAKED-IN-FROM-DATA experience channel:

  (1) SELF-GEN CURRICULUM (import time, ~<1s): sample random DSL programs (len 1-3), apply them to random
      small seeded grids -> synthetic (input, output, program) tasks. From each we extract cheap grid-pair
      *transition features* and record which op produced the transition. We train a LIGHT, sklearn-free
      feature ranker: a per-op logistic-ish scorer (one weight vector per op, trained by averaged-perceptron
      / online logistic SGD) mapping transition features -> P(op relevant). Cached at module scope.
      `make_curriculum()` is exported so later generations can train bigger models (incl. on Kaggle).

  (2) BAKED PROPOSER: at solve time we featurise the task's train pairs and ask the ranker which DSL ops are
      most relevant; search then tries the relevant few FIRST (re-orders op instantiation + biases the
      best-first frontier). This is 'experience baked in from training on data'.

  (3) BREADTH EXPANSION (the real ARC wall is bucket-A expressiveness): a small library of NEW parametric
      MACRO-CONCEPTS that capture transformation families the 32-op DSL cannot express in <=3 ops
      (multi-color maps, periodic-tile completion, mirror 2x2 layouts, half-grid boolean logic, fractal
      self-tiling, connect-the-dots, object recolor-by-size-rank, row/col dedup, symmetry overlay). Each is
      FIT deterministically from the current task's train pairs and exact-verified; this is functional
      repurposing (e.g. `recolor` used as a whole learned bijection, not a single op).

  (4) IN-SESSION LIBRARY (module-level state): every macro-concept that VERIFIES on a task is logged; its
      hit-count up-weights it for later tasks in the same run. This is the IN-SESSION experience channel
      layered on top of the baked one — the store grows as we solve.

MDL ordering: macro-concepts and DSL programs are scored by a description length (concept complexity +
how baked/in-session experience ranks them); simplest, most-experienced links are tried first and returned
first among the 2 attempts.

INTEGRITY: solve() reads ONLY (a) the current task's train pairs, (b) module state from prior solve() calls
(verified-correct only), (c) self-generated synthetic data built at import. No ARC files, no test outputs,
no network, no LLM. Run with /data/llm/.venv/bin/python.
"""
import sys, heapq, time
import numpy as np
from collections import deque, defaultdict

ARC = "/data/Windows-files/Documents/airfoil/incubation/arc"
if ARC not in sys.path:
    sys.path.insert(0, ARC)
import dsl

META = {"name": "baked_proposer_v1",
        "desc": "self-gen curriculum trains a light op-relevance ranker (baked-from-data); macro-concept "
                "breadth library + in-session experience; ranker re-orders best-first DSL search"}

# ============================================================================
# Cheap grid / transition features (sklearn-free)
# ============================================================================

def _palette(g):
    return np.unique(g)

def _ncolors(g):
    return int(len(np.unique(g)))

def _components(g):
    h, w = g.shape; seen = np.zeros_like(g, bool); comps = []
    for i in range(h):
        for j in range(w):
            if g[i, j] != 0 and not seen[i, j]:
                comp = []; q = deque([(i, j)]); seen[i, j] = True
                while q:
                    a, b = q.popleft(); comp.append((a, b))
                    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        x, y = a + di, b + dj
                        if 0 <= x < h and 0 <= y < w and g[x, y] != 0 and not seen[x, y]:
                            seen[x, y] = True; q.append((x, y))
                comps.append(comp)
    return comps


def transition_features(gi, go):
    """Cheap features of an (input,output) PAIR -> what kind of transition happened. Length-fixed vector.
    These are the signals the baked ranker keys on (shape change, color set change, symmetry, size...)."""
    hi, wi = gi.shape; ho, wo = go.shape
    si, so = gi.size, go.size
    ci = set(np.unique(gi).tolist()); co = set(np.unique(go).tolist())
    same_shape = float(gi.shape == go.shape)
    # safe ratios
    rr = ho / hi if hi else 1.0
    cr = wo / wi if wi else 1.0
    sr = so / si if si else 1.0
    nz_i = float((gi != 0).mean())
    nz_o = float((go != 0).mean())
    pix_changed = float((gi != go).mean()) if gi.shape == go.shape else 1.0
    feats = [
        1.0,                                   # bias
        same_shape,
        float(rr), float(cr), float(sr),
        float(rr == 2.0), float(cr == 2.0), float(rr == 0.5), float(cr == 0.5),
        float(ho > hi or wo > wi), float(ho < hi or wo < wi),
        float(len(ci)), float(len(co)),
        float(len(co - ci)),                   # new colors appeared
        float(len(ci - co)),                   # colors removed
        nz_i, nz_o, pix_changed,
        float(hi), float(wi), float(hi == wi),
        float(len(_components(gi))) if si <= 400 else 0.0,
    ]
    return np.array(feats, float)


def task_features(train):
    """Average transition features across a task's train pairs (the proposer's view of a task)."""
    fs = [transition_features(gi, go) for gi, go in train]
    return np.mean(fs, 0)


FEAT_DIM = len(transition_features(np.zeros((2, 2), int), np.zeros((2, 2), int)))

# ============================================================================
# SELF-GENERATED CURRICULUM  +  light baked op-relevance ranker
# ============================================================================

def _rand_grid(rng):
    h = rng.randint(2, 9); w = rng.randint(2, 9)
    ncol = rng.randint(2, 6)
    # bias toward sparse object-like grids half the time
    if rng.rand() < 0.5:
        g = np.zeros((h, w), int)
        k = rng.randint(1, max(2, (h * w) // 3))
        for _ in range(k):
            g[rng.randint(h), rng.randint(w)] = rng.randint(1, ncol)
    else:
        g = rng.randint(0, ncol, (h, w))
    return g


def make_curriculum(n=2500, seed=0):
    """Self-gen curriculum generator (EXPORTED, reusable by later generations / Kaggle for bigger models).
    Sample random DSL programs (len 1-3), apply to random seeded grids -> synthetic tasks. Returns a list of
    (feature_vector, op_name) labelling which op caused each observed (input->output) transition.

    A program of length L contributes L labelled transitions (each step's before/after). This is exactly the
    'experience baked in from training on data' signal: features of a change -> the op that makes that change.
    """
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
            # record (transition features, op) — skip no-ops which give no signal
            if not (nxt.shape == cur.shape and np.array_equal(nxt, cur)):
                samples.append((transition_features(cur, nxt), name))
            cur = nxt
            pal = [c for c in np.unique(cur).tolist() if c != 0] or pal
    return samples


def _train_ranker(samples, epochs=4, lr=0.4, seed=1):
    """Light, sklearn-free one-vs-rest online-logistic ranker: one weight vector per op.
    score(op | feats) = sigmoid(w_op . feats); used to RANK ops by relevance for a task. Trained in <~1s."""
    names = dsl.OP_NAMES
    idx = {n: k for k, n in enumerate(names)}
    W = np.zeros((len(names), FEAT_DIM))
    if not samples:
        return W, idx
    X = np.array([f for f, _ in samples])
    y = np.array([idx[n] for _, n in samples])
    # standardise features (store scale so solve-time featurisation matches)
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    rng = np.random.RandomState(seed)
    nC = len(names); N = len(Xs)
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


# ---- build the baked model at IMPORT (cached) ----
_T0 = time.time()
_CURRICULUM = make_curriculum(n=2500, seed=0)
_RANK = _train_ranker(_CURRICULUM, epochs=4)
_W, _OPIDX, _MU, _SD = _RANK
_BUILD_SEC = time.time() - _T0


def rank_ops(train, topn=None):
    """Baked proposer: featurise a task, return DSL op names ordered by predicted relevance (best first)."""
    f = task_features(train)
    fs = (f - _MU) / _SD
    scores = _W @ fs
    names = dsl.OP_NAMES
    order = sorted(range(len(names)), key=lambda k: -scores[k])
    ranked = [names[k] for k in order]
    return ranked[:topn] if topn else ranked


# ============================================================================
# IN-SESSION EXPERIENCE LIBRARY (module-level state, grows across solve() calls)
# ============================================================================
_CONCEPT_HITS = defaultdict(int)     # macro-concept name -> #tasks it has solved this run
_OP_HITS = defaultdict(int)          # dsl op name -> #verified programs using it this run
_TASKS_SEEN = 0


# ============================================================================
# BREADTH LIBRARY — NEW parametric MACRO-CONCEPTS (the bucket-A expansion)
# Each: fit(train) -> param or None ;  apply(grid, param) -> grid or None.
# 'cost' is an MDL/description-length weight (simpler concept => smaller => tried/preferred first).
# ============================================================================

def _comps_colored(g):
    h, w = g.shape; seen = np.zeros_like(g, bool); cs = []
    for i in range(h):
        for j in range(w):
            if g[i, j] != 0 and not seen[i, j]:
                comp = []; q = deque([(i, j)]); seen[i, j] = True
                while q:
                    a, b = q.popleft(); comp.append((a, b))
                    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        x, y = a + di, b + dj
                        if 0 <= x < h and 0 <= y < w and g[x, y] != 0 and not seen[x, y]:
                            seen[x, y] = True; q.append((x, y))
                cs.append(comp)
    return cs


# --- C1: global multi-color map (learned color->color, applied everywhere) ---
def fit_colormap(train):
    m = {}
    for gi, go in train:
        if gi.shape != go.shape:
            return None
        for a, b in zip(gi.ravel().tolist(), go.ravel().tolist()):
            if a in m and m[a] != b:
                return None
            m[a] = b
    if all(a == b for a, b in m.items()):
        return None
    return m
def apply_colormap(g, m):
    out = g.copy()
    for a, b in m.items():
        if a != b:
            out[g == a] = b
    return out


# --- C2: periodic-tile completion (find smallest fully-determined period, tile it) ---
def fit_period(train):
    for gi, go in train:
        if gi.shape != go.shape:
            return None
    return True
def _period_tile(g):
    h, w = g.shape
    for pr in range(1, h + 1):
        for pc in range(1, w + 1):
            if pr == h and pc == w:
                continue
            tile = np.full((pr, pc), -1, int); ok = True
            for i in range(h):
                if not ok: break
                for j in range(w):
                    v = g[i, j]
                    if v != 0:
                        ti, tj = i % pr, j % pc
                        if tile[ti, tj] == -1:
                            tile[ti, tj] = v
                        elif tile[ti, tj] != v:
                            ok = False; break
            if ok and not (tile == -1).any():
                return pr, pc, tile
    return None
def apply_period(g, _):
    res = _period_tile(g)
    if res is None:
        return None
    pr, pc, tile = res
    h, w = g.shape
    out = np.empty((h, w), int)
    for i in range(h):
        for j in range(w):
            out[i, j] = tile[i % pr, j % pc]
    return out


# --- C3: mirror 2x2 layout (output = 2x2 of {I,H,V,HV}, arrangement learned) ---
_MARR = {'I': lambda g: g, 'H': lambda g: g[:, ::-1], 'V': lambda g: g[::-1, :], 'HV': lambda g: g[::-1, ::-1]}
def fit_mirror_tile(train):
    keys = list(_MARR); layout = None
    for gi, go in train:
        h, w = gi.shape
        if go.shape != (2 * h, 2 * w):
            return None
        quads = [((0, 0), go[:h, :w]), ((0, 1), go[:h, w:]), ((1, 0), go[h:, :w]), ((1, 1), go[h:, w:])]
        cur = {}
        for pos, q in quads:
            match = [k for k in keys if q.shape == gi.shape and np.array_equal(q, _MARR[k](gi))]
            if not match:
                return None
            cur[pos] = match[0]
        if layout is None:
            layout = cur
        elif layout != cur:
            return None
    return layout
def apply_mirror_tile(g, layout):
    h, w = g.shape; out = np.zeros((2 * h, 2 * w), int)
    for (pi, pj), k in layout.items():
        out[pi * h:(pi + 1) * h, pj * w:(pj + 1) * w] = _MARR[k](g)
    return out


# --- C4: half-grid boolean logic (split in half, cellwise AND/OR/XOR/DIFF/NAND -> single color) ---
def _split_halves(g):
    h, w = g.shape; res = []
    if w % 2 == 1:
        res.append(('v_sep', g[:, :w // 2], g[:, w // 2 + 1:]))
    if w % 2 == 0:
        res.append(('v', g[:, :w // 2], g[:, w // 2:]))
    if h % 2 == 1:
        res.append(('h_sep', g[:h // 2, :], g[h // 2 + 1:, :]))
    if h % 2 == 0:
        res.append(('h', g[:h // 2, :], g[h // 2:, :]))
    return res
def _logic_mask(a, b, op):
    if op == 'and': return a & b
    if op == 'or': return a | b
    if op == 'xor': return a ^ b
    if op == 'diff': return a & ~b
    if op == 'nand': return ~(a & b)
    return None
def fit_halflogic(train):
    for s in ('v', 'v_sep', 'h', 'h_sep'):
        for op in ('and', 'or', 'xor', 'diff', 'nand'):
            outcol = None; ok = True
            for gi, go in train:
                sh = [x for x in _split_halves(gi) if x[0] == s]
                if not sh:
                    ok = False; break
                _, A, B = sh[0]
                if A.shape != B.shape or A.shape != go.shape:
                    ok = False; break
                mask = _logic_mask(A != 0, B != 0, op)
                if mask.sum() == 0:
                    ok = False; break
                ocols = set(go[mask].tolist()); zcols = set(go[~mask].tolist())
                if len(ocols) != 1 or not (zcols <= {0}):
                    ok = False; break
                oc = ocols.pop()
                if outcol is None:
                    outcol = oc
                elif outcol != oc:
                    ok = False; break
            if ok and outcol is not None:
                return (s, op, outcol)
    return None
def apply_halflogic(g, param):
    s, op, outcol = param
    sh = [x for x in _split_halves(g) if x[0] == s]
    if not sh:
        return None
    _, A, B = sh[0]
    if A.shape != B.shape:
        return None
    mask = _logic_mask(A != 0, B != 0, op)
    out = np.zeros(A.shape, int); out[mask] = outcol
    return out


# --- C5: fractal self-tiling (output = input replicated at cells where input is nonzero) ---
def fit_fractal(train):
    for gi, go in train:
        h, w = gi.shape
        if go.shape != (h * h, w * w):
            return None
    return True
def apply_fractal(g, _):
    h, w = g.shape
    out = np.zeros((h * h, w * w), int)
    for i in range(h):
        for j in range(w):
            if g[i, j] != 0:
                out[i * h:(i + 1) * h, j * w:(j + 1) * w] = g
    return out


# --- C6: connect-the-dots (same-color collinear pairs joined by a line of that color) ---
def fit_connect(train):
    for gi, go in train:
        if gi.shape != go.shape:
            return None
    return True
def apply_connect(g, _):
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


# --- C7: object recolor by size rank (largest->color, etc., learned) ---
def fit_size_recolor(train):
    rankmap = {}
    for gi, go in train:
        if gi.shape != go.shape:
            return None
        cs = _comps_colored(gi)
        if not cs:
            return None
        order = sorted(range(len(cs)), key=lambda k: -len(cs[k]))
        for rank, ix in enumerate(order):
            ocols = set(go[a, b] for a, b in cs[ix])
            if len(ocols) != 1:
                return None
            oc = ocols.pop()
            if rank in rankmap and rankmap[rank] != oc:
                return None
            rankmap[rank] = oc
    return rankmap
def apply_size_recolor(g, rankmap):
    cs = _comps_colored(g); out = g.copy()
    order = sorted(range(len(cs)), key=lambda k: -len(cs[k]))
    mx = max(rankmap) if rankmap else 0
    for rank, ix in enumerate(order):
        c = rankmap.get(rank, rankmap.get(mx))
        for a, b in cs[ix]:
            out[a, b] = c
    return out


# --- C8: row/col dedup (collapse consecutive duplicate rows then cols) ---
def fit_dedup(train):
    for gi, go in train:
        if go.shape[0] > gi.shape[0] or go.shape[1] > gi.shape[1]:
            return None
        if go.shape == gi.shape:
            return None
    return True
def apply_dedup(g, _):
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


# --- C9: symmetry overlay completion (fill zeros from mirror copies; same shape) ---
def fit_symfill(train):
    for gi, go in train:
        if gi.shape != go.shape:
            return None
    return True
def apply_symfill(g, _):
    out = g.copy()
    mirs = [g[:, ::-1], g[::-1, :], g[::-1, ::-1]]
    if g.shape[0] == g.shape[1]:
        mirs += [g.T]
    for m in mirs:
        z = out == 0
        out[z] = m[z]
    return out


# Concept registry: name -> (fit, apply, mdl_cost). Lower cost = simpler concept = tried/preferred first.
CONCEPTS = [
    ("colormap",      fit_colormap,     apply_colormap,     1.0),
    ("symfill",       fit_symfill,      apply_symfill,      1.2),
    ("period_tile",   fit_period,       apply_period,       1.4),
    ("connect_dots",  fit_connect,      apply_connect,      1.6),
    ("dedup",         fit_dedup,        apply_dedup,        1.6),
    ("size_recolor",  fit_size_recolor, apply_size_recolor, 1.8),
    ("mirror_tile",   fit_mirror_tile,  apply_mirror_tile,  2.0),
    ("half_logic",    fit_halflogic,    apply_halflogic,    2.2),
    ("fractal",       fit_fractal,      apply_fractal,      2.2),
]


def try_concepts(train):
    """Fit + exact-verify every macro-concept against the task train pairs.
    Returns verified (name, param, mdl) sorted by experience-adjusted MDL (best first)."""
    hits = []
    for name, fit, app, cost in CONCEPTS:
        try:
            param = fit(train)
        except Exception:
            param = None
        if param is None:
            continue
        ok = True
        for gi, go in train:
            try:
                o = app(gi, param)
            except Exception:
                o = None
            if o is None or o.shape != go.shape or not np.array_equal(o, go):
                ok = False; break
        if ok:
            # in-session experience: concepts that solved earlier tasks this run get a discount
            adj = cost - 0.15 * _CONCEPT_HITS.get(name, 0)
            hits.append((name, param, adj))
    hits.sort(key=lambda x: x[2])
    return hits


def _concept_apply_by_name(name):
    for nm, _fit, app, _c in CONCEPTS:
        if nm == name:
            return app
    return None


# ============================================================================
# BAKED-RANKED DSL SEARCH  (seed's best-first, but op order seeded by the ranker + in-session op hits)
# ============================================================================

def _gdist(a, b):
    if a is None:
        return 3.0
    if a.shape != b.shape:
        return 1.0 + abs(a.size - b.size) / max(a.size, b.size, 1)
    return float((a != b).mean())

def _apply_all(outs, name, args):
    res = []
    for g in outs:
        try:
            res.append(dsl.OPS[name][0](g, *args))
        except Exception:
            return None
    return res

def _exact(outs, tgt):
    return all(o is not None and o.shape == t.shape and np.array_equal(o, t) for o, t in zip(outs, tgt))


def _instantiate_ranked(train):
    """All length-1 op instances, but with op NAMES ordered by the baked ranker (+ in-session op hits).
    Instances of the most-relevant ops come first so the beam explores them earlier under tight budget."""
    pal = dsl.palette(train)
    colors = [c for c in pal if c != 0]
    ranked = rank_ops(train)
    # in-session up-weight: stable-sort ops we've seen succeed this run toward the front
    if _OP_HITS:
        ranked = sorted(ranked, key=lambda n: -_OP_HITS.get(n, 0))
    insts = []
    for name in ranked:
        if name not in dsl.OPS:
            continue
        nc = dsl.OPS[name][1]
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


def search_collect(train, B, W=25, max_len=3, K=4):
    insts = _instantiate_ranked(train)
    ins = [gi for gi, _ in train]; tgt = [go for _, go in train]
    start = sum(_gdist(a, b) for a, b in zip(ins, tgt)) / len(ins)
    heap = [(start, 0, [], ins)]; ctr = 1; nexec = 0; found = []
    while heap and nexec < B:
        _score, _c, prog, outs = heapq.heappop(heap)
        if len(prog) >= max_len:
            continue
        kids = []
        for inst in insts:
            outs2 = _apply_all(outs, inst[0], inst[1]); nexec += 1
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
    return sum(1 + len(a) for _, a in p)


# ============================================================================
# solve()
# ============================================================================

def solve(train, test_inputs, budget):
    """Best-first over: (1) verified macro-concepts (breadth library, MDL-ordered, experience-adjusted),
    then (2) baked-ranker-ordered DSL program search. Returns up to 2 attempts per test input, best-first.

    Experience channels used: BAKED-IN (curriculum-trained op ranker) + IN-SESSION (concept/op hit counts
    from prior verified solves this run). Verified solutions feed back into the library."""
    global _TASKS_SEEN
    _TASKS_SEEN += 1

    # --- channel 1: macro-concepts (breadth). These are the NEW links the seed DSL can't express. ---
    concept_hits = try_concepts(train)

    # --- channel 2: baked-ranked DSL search (covers the seed's family + ranker-accelerated). ---
    progs, _ = search_collect(train, budget, K=4)
    progs = sorted(progs, key=_plen)

    # Build a unified, MDL-ordered list of "solvers" (callables grid->grid|None) with a cost.
    solvers = []  # (cost, kind, payload)
    for name, param, mdl in concept_hits:
        solvers.append((mdl, "concept", (name, param)))
    for p in progs:
        # DSL program cost: bump above concepts of equal length only slightly; concepts preferred when tied.
        solvers.append((0.9 + _plen(p) * 0.5, "prog", p))
    solvers.sort(key=lambda x: x[0])

    # record verified experience into the in-session library
    for name, _param, _mdl in concept_hits:
        _CONCEPT_HITS[name] += 1
    for p in progs[:1]:
        for op, _args in p:
            _OP_HITS[op] += 1

    if not solvers:
        return [[] for _ in test_inputs]

    attempts = []
    for gi in test_inputs:
        cand = []
        seen = []
        for cost, kind, payload in solvers:
            if len(cand) >= 2:
                break
            try:
                if kind == "concept":
                    name, param = payload
                    o = _concept_apply_by_name(name)(gi, param)
                else:
                    o = dsl.apply_prog(gi, payload)
            except Exception:
                o = None
            if o is None:
                continue
            # dedup identical candidate grids (keep distinct best-first attempts)
            dup = any(c.shape == o.shape and np.array_equal(c, o) for c in seen)
            if dup:
                continue
            seen.append(o); cand.append(o)
        attempts.append(cand)
    return attempts
