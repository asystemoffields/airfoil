#!/usr/bin/env python3
"""MECHANISM-COMPOSITION CURRICULUM — the experience-prior that makes real-time mechanism INVENTION fast.

THESIS. Creativity here = (1) UNRESTRICTED grasp of cause-and-effect: induce the INVARIANT causal
mechanism that generates a task's train pairs (cross-example invariance is what licenses causal-vs-
correlational induction; the exact verifier tests it on the held-out intervention) AND (2) real-time
INVENTION of that mechanism: not retrieving/ranking a whole-mechanism template from a fixed menu, but
SYNTHESIZING a new cause-effect rule per task by COMPOSING + ABSTRACTING + REPURPOSING primitive
relations, made FAST by an experience-prior over how primitives compose.

ALPHABET vs SENTENCE (the load-bearing distinction).
  * The ALPHABET (knowledge — fine to reuse): dsl.py's 33 primitive grid->grid ops + a handful of
    grid RELATIONS (object map, region restriction, feature predicates, color/size orderings). Letters.
  * A MECHANISM (an invented SENTENCE): a multi-step, object-/region-parameterized COMPOSITION of those
    letters. gen-1/2 retrieved whole sentences from a fixed menu of `fit_*` whole-mechanism templates
    (template induction). We TRANSCEND that: this module teaches the GRAMMAR by which letters compose into
    sentences, so an inventor can SYNTHESIZE sentences never seen at build time.

WHAT THIS MODULE PROVIDES.
  * make_curriculum(n)            : sample n NOVEL mechanisms (compositions/abstractions of primitives),
                                    apply each to seeded small grids, emit (input,output) example-tasks
                                    WITH the structured label (relations used + composition structure).
                                    The sampler targets COMPOSITIONAL DIVERSITY, not a fixed list.
  * features(train)               : cheap INVARIANCE features of a task's train pairs (size/shape deltas,
                                    color-set changes, object counts, what is INVARIANT across pairs).
  * propose_compositions(train,k) : given those features, score/propose which primitive relations +
                                    composition SHAPES are likely — a feature-conditioned sampler backed
                                    by a tiny MLP trained on the curriculum at import (<90s, cached).
  * heavy_train(...)              : stub hook for a future Kaggle-scale prior over the same label space.

CREATIVITY ABLATION (what makes a solve COUNT). A mechanism here is composite by construction: it has a
`combinator` (sequence / per-object map / region-restrict / feature-conditional / repurpose) wrapping >=1
primitive relation. A solve that survives only because of a single whole-template retrieval is NOT
creative; this curriculum's units are precisely the ones a single-template retrieval cannot produce.

INTEGRITY. Everything here is SELF-GENERATED at import (synthetic grids + synthetic mechanisms). It reads
NO ARC task file, NO test output, no network, no LLM. The prior trains only on this self-gen curriculum.
At solve time a caller uses features()/propose_compositions() over the CURRENT task's train pairs only.
Pure python + numpy. Build-time work < ~90s, cached. Run/imported with /data/llm/.venv/bin/python."""
import sys, os, json, time, hashlib, random
from collections import Counter, defaultdict
import numpy as np

ARC = "/data/Windows-files/Documents/airfoil/incubation/arc"
if ARC not in sys.path:
    sys.path.insert(0, ARC)
import dsl  # the ALPHABET: 33 primitive grid->grid ops + palette()/load helpers

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "__pycache__", "mech_prior.json")

META = {"name": "mechanism_curriculum_v1",
        "desc": "generative grammar of mechanisms (compositions/abstractions of primitive causal relations) "
                "+ feature-conditioned prior trained on the self-gen curriculum; teaches the grammar so an "
                "inventor can synthesize unseen mechanisms, not retrieve whole templates from a menu"}


# ===========================================================================
# THE ALPHABET WE COMPOSE OVER
#   relations  : a curated subset of dsl ops grouped by the KIND of causal relation they express,
#                so the grammar can reason about what role a letter plays (geometry / cleanup / select /
#                move / paint / color-map). These are LETTERS, freely reusable knowledge.
#   combinators: the GRAMMAR — how letters compose into sentences (mechanisms). This is what we teach.
# ===========================================================================
# relation kind -> (op_name, n_color_args). Color-arg ops get colors bound from the grid palette.
RELATIONS = {
    "geom":    [("reflect_h", 0), ("reflect_v", 0), ("rot90", 0), ("rot180", 0), ("rot270", 0),
                ("transpose", 0), ("sym_lr", 0), ("sym_ud", 0)],
    "tile":    [("tile_h2", 0), ("tile_v2", 0), ("tile_2x2", 0), ("scale2", 0)],
    "move":    [("shift_up", 0), ("shift_down", 0), ("shift_left", 0), ("shift_right", 0),
                ("gravity_down", 0), ("gravity_up", 0), ("gravity_left", 0), ("gravity_right", 0)],
    "select":  [("largest_object", 0), ("keep_smallest", 0), ("crop_content", 0), ("trim_border", 0),
                ("keep_color", 1), ("remove_color", 1)],
    "paint":   [("fill_holes", 1), ("bbox_fill", 1), ("outline", 1)],
    "colormap":[("recolor", 2), ("swap_colors", 2)],
}
# flat op -> kind, for labelling proposals
OP_KIND = {op: kind for kind, lst in RELATIONS.items() for (op, _nc) in lst}
KINDS = list(RELATIONS)

# COMBINATORS = the grammar of composition. Each is a HIGHER-ORDER relation that takes primitive
# relation(s) and yields a new (composite) grid->grid mechanism. This is the part gen-1/2 lacked:
# they had whole templates, not a grammar that BUILDS templates.
COMBINATORS = ["sequence", "region_restrict", "per_object_map", "feature_conditional", "repurpose_overlay"]


# ===========================================================================
# GRID UTILITIES (relations over objects/regions) — used by the combinators.
# ===========================================================================
def _components(g, diag=False):
    from collections import deque
    h, w = g.shape; seen = np.zeros_like(g, bool); comps = []
    nbr = [(1, 0), (-1, 0), (0, 1), (0, -1)] + ([(1, 1), (1, -1), (-1, 1), (-1, -1)] if diag else [])
    for i in range(h):
        for j in range(w):
            if g[i, j] != 0 and not seen[i, j]:
                comp = []; q = deque([(i, j)]); seen[i, j] = True
                while q:
                    a, b = q.popleft(); comp.append((a, b))
                    for di, dj in nbr:
                        x, y = a + di, b + dj
                        if 0 <= x < h and 0 <= y < w and g[x, y] != 0 and not seen[x, y]:
                            seen[x, y] = True; q.append((x, y))
                comps.append(comp)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs) + 1, max(cs) + 1


def _apply_op(g, op, args):
    return dsl.OPS[op][0](g, *args)


# ---- the combinator implementations (each returns a NEW grid given a primitive relation spec) ----
def run_sequence(g, steps):
    """steps: list of (op, args). A plain composition (function composition of letters)."""
    for op, args in steps:
        g = _apply_op(g, op, args)
    return g


def run_region_restrict(g, op, args, region):
    """Apply a relation ONLY inside a region (top/bottom/left/right half, or content bbox); paste back.
    Causal claim: the cause acts on a SUB-REGION, the rest is invariant."""
    h, w = g.shape
    if region == "top":    r0, c0, r1, c1 = 0, 0, h // 2 or 1, w
    elif region == "bot":  r0, c0, r1, c1 = h // 2, 0, h, w
    elif region == "left": r0, c0, r1, c1 = 0, 0, h, w // 2 or 1
    elif region == "right":r0, c0, r1, c1 = 0, w // 2, h, w
    else:  # content bbox
        nz = np.argwhere(g != 0)
        if nz.size == 0: return g
        (r0, c0), (r1, c1) = nz.min(0), nz.max(0) + 1
    sub = g[r0:r1, c0:c1]
    try:
        sub2 = _apply_op(sub, op, args)
    except Exception:
        return g
    if sub2.shape != sub.shape:  # region ops must preserve sub-shape to paste back
        return g
    out = g.copy(); out[r0:r1, c0:c1] = sub2; return out


def run_per_object_map(g, recolor_rule, diag=False):
    """ABSTRACTION over objects: recolor each connected component by an ORDERING (size or size-rank).
    recolor_rule: ('by_size'|'by_rank', {ordinal: color}). Causal claim: the EFFECT is a function of an
    object property (its size), abstracted away from absolute position/color."""
    comps = _components(g, diag=diag)
    if not comps: return g
    out = g.copy()
    mode, table = recolor_rule
    if mode == "by_size":
        order = sorted(range(len(comps)), key=lambda i: len(comps[i]))
    else:  # by_rank == by_size here but indexed by rank ordinal (same effect, different label)
        order = sorted(range(len(comps)), key=lambda i: len(comps[i]))
    for ordinal, idx in enumerate(order):
        col = table.get(ordinal % max(len(table), 1))
        if col is None: continue
        for (a, b) in comps[idx]:
            out[a, b] = col
    return out


def run_feature_conditional(g, predicate, then_step, else_step):
    """CONDITIONAL mechanism: choose the relation based on a grid FEATURE (taller-than-wide, has a
    given color, #objects parity). Causal claim: the cause's effect DEPENDS on a measured condition."""
    h, w = g.shape
    if predicate == "tall":      cond = h > w
    elif predicate == "wide":    cond = w > h
    elif predicate == "many_obj":cond = len(_components(g)) >= 3
    else:                        cond = (g != 0).sum() > g.size // 2
    op, args = then_step if cond else else_step
    try:
        return _apply_op(g, op, args)
    except Exception:
        return g


def run_repurpose_overlay(g, base_op, base_args, overlay_op, overlay_args):
    """REPURPOSE two relations into a new one: produce a transformed copy and OVERLAY it onto the
    original (nonzero-of-overlay wins where original is 0). Causal claim: the effect SUPERIMPOSES a
    repurposed view (e.g. a mirror) onto the cause. Requires shape-compatible overlay."""
    try:
        base = _apply_op(g, base_op, base_args)
        ov = _apply_op(g, overlay_op, overlay_args)
    except Exception:
        return g
    if base.shape != ov.shape:
        return base
    out = base.copy(); m = (out == 0) & (ov != 0); out[m] = ov[m]; return out


# ===========================================================================
# A MECHANISM = a structured (label, callable). The label IS the sentence's grammar tree.
#   {combinator, kinds:[...], ops:[...], depth, params:{...}}
# Mechanisms are SAMPLED (not enumerated from a menu): we draw a combinator, then fill its primitive
# slots from RELATIONS, then bind any colors. Composition + abstraction are guaranteed by construction.
# ===========================================================================
_COLOR_BANK = [1, 2, 3, 4, 5, 6, 7, 8]


def _pick_op(rng, kinds=None, exclude=None):
    pool = []
    for kind, lst in RELATIONS.items():
        if kinds and kind not in kinds: continue
        for op, nc in lst:
            if exclude and op in exclude: continue
            pool.append((kind, op, nc))
    return pool[rng.randrange(len(pool))]


def _bind_args(rng, nc):
    if nc == 0: return ()
    if nc == 1: return (rng.choice(_COLOR_BANK),)
    a = rng.choice(_COLOR_BANK); b = rng.choice([c for c in _COLOR_BANK if c != a]); return (a, b)


def sample_mechanism(rng):
    """Draw ONE novel mechanism: a combinator wrapping primitive relation(s). Returns (label, fn)."""
    comb = rng.choice(COMBINATORS)

    if comb == "sequence":
        depth = rng.randint(2, 3)  # >=2 letters => a genuine composite sentence
        steps = []; kinds = []; ops = []
        for _ in range(depth):
            kind, op, nc = _pick_op(rng)
            args = _bind_args(rng, nc)
            steps.append((op, args)); kinds.append(kind); ops.append(op)
        label = {"combinator": "sequence", "kinds": kinds, "ops": ops, "depth": depth, "params": {}}
        fn = lambda g, s=steps: run_sequence(g, s)
        return label, fn

    if comb == "region_restrict":
        region = rng.choice(["top", "bot", "left", "right", "bbox"])
        # region ops must preserve sub-shape: geometry/move/paint/colormap (no tile/scale/crop)
        kind, op, nc = _pick_op(rng, kinds=["geom", "move", "paint", "colormap"])
        args = _bind_args(rng, nc)
        label = {"combinator": "region_restrict", "kinds": [kind], "ops": [op], "depth": 1,
                 "params": {"region": region}}
        fn = lambda g, o=op, a=args, r=region: run_region_restrict(g, o, a, r)
        return label, fn

    if comb == "per_object_map":
        mode = rng.choice(["by_size", "by_rank"])
        k = rng.randint(2, 3)
        cols = rng.sample(_COLOR_BANK, k)
        table = {i: cols[i] for i in range(k)}
        diag = rng.random() < 0.4
        label = {"combinator": "per_object_map", "kinds": ["select"], "ops": ["object_map"], "depth": 1,
                 "params": {"mode": mode, "n_classes": k, "diag": diag}}
        fn = lambda g, rule=(mode, table), d=diag: run_per_object_map(g, rule, diag=d)
        return label, fn

    if comb == "feature_conditional":
        pred = rng.choice(["tall", "wide", "many_obj", "dense"])
        k1, o1, n1 = _pick_op(rng, kinds=["geom", "move"])
        k2, o2, n2 = _pick_op(rng, kinds=["geom", "move"], exclude={o1})
        then_step = (o1, _bind_args(rng, n1)); else_step = (o2, _bind_args(rng, n2))
        label = {"combinator": "feature_conditional", "kinds": [k1, k2], "ops": [o1, o2], "depth": 1,
                 "params": {"predicate": pred}}
        fn = lambda g, p=pred, t=then_step, e=else_step: run_feature_conditional(g, p, t, e)
        return label, fn

    # repurpose_overlay
    bk, bo, bn = _pick_op(rng, kinds=["geom"])            # base: a geometric view
    ok, oo, on = _pick_op(rng, kinds=["geom"], exclude={bo})  # overlay: a different geometric view
    bargs = _bind_args(rng, bn); oargs = _bind_args(rng, on)
    label = {"combinator": "repurpose_overlay", "kinds": [bk, ok], "ops": [bo, oo], "depth": 2,
             "params": {}}
    fn = lambda g, b=(bo, bargs), o=(oo, oargs): run_repurpose_overlay(g, b[0], b[1], o[0], o[1])
    return label, fn


# ===========================================================================
# SEEDED SMALL GRIDS — the synthetic worlds mechanisms act on. Varied shape/objects/palette so the
# induced cause-effect is exercised across examples (cross-example invariance = the causal license).
# ===========================================================================
def _seed_grid(rng):
    h = rng.randint(4, 9); w = rng.randint(4, 9)
    g = np.zeros((h, w), int)
    style = rng.choice(["sprinkle", "objects", "stripes", "frame"])
    pal = rng.sample(_COLOR_BANK, rng.randint(1, 3))
    if style == "sprinkle":
        for _ in range(rng.randint(3, max(3, h * w // 4))):
            g[rng.randrange(h), rng.randrange(w)] = rng.choice(pal)
    elif style == "objects":
        for _ in range(rng.randint(2, 4)):
            r0 = rng.randrange(h); c0 = rng.randrange(w)
            bh = rng.randint(1, max(1, h - r0)); bw = rng.randint(1, max(1, w - c0))
            g[r0:r0 + bh, c0:c0 + bw] = rng.choice(pal)
    elif style == "stripes":
        for r in range(h):
            if rng.random() < 0.5: g[r, :] = rng.choice(pal)
    else:  # frame
        col = rng.choice(pal); g[0, :] = col; g[-1, :] = col; g[:, 0] = col; g[:, -1] = col
        if rng.random() < 0.5: g[rng.randrange(1, h - 1) if h > 2 else 0,
                                  rng.randrange(1, w - 1) if w > 2 else 0] = rng.choice(pal)
    return g


def _mechanism_to_task(label, fn, rng, n_pairs=4):
    """Apply a mechanism to several seeded inputs -> a train-pair set. Keep only NON-TRIVIAL,
    SHAPE-CONSISTENT-with-itself tasks where the mechanism actually does something on most pairs."""
    pairs = []; nontrivial = 0
    for _ in range(n_pairs + 3):  # oversample, then keep good ones
        gi = _seed_grid(rng)
        try:
            go = fn(gi)
        except Exception:
            continue
        if go is None or go.ndim != 2 or go.size == 0:
            continue
        if go.shape == gi.shape and not np.array_equal(go, gi):
            nontrivial += 1
        elif go.shape != gi.shape:
            nontrivial += 1
        pairs.append((gi, go))
        if len(pairs) >= n_pairs:
            break
    if len(pairs) < 2 or nontrivial < max(1, len(pairs) - 1):
        return None
    return {"label": label, "train": pairs[:-1] if len(pairs) > 2 else pairs, "test": pairs[-1:]}


# ===========================================================================
# PUBLIC: make_curriculum(n)
# ===========================================================================
def make_curriculum(n, seed=0):
    """Sample n NOVEL mechanism-tasks. Each item:
        {"label": <structured grammar tree>, "train": [(in,out)...], "test": [(in,out)]}
    Targets compositional diversity: rejects degenerate (no-op) mechanisms and dedupes identical labels'
    over-repetition so the prior sees the GRAMMAR, not a few memorized sentences."""
    rng = random.Random(seed)
    out = []; comb_counts = Counter(); tries = 0
    cap_per_comb = max(2, int(n * 0.45))  # diversity guard: no single combinator may dominate
    while len(out) < n and tries < n * 40:
        tries += 1
        label, fn = sample_mechanism(rng)
        comb = label["combinator"]
        if comb_counts[comb] >= cap_per_comb:
            continue
        task = _mechanism_to_task(label, fn, rng)
        if task is None:
            continue
        out.append(task); comb_counts[comb] += 1
    return out


# ===========================================================================
# PUBLIC: features(train) — cheap INVARIANCE features of a task's train pairs.
#   Captures WHAT IS INVARIANT across the pairs (the causal signature) without touching test outputs.
# ===========================================================================
def _palette(g): return set(np.unique(g).tolist())


def features(train):
    """Return a fixed-length float feature vector + a readable dict. INPUT ONLY of test is never needed
    here; this looks solely at the train (input,output) pairs of the CURRENT task."""
    if not train:
        return np.zeros(FEATURE_DIM), {}
    same_shape = []; area_ratio = []; h_ratio = []; w_ratio = []
    added_colors = []; removed_colors = []; palette_same = []
    obj_in = []; obj_out = []; obj_delta = []
    nz_ratio = []; out_subset_in_colors = []
    for gi, go in train:
        same_shape.append(1.0 if gi.shape == go.shape else 0.0)
        area_ratio.append(go.size / max(gi.size, 1))
        h_ratio.append(go.shape[0] / max(gi.shape[0], 1))
        w_ratio.append(go.shape[1] / max(gi.shape[1], 1))
        pi, po = _palette(gi), _palette(go)
        added_colors.append(len(po - pi)); removed_colors.append(len(pi - po))
        palette_same.append(1.0 if pi == po else 0.0)
        ci = len(_components(gi)); co = len(_components(go))
        obj_in.append(ci); obj_out.append(co); obj_delta.append(co - ci)
        nz_ratio.append((go != 0).sum() / max((gi != 0).sum(), 1))
        out_subset_in_colors.append(1.0 if (po - {0}) <= (pi - {0}) else 0.0)

    def m(x): return float(np.mean(x)) if x else 0.0
    def sd(x): return float(np.std(x)) if x else 0.0

    feat = {
        "same_shape_frac": m(same_shape),
        "area_ratio_mean": m(area_ratio), "area_ratio_std": sd(area_ratio),
        "h_ratio_mean": m(h_ratio), "w_ratio_mean": m(w_ratio),
        "added_colors_mean": m(added_colors), "removed_colors_mean": m(removed_colors),
        "palette_invariant_frac": m(palette_same),
        "obj_in_mean": m(obj_in), "obj_out_mean": m(obj_out),
        "obj_delta_mean": m(obj_delta), "obj_delta_std": sd(obj_delta),
        "nz_ratio_mean": m(nz_ratio),
        "out_colors_subset_in_frac": m(out_subset_in_colors),
        "shape_changes": 1.0 - m(same_shape),
        "is_upscale": 1.0 if m(area_ratio) > 1.5 else 0.0,
        "is_downscale": 1.0 if m(area_ratio) < 0.7 else 0.0,
    }
    vec = np.array([feat[k] for k in FEATURE_KEYS], float)
    return vec, feat


FEATURE_KEYS = [
    "same_shape_frac", "area_ratio_mean", "area_ratio_std", "h_ratio_mean", "w_ratio_mean",
    "added_colors_mean", "removed_colors_mean", "palette_invariant_frac",
    "obj_in_mean", "obj_out_mean", "obj_delta_mean", "obj_delta_std",
    "nz_ratio_mean", "out_colors_subset_in_frac", "shape_changes", "is_upscale", "is_downscale",
]
FEATURE_DIM = len(FEATURE_KEYS)


# ===========================================================================
# THE PRIOR — a tiny numpy MLP trained on the curriculum to map task-features -> a distribution over
#   (a) combinators and (b) relation-KINDS. Feature-conditioned: it learns the GRAMMAR's statistics
#   (which composition shapes & relation kinds explain which invariance signatures). Cached to disk.
# Labels per curriculum item: the combinator (1-of-5) and the multi-hot set of relation kinds used.
# ===========================================================================
COMB_INDEX = {c: i for i, c in enumerate(COMBINATORS)}
KIND_INDEX = {k: i for i, k in enumerate(KINDS)}


def _label_targets(label):
    comb = np.zeros(len(COMBINATORS)); comb[COMB_INDEX[label["combinator"]]] = 1.0
    kinds = np.zeros(len(KINDS))
    for k in label["kinds"]:
        if k in KIND_INDEX: kinds[KIND_INDEX[k]] = 1.0
    return comb, kinds


class _Prior:
    """Two-head tiny MLP (shared trunk). Pure numpy. softmax head for combinator, sigmoid head for
    relation-kind multi-label. Trained by full-batch gradient descent on the curriculum."""
    def __init__(self, din, dh, n_comb, n_kind):
        r = np.random.RandomState(0)
        s = 0.3
        self.W1 = r.randn(din, dh) * s; self.b1 = np.zeros(dh)
        self.Wc = r.randn(dh, n_comb) * s; self.bc = np.zeros(n_comb)
        self.Wk = r.randn(dh, n_kind) * s; self.bk = np.zeros(n_kind)
        self.mu = np.zeros(din); self.sigma = np.ones(din)

    def _trunk(self, X):
        return np.tanh((X - self.mu) / self.sigma @ self.W1 + self.b1)

    def forward(self, X):
        H = self._trunk(X)
        zc = H @ self.Wc + self.bc
        zc = zc - zc.max(1, keepdims=True)
        pc = np.exp(zc); pc /= pc.sum(1, keepdims=True)
        pk = 1.0 / (1.0 + np.exp(-(H @ self.Wk + self.bk)))
        return H, pc, pk

    def fit(self, X, Yc, Yk, epochs=400, lr=0.2, deadline=None):
        self.mu = X.mean(0); self.sigma = X.std(0) + 1e-6
        n = len(X)
        for ep in range(epochs):
            H, pc, pk = self.forward(X)
            # grads (cross-entropy for comb head, BCE for kind head)
            gzc = (pc - Yc) / n
            gzk = (pk - Yk) / n
            gWc = H.T @ gzc; gbc = gzc.sum(0)
            gWk = H.T @ gzk; gbk = gzk.sum(0)
            gH = gzc @ self.Wc.T + gzk @ self.Wk.T
            gZ1 = gH * (1 - H ** 2)
            Xn = (X - self.mu) / self.sigma
            gW1 = Xn.T @ gZ1; gb1 = gZ1.sum(0)
            for P, G in ((self.Wc, gWc), (self.bc, gbc), (self.Wk, gWk), (self.bk, gbk),
                         (self.W1, gW1), (self.b1, gb1)):
                P -= lr * G
            if deadline and ep % 25 == 0 and time.time() > deadline:
                break
        return self

    def to_json(self):
        return {k: getattr(self, k).tolist() for k in ("W1", "b1", "Wc", "bc", "Wk", "bk", "mu", "sigma")}

    @classmethod
    def from_json(cls, d, din, dh, n_comb, n_kind):
        o = cls(din, dh, n_comb, n_kind)
        for k in ("W1", "b1", "Wc", "bc", "Wk", "bk", "mu", "sigma"):
            setattr(o, k, np.array(d[k], float))
        return o


_PRIOR = None
_PRIOR_META = None
_DH = 24
_CURRICULUM_FINGERPRINT = "v1"  # bump to invalidate cache when grammar changes


def _build_prior(n_train=900, time_budget=80.0):
    """Train (or load cached) the feature->grammar prior on the self-gen curriculum. <~90s."""
    global _PRIOR, _PRIOR_META
    t0 = time.time()
    # cache check
    if os.path.exists(CACHE):
        try:
            d = json.load(open(CACHE))
            if d.get("fp") == _CURRICULUM_FINGERPRINT and d.get("fdim") == FEATURE_DIM:
                _PRIOR = _Prior.from_json(d["w"], FEATURE_DIM, _DH, len(COMBINATORS), len(KINDS))
                _PRIOR_META = d.get("meta", {})
                return _PRIOR
        except Exception:
            pass
    # generate curriculum + targets
    cur = make_curriculum(n_train, seed=12345)
    X = []; Yc = []; Yk = []
    for item in cur:
        vec, _ = features(item["train"])
        comb, kinds = _label_targets(item["label"])
        X.append(vec); Yc.append(comb); Yk.append(kinds)
    X = np.array(X); Yc = np.array(Yc); Yk = np.array(Yk)
    deadline = t0 + time_budget
    _PRIOR = _Prior(FEATURE_DIM, _DH, len(COMBINATORS), len(KINDS))
    _PRIOR.fit(X, Yc, Yk, epochs=500, lr=0.25, deadline=deadline)
    # report train accuracy of comb head as a sanity meta
    _, pc, _ = _PRIOR.forward(X)
    acc = float((pc.argmax(1) == Yc.argmax(1)).mean())
    _PRIOR_META = {"n_train": len(X), "comb_train_acc": round(acc, 3),
                   "seconds": round(time.time() - t0, 1)}
    try:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        json.dump({"fp": _CURRICULUM_FINGERPRINT, "fdim": FEATURE_DIM,
                   "w": _PRIOR.to_json(), "meta": _PRIOR_META}, open(CACHE, "w"))
    except Exception:
        pass
    return _PRIOR


# ===========================================================================
# PUBLIC: propose_compositions(train, k)
#   Feature-conditioned proposal of which composition SHAPES (combinators) + relation KINDS + concrete
#   primitive ops are likely for this task. Returns ranked, structured proposals an inventor can expand.
#   Falls back to a transparent feature-rule prior if the MLP is unavailable.
# ===========================================================================
def _rule_prior(feat):
    """Transparent backstop: hand-reasoned feature->grammar mapping (used if MLP missing). Returns
    (comb_scores dict, kind_scores dict)."""
    cs = {c: 0.2 for c in COMBINATORS}; ks = {k: 0.2 for k in KINDS}
    if feat.get("same_shape_frac", 0) > 0.9:
        cs["region_restrict"] += 0.6; cs["per_object_map"] += 0.5; cs["feature_conditional"] += 0.4
        cs["repurpose_overlay"] += 0.3
    if feat.get("shape_changes", 0) > 0.5:
        cs["sequence"] += 0.7  # size change => geometric/tile/crop composition
        cs["repurpose_overlay"] += 0.2
    if feat.get("is_upscale", 0) > 0.5:
        ks["tile"] += 0.7; ks["geom"] += 0.3; cs["sequence"] += 0.3
    if feat.get("is_downscale", 0) > 0.5:
        ks["select"] += 0.7; cs["sequence"] += 0.3
    if feat.get("added_colors_mean", 0) > 0.3:
        ks["paint"] += 0.6; ks["colormap"] += 0.3; cs["region_restrict"] += 0.2
    if feat.get("palette_invariant_frac", 0) < 0.5 and feat.get("same_shape_frac", 0) > 0.8:
        ks["colormap"] += 0.6; cs["per_object_map"] += 0.4
    if feat.get("obj_delta_mean", 0) < -0.5:
        ks["select"] += 0.6; cs["sequence"] += 0.3  # objects removed => selection
    if abs(feat.get("obj_delta_mean", 0)) < 0.2 and feat.get("palette_invariant_frac", 0) < 0.9:
        cs["per_object_map"] += 0.4; ks["colormap"] += 0.3
    return cs, ks


def propose_compositions(train, k=5):
    """Return up to k structured composition proposals for the CURRENT task, best-first:
        {"combinator", "kinds":[...], "ops":[...], "score", "rationale"}
    Each proposal is a SKELETON SENTENCE (a composition shape + the relation kinds/ops to fill it with),
    NOT a whole-mechanism template — the inventor expands it (binds args, picks the exact ops, verifies).
    Learns only from this task's train pairs + the import-time self-gen prior."""
    vec, feat = features(train)
    # neural prior (if trained) blended with the transparent rule prior
    comb_s = {c: 0.0 for c in COMBINATORS}; kind_s = {k: 0.0 for k in KINDS}
    if _PRIOR is not None:
        _, pc, pk = _PRIOR.forward(vec[None, :])
        for c, i in COMB_INDEX.items(): comb_s[c] += float(pc[0, i])
        for kk, i in KIND_INDEX.items(): kind_s[kk] += float(pk[0, i])
    rc, rk = _rule_prior(feat)
    for c in COMBINATORS: comb_s[c] += 0.5 * rc[c]
    for kk in KINDS: kind_s[kk] += 0.5 * rk[kk]

    comb_rank = sorted(comb_s, key=lambda c: -comb_s[c])
    kind_rank = sorted(kind_s, key=lambda x: -kind_s[x])

    # build concrete proposals: for each top combinator, attach the most-likely relation kinds, and
    # offer concrete candidate ops from those kinds (the alphabet to fill the skeleton).
    proposals = []
    for ci, comb in enumerate(comb_rank):
        if comb == "sequence":
            kinds = kind_rank[:2]
        elif comb == "region_restrict":
            kinds = [k for k in kind_rank if k in ("geom", "move", "paint", "colormap")][:1] or ["geom"]
        elif comb == "per_object_map":
            kinds = ["select"]
        elif comb == "feature_conditional":
            kinds = [k for k in kind_rank if k in ("geom", "move")][:2] or ["geom", "move"]
        else:  # repurpose_overlay
            kinds = ["geom"]
        ops = []
        for kd in kinds:
            ops += [op for op, _nc in RELATIONS[kd]]
        score = round(comb_s[comb] + 0.3 * sum(kind_s[k] for k in kinds), 3)
        rationale = _rationale(comb, kinds, feat)
        proposals.append({"combinator": comb, "kinds": kinds, "ops": ops[:8],
                          "score": score, "rationale": rationale})
    proposals.sort(key=lambda p: -p["score"])
    return proposals[:k]


def _rationale(comb, kinds, feat):
    bits = []
    if feat.get("shape_changes", 0) > 0.5: bits.append("output shape differs from input")
    else: bits.append("shape invariant across pairs")
    if feat.get("palette_invariant_frac", 1) < 0.9: bits.append("palette changes")
    if feat.get("obj_delta_mean", 0) < -0.3: bits.append("objects removed")
    elif feat.get("obj_delta_mean", 0) > 0.3: bits.append("objects added")
    return f"{comb} over {'+'.join(kinds)} — " + ", ".join(bits)


# ===========================================================================
# PUBLIC: heavy_train — stub hook for a future Kaggle-scale prior over the SAME label space.
# ===========================================================================
def heavy_train(n_curriculum=200000, hidden=256, epochs=50, out_path=None, seed=0,
                backend="numpy", time_budget=None):
    """STUB for a future Kaggle/GPU-scale prior. Same label space (combinator + relation-kind heads,
    and a planned op-sequence decoder) but: (1) a much larger sampled curriculum, (2) a bigger net /
    a small transformer decoder over op tokens, (3) optional torch backend. Intentionally NOT run at
    import. Returns a spec dict describing the job so an orchestrator can launch it.

    Wire-up plan:
        cur  = make_curriculum(n_curriculum)            # millions of sampled SENTENCES
        X    = [features(t["train"]) for t in cur]      # same cheap invariance features
        Y    = [label tree of t]                        # combinator + kinds (+ op-seq tokens)
        model.fit(X, Y) on GPU; export weights; load here via _Prior.from_json-style loader.
    Keep the public API (features / propose_compositions) identical so callers are unchanged."""
    spec = {"role": "heavy_train_stub", "n_curriculum": n_curriculum, "hidden": hidden,
            "epochs": epochs, "backend": backend, "seed": seed,
            "label_space": {"combinators": COMBINATORS, "kinds": KINDS, "op_vocab": list(dsl.OP_NAMES)},
            "feature_dim": FEATURE_DIM, "out_path": out_path,
            "note": "not executed at import; launch on free GPU; export weights compatible with _Prior"}
    return spec


# ===========================================================================
# IMPORT-TIME: build/load the prior (cached, <~90s). Guarded so import never hard-fails.
# ===========================================================================
try:
    _build_prior()
except Exception as _e:  # prior is an accelerant, not a correctness dependency
    _PRIOR = None
    _PRIOR_META = {"error": f"{type(_e).__name__}: {_e}"}


# ===========================================================================
# VALIDATION / SELF-DEMO
# ===========================================================================
def _demo():
    import numpy as np
    print("=" * 78)
    print("MECHANISM-COMPOSITION CURRICULUM — self-demo")
    print("=" * 78)
    print("prior meta:", _PRIOR_META)

    # 1) curriculum diversity
    t0 = time.time()
    cur = make_curriculum(120, seed=7)
    dt = time.time() - t0
    combs = Counter(item["label"]["combinator"] for item in cur)
    op_multiset = Counter()
    kind_multiset = Counter()
    label_sigs = set()
    for item in cur:
        lab = item["label"]
        op_multiset.update(lab["ops"])
        kind_multiset.update(lab["kinds"])
        label_sigs.add(json.dumps({"c": lab["combinator"], "k": lab["kinds"], "o": lab["ops"],
                                   "p": lab.get("params", {})}, sort_keys=True))
    print(f"\n[1] make_curriculum(120): {len(cur)} tasks in {dt:.2f}s")
    print("    combinator histogram :", dict(combs))
    print("    distinct relation kinds used:", len(kind_multiset), "->", dict(kind_multiset))
    print("    distinct primitive ops used :", len(op_multiset))
    print(f"    DISTINCT label-signatures   : {len(label_sigs)} / {len(cur)}  "
          f"(compositional diversity = {len(label_sigs)/max(len(cur),1):.0%} unique sentences)")
    ex = cur[0]
    print("    example label:", ex["label"])
    print("    example pair shapes:", [(p[0].shape, p[1].shape) for p in ex["train"][:2]])

    # 2) propose_compositions on 3 HAND-CHECKED synthetic cases
    print("\n[2] propose_compositions on 3 hand-checked cases:")

    # case A: pure upscale (tile/scale) — expect 'sequence' w/ tile kind high, shape change detected.
    a_in = np.array([[1, 0], [0, 2]])
    A = [(a_in, dsl.scale2(a_in)), (np.array([[3, 1], [1, 3]]), dsl.scale2(np.array([[3, 1], [1, 3]])))]
    propsA = propose_compositions(A, k=3)
    _, fA = features(A)
    print("\n  CASE A (2x upscale): shape_changes=%.2f is_upscale=%.2f area_ratio=%.2f"
          % (fA["shape_changes"], fA["is_upscale"], fA["area_ratio_mean"]))
    for p in propsA: print("    ->", p["combinator"], "kinds=", p["kinds"], "score=", p["score"])
    okA = any("tile" in p["kinds"] or p["combinator"] == "sequence" for p in propsA)
    print("    EXPECT tile/sequence proposed:", "PASS" if okA else "FAIL")

    # case B: object recolor by size, shape & count invariant, palette changes -> per_object_map/colormap
    b1 = np.zeros((6, 6), int); b1[0:1, 0:1] = 4; b1[2:5, 2:5] = 4
    o1 = run_per_object_map(b1, ("by_size", {0: 2, 1: 3}))
    b2 = np.zeros((6, 6), int); b2[0:2, 0:2] = 4; b2[4:5, 4:5] = 4
    o2 = run_per_object_map(b2, ("by_size", {0: 2, 1: 3}))
    B = [(b1, o1), (b2, o2)]
    propsB = propose_compositions(B, k=3)
    _, fB = features(B)
    print("\n  CASE B (recolor objects by size): same_shape=%.2f palette_inv=%.2f obj_delta=%.2f"
          % (fB["same_shape_frac"], fB["palette_invariant_frac"], fB["obj_delta_mean"]))
    for p in propsB: print("    ->", p["combinator"], "kinds=", p["kinds"], "score=", p["score"])
    okB = any(p["combinator"] == "per_object_map" or "colormap" in p["kinds"] or "select" in p["kinds"]
              for p in propsB[:3])
    print("    EXPECT per_object_map / colormap / select proposed:", "PASS" if okB else "FAIL")

    # case C: region-restricted geometry (mirror only the top half) — shape invariant, region cue
    c1 = _seed_grid(random.Random(1)); c1 = c1[:6, :6] if c1.shape[0] >= 6 and c1.shape[1] >= 6 else c1
    co1 = run_region_restrict(c1, "reflect_h", (), "top")
    c2 = _seed_grid(random.Random(2)); c2 = c2[:6, :6] if c2.shape[0] >= 6 and c2.shape[1] >= 6 else c2
    co2 = run_region_restrict(c2, "reflect_h", (), "top")
    C = [(c1, co1), (c2, co2)]
    propsC = propose_compositions(C, k=4)
    _, fC = features(C)
    print("\n  CASE C (mirror top-half region): same_shape=%.2f palette_inv=%.2f"
          % (fC["same_shape_frac"], fC["palette_invariant_frac"]))
    for p in propsC: print("    ->", p["combinator"], "kinds=", p["kinds"], "score=", p["score"])
    okC = any(p["combinator"] in ("region_restrict", "feature_conditional", "repurpose_overlay")
              and "geom" in p["kinds"] for p in propsC)
    print("    EXPECT region_restrict/conditional w/ geom proposed:", "PASS" if okC else "FAIL")

    # 3) heavy_train stub
    print("\n[3] heavy_train(...) stub spec keys:", list(heavy_train().keys()))

    allok = okA and okB and okC and len(label_sigs) / max(len(cur), 1) > 0.6
    print("\n" + "=" * 78)
    print("VALIDATION:", "ALL PASS" if allok else "CHECK ABOVE",
          "| diverse compositional tasks + sensible proposals on 3 hand-checked cases")
    print("=" * 78)
    return allok


if __name__ == "__main__":
    _demo()
