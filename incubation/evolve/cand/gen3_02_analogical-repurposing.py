#!/usr/bin/env python3
"""GEN-3 INVENTOR #2 — ANALOGICAL REPURPOSING.

THESIS (creativity = grasp + invention).
  (1) UNRESTRICTED grasp of cause-and-effect: read a task's INVARIANT causal structure off its train
      pairs as a SIGNATURE (a feature vector of what is invariant input->output across examples), the
      cross-example invariance that licenses causal (not correlational) induction; the held-out test is
      the exact intervention the verifier checks.
  (2) Real-time INVENTION of the mechanism, not retrieval of a whole template from a fixed menu. Here
      invention is ANALOGICAL REPURPOSING: find the NEAREST mechanism in an experience store by causal
      signature, then REPURPOSE it — deploy its cause->effect RELATION in the NEW task's role even when
      the surface differs (functional re-use-out-of-context), RE-FITTING its free parameters (which ops
      fill the combinator slots, which colors, which region, which object ordering) to the current train
      pairs, and EXACT-VERIFYING. A repurposed mechanism is a SENTENCE synthesized for this task; a single
      whole-template retrieval cannot produce it (that is exactly what the ablation removes).

ALPHABET vs SENTENCE. The ALPHABET = dsl.py's ops (knowledge, freely reused) grouped into relation KINDS,
plus the curriculum's grid combinators (sequence / region_restrict / per_object_map / feature_conditional
/ repurpose_overlay). A MECHANISM (sentence) = a combinator skeleton with its slots FILLED + parameters
BOUND. Our experience store holds mechanism SKELETONS (combinator + slot-kinds), each carrying a re-FITTER
that searches the small per-task arg space to instantiate the sentence for THIS task. Repurposing = taking
a skeleton whose stored binding came from a DIFFERENT surface and re-binding it here.

HOW solve() WORKS.
  a. signature(train): the curriculum's invariance features of the current task (input-only of test never
     touched). This is the causal grasp.
  b. propose_compositions(train): the experience-prior (the curriculum's feature->grammar MLP) ranks which
     combinator SHAPES + relation KINDS fit this signature -> fast candidate skeletons.
  c. NEAREST-MECHANISM RETRIEVAL: rank stored experience-mechanisms by signature distance (cosine over the
     normalized feature vector), blended with the prior's combinator scores.
  d. REPURPOSE + RE-FIT: for each nearest skeleton, run its fitter against THIS task's train pairs. The
     fitter enumerates the small per-task slot space (ops from the proposed kinds, colors from the palette,
     regions, object orderings) and keeps only fills that EXACTLY reproduce every train output. This is the
     invention: the relation is redeployed in a new role with freshly-fitted parameters.
  e. VERIFY + EMIT: apply the verified mechanism to each test input; ARC 2 attempts, best-first by MDL
     (skeleton complexity, then arg count).
  f. LEARN: a verified mechanism is added to the experience store (signature + skeleton + the concrete
     fitted binding) so later tasks retrieve & repurpose it (in-session transfer).

ABLATION (certifies invention). solve_ablated = EXACT-REUSE ONLY: replay stored mechanisms with their
STORED concrete bindings (no re-fit, no adaptation, no repurposing, no fresh enumeration) + the trivial
identity. That is single-whole-template retrieval. INVENTED = solved(full) - solved(ablated): solves that
required the repurposing/re-fit the ablation forbids.

reset_library() clears the cross-task store (for the transfer gate).

INTEGRITY. solve()/solve_ablated() learn ONLY from (a) the current task's train pairs, (b) module-level
state from PRIOR verified solve() calls this run, (c) self-generated synthetic data built at import (the
curriculum store). They NEVER read ARC task files or test OUTPUTS (test INPUTS only), no network, no LLM.
Budget respected. Pure python + numpy. Build-time < ~90s (seeding the store reuses the cached prior).
Run/imported with /data/llm/.venv/bin/python from .../incubation/evolve."""
import sys, os, time
from collections import Counter
import numpy as np

ARC = "/data/Windows-files/Documents/airfoil/incubation/arc"
if ARC not in sys.path:
    sys.path.insert(0, ARC)
import dsl

EVOLVE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EVOLVE not in sys.path:
    sys.path.insert(0, EVOLVE)
import mechanism_curriculum as mc  # experience-prior + combinators + features (the curriculum)

META = {"name": "analogical_repurposing_v1",
        "desc": "grasp the task's causal SIGNATURE, retrieve the NEAREST experience-mechanism, REPURPOSE it "
                "by re-fitting its slots to the current train pairs, exact-verify; invention = repurposed "
                "re-fits an exact-reuse ablation cannot reach; in-session experience store grows + transfers"}


# ===========================================================================
# RELATION KINDS we fill combinator slots from (the alphabet, grouped).
# A superset of the curriculum's RELATIONS so repurposing has room to re-bind a different op than was
# originally stored — that re-binding IS the functional repurposing.
# ===========================================================================
GEOM = ["reflect_h", "reflect_v", "rot90", "rot180", "rot270", "transpose", "sym_lr", "sym_ud"]
MOVE = ["shift_up", "shift_down", "shift_left", "shift_right",
        "gravity_down", "gravity_up", "gravity_left", "gravity_right"]
PAINT = ["fill_holes", "bbox_fill", "outline"]                       # 1 color arg
COLORMAP = ["recolor", "swap_colors"]                                # 2 color args
SELECT0 = ["largest_object", "keep_smallest", "crop_content", "trim_border"]  # 0-arg selectors
SELECT1 = ["keep_color", "remove_color"]                            # 1 color arg
TILE = ["tile_h2", "tile_v2", "tile_2x2", "scale2"]

KIND_OPS = {"geom": GEOM, "move": MOVE, "paint": PAINT, "colormap": COLORMAP,
            "select": SELECT0 + SELECT1, "tile": TILE}
_NC = {name: nc for name, (_fn, nc) in dsl.OPS.items()}


def _palette_nonzero(train):
    cs = set()
    for gi, go in train:
        cs |= set(np.unique(gi).tolist()) | set(np.unique(go).tolist())
    cs.discard(0)
    return sorted(cs)


def _train_hash(train):
    """Stable hash of a task's train pairs — identifies the EXACT task. Used to forbid the ablation from
    'retrieving' a binding that the full solver just learned from THIS SAME task (self-leakage, not
    experience-from-other-tasks). The invention gate runs solve() then solve_ablated() on the same train
    object, so without this guard a fresh invention would be mis-tagged 'retrieved'."""
    h = []
    for gi, go in train:
        h.append((gi.shape, go.shape, gi.tobytes(), go.tobytes()))
    return hash(tuple(h))


def _ops_for_kind(kind, palette, cap=None):
    """All concrete (op,args) instances for a relation kind, colors drawn from the task palette."""
    out = []
    for name in KIND_OPS.get(kind, []):
        nc = _NC.get(name, 0)
        if nc == 0:
            out.append((name, ()))
        elif nc == 1:
            for c in palette:
                out.append((name, (c,)))
        else:  # 2 color args, a != b
            for a in palette:
                for b in palette:
                    if a != b:
                        out.append((name, (a, b)))
    if cap and len(out) > cap:
        out = out[:cap]
    return out


# ===========================================================================
# MECHANISM EXECUTION — a "binding" is a fully-instantiated sentence. We build a callable from it.
# Combinators reuse the curriculum's interpreters so the grammar is shared with the experience-prior.
# ===========================================================================
def _exec_binding(g, binding):
    comb = binding["combinator"]
    try:
        if comb == "sequence":
            return mc.run_sequence(g, binding["steps"])
        if comb == "region_restrict":
            op, args = binding["step"]
            return mc.run_region_restrict(g, op, args, binding["region"])
        if comb == "per_object_map":
            return mc.run_per_object_map(g, binding["rule"], diag=binding.get("diag", False))
        if comb == "feature_conditional":
            return mc.run_feature_conditional(g, binding["predicate"], binding["then"], binding["else"])
        if comb == "repurpose_overlay":
            (bo, ba) = binding["base"]; (oo, oa) = binding["overlay"]
            return mc.run_repurpose_overlay(g, bo, ba, oo, oa)
        if comb == "color_perm":
            return _run_color_perm(g, binding["map"])
        if comb == "symmetry_repair":
            return _run_symmetry_repair(g, binding["sym"], binding["hole"])
        if comb == "local_rule":
            return _run_local_rule(g, binding["table"], binding["bg"], binding.get("orient", True))
        if comb == "identity":
            return g
    except Exception:
        return None
    return None


# ---- extended combinator interpreters (new abstraction sentences, re-fit per task) ----
def _run_color_perm(g, cmap):
    """ABSTRACTION: a learned cell-wise color relabeling (a colormap induced as a function color->color)."""
    out = g.copy()
    for a, b in cmap.items():
        out[g == a] = b
    return out


_SYMS = {
    "lr": lambda g: g[:, ::-1],
    "ud": lambda g: g[::-1, :],
    "rot180": lambda g: g[::-1, ::-1],
    "transpose": lambda g: g.T,
    "anti": lambda g: g[::-1, ::-1].T,
}


def _run_symmetry_repair(g, sym, hole):
    """REPURPOSE the grid's own symmetry image as the cause: where a cell holds the `hole` color, fill it
    from its symmetric partner. The symmetry axis + occlusion color are induced from train."""
    img = _SYMS[sym](g)
    if img.shape != g.shape:
        return None
    out = g.copy()
    m = (g == hole) & (img != hole)
    out[m] = img[m]
    return out


def _run_local_rule(g, table, bg, orient):
    """ABSTRACTION: output[i,j] = f(3x3 neighborhood of input). `table` maps a neighborhood key -> color,
    learned from train. Cells whose neighborhood is unseen keep their input value. orient=True keys on the
    raw 3x3; orient=False keys on a canonical (dihedral-min) 3x3 for invariance."""
    h, w = g.shape
    pad = np.full((h + 2, w + 2), bg, int)
    pad[1:-1, 1:-1] = g
    out = g.copy()
    for i in range(h):
        for j in range(w):
            patch = pad[i:i + 3, j:j + 3]
            key = _patch_key(patch, orient)
            v = table.get(key)
            if v is not None:
                out[i, j] = v
    return out


def _patch_key(patch, orient):
    if orient:
        return patch.tobytes()
    best = None
    p = patch
    for _ in range(4):
        for q in (p, p[:, ::-1]):
            b = q.tobytes()
            if best is None or b < best:
                best = b
        p = np.rot90(p)
    return best


def _verifies(binding, train):
    for gi, go in train:
        out = _exec_binding(gi, binding)
        if out is None or out.shape != go.shape or not np.array_equal(out, go):
            return False
    return True


def _binding_cost(binding):
    """MDL: prefer simpler sentences. Skeleton size + total arg count."""
    comb = binding["combinator"]
    if comb == "sequence":
        return 10 * len(binding["steps"]) + sum(len(a) for _, a in binding["steps"])
    if comb == "region_restrict":
        return 14 + len(binding["step"][1])
    if comb == "per_object_map":
        return 16 + binding["rule"][1].__len__() if isinstance(binding["rule"][1], dict) else 16
    if comb == "feature_conditional":
        return 18
    if comb == "repurpose_overlay":
        return 20
    if comb == "color_perm":
        return 8 + len(binding["map"])          # cheap, precise
    if comb == "symmetry_repair":
        return 12
    if comb == "local_rule":
        return 22 + min(len(binding["table"]), 30) // 5
    return 1  # identity


# ===========================================================================
# THE FITTERS — one per combinator SKELETON. Each enumerates the SMALL per-task arg space (ops from the
# proposed kinds, palette colors, regions, object orderings) and yields concrete bindings that EXACTLY
# reproduce every train output. This enumeration over the current task is the RE-FIT / repurposing step.
# Each fitter respects an exec-budget counter so we never blow the per-task budget.
# ===========================================================================
class _Budget:
    def __init__(self, n):
        self.n = n; self.used = 0
    def ok(self):
        return self.used < self.n
    def tick(self, k=1):
        self.used += k


def _fit_sequence(train, kinds, palette, bud, max_len=3):
    """Fit a composition (sentence) of length up to max_len, drawing each step's op from the candidate
    kinds (best-first), keeping any prefix that already verifies. Beam over partial compositions scored
    by mean grid-distance to the targets (the curriculum's 'grasp by invariance' made operational)."""
    ins = [gi for gi, _ in train]; tgt = [go for _, go in train]

    # candidate op instances, ordered: geometry/tile/move first (cheap structural), then paint/colormap/select
    order = ["geom", "tile", "move", "select", "paint", "colormap"]
    cand = []
    for k in order:
        if k in kinds:
            cand += _ops_for_kind(k, palette)
    # also always allow geom/tile as connective tissue even if not top-proposed (composition needs them)
    for k in ("geom", "tile"):
        if k not in kinds:
            cand += _ops_for_kind(k, palette)
    if not cand:
        return []

    def gdist(outs):
        s = 0.0
        for a, b in zip(outs, tgt):
            if a is None:
                s += 3.0
            elif a.shape != b.shape:
                s += 1.0 + abs(a.size - b.size) / max(a.size, b.size, 1)
            else:
                s += float((a != b).mean())
        return s / len(outs)

    found = []
    beam = [([], ins, gdist(ins))]
    for _depth in range(max_len):
        nxt = []
        for prog, outs, _sc in beam:
            for (op, args) in cand:
                if not bud.ok():
                    break
                bud.tick()
                try:
                    outs2 = [dsl.OPS[op][0](g, *args) for g in outs]
                except Exception:
                    continue
                if any(o is None for o in outs2):
                    continue
                prog2 = prog + [(op, args)]
                if all(o.shape == t.shape and np.array_equal(o, t) for o, t in zip(outs2, tgt)):
                    found.append({"combinator": "sequence", "steps": prog2})
                    if len(found) >= 4:
                        return found
                else:
                    nxt.append((prog2, outs2, gdist(outs2)))
            if not bud.ok():
                break
        nxt.sort(key=lambda x: x[2])
        beam = nxt[:18]
        if not beam or not bud.ok():
            break
    return found


def _fit_region_restrict(train, kinds, palette, bud):
    """Fit: apply ONE relation inside a sub-region, rest invariant. Re-bind region + op + colors."""
    found = []
    rkinds = [k for k in ("geom", "move", "paint", "colormap") if k in kinds] or ["geom", "move", "paint", "colormap"]
    cand = []
    for k in rkinds:
        cand += _ops_for_kind(k, palette)
    for region in ("top", "bot", "left", "right", "bbox"):
        for (op, args) in cand:
            if not bud.ok():
                return found
            bud.tick()
            b = {"combinator": "region_restrict", "step": (op, args), "region": region}
            if _verifies(b, train):
                found.append(b)
                if len(found) >= 3:
                    return found
    return found


def _fit_per_object_map(train, palette, bud):
    """ABSTRACTION over objects: recolor each component by a size ORDERING. Fit the ordinal->color table
    and the connectivity (4/8) directly from train: the mapping a property->color is INDUCED, not bound."""
    found = []
    for diag in (False, True):
        # induce the ordinal->color table from the FIRST train pair, then verify on all.
        gi0, go0 = train[0]
        comps = mc._components(gi0, diag=diag)
        if not comps:
            continue
        order = sorted(range(len(comps)), key=lambda i: len(comps[i]))
        table = {}
        ok = True
        for ordinal, idx in enumerate(order):
            cols = {int(go0[a, b]) for (a, b) in comps[idx]}
            if len(cols) != 1:
                ok = False; break
            table[ordinal] = cols.pop()
        if not ok or not table:
            continue
        for mode in ("by_size", "by_rank"):
            if not bud.ok():
                return found
            bud.tick()
            b = {"combinator": "per_object_map", "rule": (mode, table), "diag": diag}
            if _verifies(b, train):
                found.append(b)
                if len(found) >= 2:
                    return found
    return found


def _fit_feature_conditional(train, kinds, palette, bud):
    """CONDITIONAL: pick the relation by a measured grid feature. Re-bind predicate + both branch ops."""
    found = []
    bkinds = [k for k in ("geom", "move") if k in kinds] or ["geom", "move"]
    cand = []
    for k in bkinds:
        cand += _ops_for_kind(k, palette)
    if len(cand) > 14:
        cand = cand[:14]
    for pred in ("tall", "wide", "many_obj", "dense"):
        for (o1, a1) in cand:
            for (o2, a2) in cand:
                if (o1, a1) == (o2, a2):
                    continue
                if not bud.ok():
                    return found
                bud.tick()
                b = {"combinator": "feature_conditional", "predicate": pred,
                     "then": (o1, a1), "else": (o2, a2)}
                if _verifies(b, train):
                    found.append(b)
                    if len(found) >= 2:
                        return found
    return found


def _fit_repurpose_overlay(train, palette, bud):
    """REPURPOSE two geometric views into a new relation by overlaying one onto the other."""
    found = []
    cand = _ops_for_kind("geom", palette)
    for (bo, ba) in cand:
        for (oo, oa) in cand:
            if bo == oo:
                continue
            if not bud.ok():
                return found
            bud.tick()
            b = {"combinator": "repurpose_overlay", "base": (bo, ba), "overlay": (oo, oa)}
            if _verifies(b, train):
                found.append(b)
                if len(found) >= 2:
                    return found
    return found


def _fit_color_perm(train, bud):
    """ABSTRACTION: induce a single cell-wise color relabeling color->color from train, verify on all.
    Same shape only. A genuine sentence (the map is INDUCED per task, never a fixed binding)."""
    if any(gi.shape != go.shape for gi, go in train):
        return []
    cmap = {}
    for gi, go in train:
        for k, v in zip(gi.ravel().tolist(), go.ravel().tolist()):
            if k in cmap and cmap[k] != v:
                return []
            cmap[k] = v
    # must be non-trivial (some color actually changes)
    if all(k == v for k, v in cmap.items()):
        return []
    bud.tick()
    b = {"combinator": "color_perm", "map": {int(k): int(v) for k, v in cmap.items()}}
    return [b] if _verifies(b, train) else []


def _fit_symmetry_repair(train, palette, bud):
    """REPURPOSE the grid's own symmetry as cause: induce (axis, occlusion-color) so filling occluded
    cells from their symmetric partner reproduces train. axis + hole color fitted per task."""
    if any(gi.shape != go.shape for gi, go in train):
        return []
    found = []
    holes = [0] + [c for c in palette]
    for sym in ("lr", "ud", "rot180", "transpose", "anti"):
        for hole in holes:
            if not bud.ok():
                return found
            bud.tick()
            b = {"combinator": "symmetry_repair", "sym": sym, "hole": int(hole)}
            if _verifies(b, train):
                found.append(b)
                if len(found) >= 2:
                    return found
    return found


def _fit_local_rule(train, bud, max_cells=900):
    """ABSTRACTION: output[i,j] = f(3x3 neighborhood). Build the neighborhood->color table from train and
    verify it is consistent + reproduces train. Skip if grids are large (cost guard)."""
    if any(gi.shape != go.shape for gi, go in train):
        return []
    if sum(gi.size for gi, _ in train) > max_cells:
        return []
    found = []
    for orient in (True, False):
        bg = int(Counter(int(v) for gi, _ in train for v in gi.ravel()).most_common(1)[0][0])
        table = {}
        consistent = True
        for gi, go in train:
            h, w = gi.shape
            pad = np.full((h + 2, w + 2), bg, int); pad[1:-1, 1:-1] = gi
            for i in range(h):
                for j in range(w):
                    key = _patch_key(pad[i:i + 3, j:j + 3], orient)
                    v = int(go[i, j])
                    if key in table and table[key] != v:
                        consistent = False; break
                    table[key] = v
                if not consistent:
                    break
            if not consistent:
                break
        if not consistent or not table:
            continue
        # non-trivial: must not be pure identity
        bud.tick()
        b = {"combinator": "local_rule", "table": table, "bg": bg, "orient": orient}
        if _verifies(b, train):
            found.append(b)
            if len(found) >= 1:
                return found
    return found


def _fit_skeleton(combinator, train, kinds, palette, bud):
    if combinator == "sequence":
        return _fit_sequence(train, kinds, palette, bud)
    if combinator == "region_restrict":
        return _fit_region_restrict(train, kinds, palette, bud)
    if combinator == "per_object_map":
        return _fit_per_object_map(train, palette, bud)
    if combinator == "feature_conditional":
        return _fit_feature_conditional(train, kinds, palette, bud)
    if combinator == "repurpose_overlay":
        return _fit_repurpose_overlay(train, palette, bud)
    if combinator == "color_perm":
        return _fit_color_perm(train, bud)
    if combinator == "symmetry_repair":
        return _fit_symmetry_repair(train, palette, bud)
    if combinator == "local_rule":
        return _fit_local_rule(train, bud)
    return []


# Extended combinators NOT in the curriculum prior's combinator head — proposed by signature heuristics
# (the experience-prior ranks the 5 curriculum shapes; these three are additional invention-eligible
# sentences whose re-fit is gated by the task signature so they are tried only when plausible).
_EXTRA_COMBINATORS = ["color_perm", "symmetry_repair", "local_rule"]


# ===========================================================================
# THE EXPERIENCE STORE — nearest-mechanism retrieval by causal signature.
#   Each entry: {sig (np vec), combinator, kinds, binding (concrete), source}
#   Seeded at import from the curriculum (skeletons + their generative bindings) and GROWN in-session
#   from verified solves. Retrieval = cosine-nearest signatures, so a mechanism learned on one surface is
#   redeployed (repurposed) on a different surface with a similar causal signature.
# ===========================================================================
def _signature(train):
    vec, feat = mc.features(train)
    return vec, feat


def _normsig(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _curriculum_binding(label, train_pairs):
    """Reconstruct a concrete binding from a curriculum item by RE-FITTING its skeleton to its own pairs
    (we only keep the skeleton + a verified concrete binding, never any ARC data)."""
    comb = label["combinator"]
    pal = _palette_nonzero(train_pairs)
    bud = _Budget(4000)
    fills = _fit_skeleton(comb, train_pairs, set(label.get("kinds", [])) | set(["geom", "tile", "move", "select", "paint", "colormap"]), pal, bud)
    return fills[0] if fills else None


_STORE = []                 # cross-task experience store (list of entries); persists across solve() calls
_SEED_STORE = []            # the import-time seed (curriculum mechanisms); never mutated
_LIB_HITS = Counter()       # combinator -> #verified solves this run (experience prior on attempt order)


def _seed_store(n=140, seed=4242):
    """Seed the experience store with curriculum mechanisms: signature + skeleton + a verified concrete
    binding. SELF-GENERATED at import; reads no ARC data."""
    store = []
    cur = mc.make_curriculum(n, seed=seed)
    for item in cur:
        label = item["label"]
        tp = item["train"]
        vec, _ = _signature(tp)
        binding = _curriculum_binding(label, tp)
        if binding is None:
            continue
        store.append({
            "sig": np.array(vec, float),
            "nsig": _normsig(np.array(vec, float)),
            "combinator": label["combinator"],
            "kinds": list(label.get("kinds", [])),
            "binding": binding,
            "source": "curriculum",
        })
    return store


def reset_library():
    """Clear the cross-task experience grown in-session; restore to the import-time curriculum seed.
    Used by the transfer gate to force a cold run. The self-gen seed is NOT experience-from-other-ARC-
    tasks, so it stays (it is the import-time prior, allowed)."""
    global _STORE, _LIB_HITS
    _STORE = [dict(e) for e in _SEED_STORE]
    _LIB_HITS = Counter()


def _retrieve(sig_n, comb_scores, k=6):
    """Rank store entries by cosine signature similarity blended with the prior's combinator score and an
    experience discount for combinators that have already paid off this run. Returns combinators+kinds to
    try, best-first (deduped) — the ANALOGY step (nearest cause-effect structure)."""
    scored = []
    for e in _STORE:
        cos = float(np.dot(sig_n, e["nsig"]))
        cscore = comb_scores.get(e["combinator"], 0.0)
        hit = 0.15 * _LIB_HITS.get(e["combinator"], 0)
        scored.append((cos + 0.6 * cscore + hit, e))
    scored.sort(key=lambda x: -x[0])
    seen = set(); ranked = []
    for s, e in scored:
        key = (e["combinator"], tuple(sorted(e["kinds"])))
        if key in seen:
            continue
        seen.add(key)
        ranked.append(e)
        if len(ranked) >= k:
            break
    return ranked


# ===========================================================================
# solve() — analogical repurposing inventor
# ===========================================================================
def _attempts_from_bindings(bindings, test_inputs):
    """Best-first (already MDL-sorted) up to 2 attempts per test input."""
    attempts = []
    for gi in test_inputs:
        cand = []
        for b in bindings[:2]:
            o = _exec_binding(gi, b)
            if o is not None:
                cand.append(o)
        attempts.append(cand)
    return attempts


def solve(train, test_inputs, budget):
    if not train:
        return [[gi] for gi in test_inputs]
    t0 = time.time()
    bud = _Budget(max(800, int(budget)))
    palette = _palette_nonzero(train)
    sig, feat = _signature(train)
    sig_n = _normsig(sig)

    # (b) experience-prior: rank combinator SHAPES + relation KINDS for this signature (fast skeletons).
    try:
        props = mc.propose_compositions(train, k=5)
    except Exception:
        props = []
    comb_scores = {}
    kinds_by_comb = {}
    for p in props:
        comb_scores[p["combinator"]] = max(comb_scores.get(p["combinator"], 0.0), p["score"])
        kinds_by_comb.setdefault(p["combinator"], set()).update(p["kinds"])

    # (c) nearest-mechanism retrieval from the experience store (the ANALOGY).
    nearest = _retrieve(sig_n, comb_scores, k=6)

    # Build the ordered list of (combinator, kinds) skeletons to repurpose: nearest-store first, then any
    # high-prior combinator the store didn't surface (so a never-stored shape can still be invented).
    order = []
    seen = set()
    for e in nearest:
        key = (e["combinator"], tuple(sorted(set(e["kinds"]) | kinds_by_comb.get(e["combinator"], set()))))
        if key in seen:
            continue
        seen.add(key)
        order.append((e["combinator"], set(e["kinds"]) | kinds_by_comb.get(e["combinator"], set())))
    for c in sorted(comb_scores, key=lambda c: -comb_scores[c]):
        key = (c, tuple(sorted(kinds_by_comb.get(c, set()))))
        if key in seen:
            continue
        seen.add(key)
        order.append((c, kinds_by_comb.get(c, set()) or {"geom", "move", "tile"}))

    # EXTRA invention-eligible sentences, signature-gated (tried first when cheap+plausible; they induce a
    # table that verifies or fails fast). same-shape gates them all; palette-change favors color/local;
    # an occlusion-like signature (palette change small, objects ~invariant) favors symmetry repair.
    same_shape = feat.get("same_shape_frac", 0) > 0.99
    if same_shape:
        extra = []
        # color_perm: a clean palette relabel -> very cheap, very precise. Always try when same-shape.
        extra.append(("color_perm", set()))
        # symmetry_repair: when shape invariant (occlusion/repair signature).
        extra.append(("symmetry_repair", set()))
        # local_rule: when grids are small (cost guard inside the fitter).
        if sum(gi.size for gi, _ in train) <= 900:
            extra.append(("local_rule", set()))
        order = extra + order

    # (d) REPURPOSE + RE-FIT each skeleton against THIS task's train pairs; keep exact-verified bindings.
    verified = []
    for comb, kinds in order:
        if not bud.ok() or time.time() - t0 > 25:
            break
        sub = _Budget(max(300, bud.n - bud.used))  # share the remaining budget
        fills = _fit_skeleton(comb, train, kinds, palette, sub)
        bud.tick(sub.used)
        for b in fills:
            verified.append(b)
        if verified and comb in ("per_object_map", "region_restrict", "repurpose_overlay",
                                 "color_perm", "symmetry_repair", "local_rule"):
            # these are cheap+specific; if they verified we usually have the right mechanism
            break

    if not verified:
        return [[gi] for gi in test_inputs]  # no mechanism found; emit identity (never crash)

    # (e) MDL order, emit ARC 2 attempts.
    verified.sort(key=_binding_cost)
    # (f) LEARN: store the simplest verified mechanism with this task's signature for later repurposing.
    best = verified[0]
    _STORE.append({
        "sig": np.array(sig, float), "nsig": sig_n,
        "combinator": best["combinator"],
        "kinds": list(kinds_by_comb.get(best["combinator"], set())) or _binding_kinds(best),
        "binding": best, "source": "solved",
        "learned_from": _train_hash(train),   # guard: ablation must not retrieve THIS task's own binding
    })
    _LIB_HITS[best["combinator"]] += 1

    return _attempts_from_bindings(verified, test_inputs)


def _binding_kinds(binding):
    comb = binding["combinator"]
    if comb == "sequence":
        return list({mc.OP_KIND.get(op, "geom") for op, _ in binding["steps"]})
    if comb == "region_restrict":
        return [mc.OP_KIND.get(binding["step"][0], "geom")]
    if comb == "per_object_map":
        return ["select"]
    if comb == "feature_conditional":
        return list({mc.OP_KIND.get(binding["then"][0], "geom"), mc.OP_KIND.get(binding["else"][0], "geom")})
    if comb == "repurpose_overlay":
        return ["geom"]
    if comb == "color_perm":
        return ["colormap"]
    if comb == "symmetry_repair":
        return ["geom"]
    if comb == "local_rule":
        return ["paint"]
    return []


# ===========================================================================
# solve_ablated() — EXACT-REUSE ONLY (invention OFF): single-whole-template retrieval.
#   Replay stored mechanisms with their STORED concrete bindings (no re-fit, no enumeration, no
#   repurposing, no fresh arg search), plus the trivial identity. If a stored binding happens to already
#   reproduce this task's train outputs as-is, it counts (genuine retrieval). Otherwise the ablation
#   fails — and any solve the full path gets there is certified INVENTED.
# ===========================================================================
def solve_ablated(train, test_inputs, budget):
    if not train:
        return [[gi] for gi in test_inputs]
    # retrieval candidates = every stored binding, ordered by signature similarity (no re-fit at all).
    # EXCLUDE any binding the full solver learned from THIS SAME task (self-leakage, not experience):
    # retrieving the current task's own freshly-learned binding would not be a genuine whole-template
    # retrieval-from-experience, so it must not be allowed to defeat the invention certification.
    th = _train_hash(train)
    sig, _ = _signature(train)
    sig_n = _normsig(sig)
    scored = []
    for e in _STORE:
        if e.get("learned_from") == th:
            continue
        cos = float(np.dot(sig_n, e["nsig"]))
        scored.append((cos, e["binding"]))
    scored.sort(key=lambda x: -x[0])

    verified = []
    for _cos, binding in scored:
        if _verifies(binding, train):          # stored sentence reproduces train EXACTLY, as-is
            verified.append(binding)
            if len(verified) >= 2:
                break
    # identity is the degenerate whole-template; include it only if it actually solves train (rare).
    ident = {"combinator": "identity"}
    if _verifies(ident, train):
        verified.append(ident)

    if not verified:
        return [[gi] for gi in test_inputs]
    verified.sort(key=_binding_cost)
    return _attempts_from_bindings(verified, test_inputs)


# ===========================================================================
# IMPORT-TIME: seed the experience store from the curriculum (self-gen, cached prior). Guarded.
# ===========================================================================
def _init():
    global _SEED_STORE, _STORE
    try:
        _SEED_STORE = _seed_store()
    except Exception:
        _SEED_STORE = []
    _STORE = [dict(e) for e in _SEED_STORE]


_init()


# ===========================================================================
# self-demo
# ===========================================================================
if __name__ == "__main__":
    import json
    print("seed store size:", len(_SEED_STORE),
          "| combinators:", dict(Counter(e["combinator"] for e in _SEED_STORE)))
    # sanity: a hand-made 2x upscale (sequence/tile) should be INVENTED, and exact-reuse ablation handles
    # a stored mechanism replayed on its own surface.
    a = np.array([[1, 0], [0, 2]]); b = np.array([[3, 1], [1, 3]])
    train = [(a, dsl.scale2(a)), (b, dsl.scale2(b))]
    test_in = [np.array([[4, 5], [6, 7]])]
    att = solve(train, test_in, 3000)
    print("upscale solve attempt[0] shape:", None if not att[0] else att[0][0].shape,
          "expected", dsl.scale2(test_in[0]).shape,
          "correct:", bool(att[0]) and np.array_equal(att[0][0], dsl.scale2(test_in[0])))
    aab = solve_ablated(train, test_in, 3000)
    print("ablated upscale solved-train-as-is:", bool(aab[0]) and np.array_equal(aab[0][0], dsl.scale2(test_in[0])))
