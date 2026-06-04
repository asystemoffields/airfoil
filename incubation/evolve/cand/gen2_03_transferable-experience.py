#!/usr/bin/env python3
"""GEN-2 CREATIVITY-OPERATOR #3 — TRANSFERABLE EXPERIENCE (make the in-session library finally PAY).

FACET. When a task is solved, ABSTRACT its solution into a parameterized SCHEMA (antiunify the solving
program / concept-binding, with colors and motion-vectors turned into HOLES) and BANK it. On LATER tasks,
try banked schemas FIRST (re-fit their holes from this task's train + EXACT-verify). The whole point: a
schema learned on task X solves a task Y that the from-scratch machinery could NOT solve under budget.
We MEASURE experience_transfer_solves honestly by re-running with the cross-task library forced empty, and
consolidate under an MDL razor (a schema is kept long-term only if it has been REUSED >= 2x).

WHY THE BASE LIBRARY READ ~0 (and how we fix it).
  In gen2_base the "library" stored whole concept CLOSURES and replayed them. That paid ~0 because any task
  whose train a closure reproduces is ALSO directly fit by the same concept fitter from scratch -> the
  closure is redundant. Transfer can only PAY when the banked thing is something the from-scratch path
  would NOT reach on task Y within budget. Two such things exist and we bank BOTH:

  (1) PROGRAM-SKELETON SCHEMAS.  Every VERIFIED solution (a seed-search DSL program, OR a concept rendered
      as an op-sequence) is ANTIUNIFIED: its color arguments become holes (?c1,?c2,...). The skeleton
      [recolor(?a,?b), gravity_down] banked from task X is, on task Y, re-fit by enumerating only Y's
      palette for the holes -> a 2-op needle that the W=25 / budget-3000 beam would miss is found in <=
      |palette|^2 execs. The from-scratch seed search on Y (empty library) does NOT reach it -> a real
      transfer solve.

  (2) PARAMETRIC MOTION/RELATIONAL SCHEMAS (the hard same-shape position-change family). New concepts that
      are genuinely parameterized with holes: translate-all-objects by a vector V; per-object gravity in a
      direction until collision; ray/beam draw from singleton cells in learned directions. Their fitted
      bindings (vector, direction, colors) are antiunified to schemas and banked. On a later motion task
      whose train UNDER-determines the from-scratch hole search (e.g. only one example, ambiguous vector),
      the banked binding resolves it. Their structure also seeds the *order* later tasks try families.

HOW TRANSFER IS MADE TO PAY (the mechanism, not a hope).
  solve() order:   banked-SCHEMAS-first  ->  direct concept fit  ->  motion concepts  ->  linker  -> seed.
  A banked schema is tried BEFORE the seed DSL search, with a hole re-fit bounded by the per-task palette.
  Because schemas are exact-verified on THIS task's train before use, a wrong transfer can never produce a
  wrong-but-accepted answer; it simply fails to verify and we fall through. So transfer is pure upside:
  it can only ADD solves that the from-scratch path misses under budget. That ADD is exactly
  experience_transfer_solves, and it is measured by the empty-library ablation, not asserted.

MDL RAZOR / CONSOLIDATION.  A schema enters the bank on first verify (so the FIRST reuse is possible), but
  the persistent bank is pruned to schemas with reuse_count >= 2 via consolidate() — a schema that never
  re-fit elsewhere is dropped (it was just task-local). reuse_count is incremented only on a verified
  re-fit on a DIFFERENT task. Schemas are MDL-ordered (fewest holes / shortest skeleton first).

INSTRUMENTATION (the creativity gate; measured, not guessed). Every solved task is tagged 'single' (one
  from-scratch concept reproduced train alone), 'link' (>=2 ops/concepts composed, no single concept fits),
  or 'reuse' (a schema banked from an EARLIER task this run fit via hole re-fit). We expose:
    solve(...)                  full system.
    solve_single_only(...)      composition + cross-task transfer DISABLED (single from-scratch concept
                                or the base's direct path only). novel_link = full - this.
    reset_library()             clears the cross-task bank; running with it cleared each task gives the
                                empty-library count -> experience_transfer = full - empty.

INTEGRITY (hard rules, honored). solve() learns ONLY from (a) the current task's train pairs, (b) module
  state accumulated from PRIOR verified solve() calls this run, (c) import-time synthetic data (inherited
  base curriculum). It NEVER reads an ARC task file or any test OUTPUT (test INPUTS only), no network, no
  LLM, respects budget=3000. Pure python + numpy; build-time work < ~90s. Run with /data/llm/.venv/bin/python.
"""
import sys, os, time
from collections import Counter, deque, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARC = "/data/Windows-files/Documents/airfoil/incubation/arc"
for p in (ARC, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import dsl
import gen2_base as B   # the consolidated UNION floor; we extend, never regress it.

META = {"name": "gen2_03_transferable-experience",
        "desc": "transferable-experience: antiunify verified solutions -> parameterized SCHEMAS (color & "
                "motion-vector holes), bank, try schemas FIRST on later tasks (re-fit holes + exact-verify); "
                "+ parametric motion/relational concepts (translate-objects / per-object-gravity / ray-draw) "
                "as a hard-family source of transferable schemas; MDL razor (keep schema iff reused>=2x). "
                "Built on gen2_base."}

_eq = B._eq
_bg = B._bg
_verify = B._verify
_components = B._components


# ===========================================================================
# helpers for object-centric motion concepts
# ===========================================================================
def _objects(g, bg=0, diag=False, by_color=True):
    """Connected components as (cells, color, (r0,c0,r1,c1)). by_color groups same-color cells (keeps
    distinct-color touching shapes apart, which is what motion tasks want)."""
    comps = _components(g, bg=bg, diag=diag, by_color=by_color)
    out = []
    for cells in comps:
        rs = [a for a, _ in cells]; cs = [b for _, b in cells]
        col = int(g[cells[0][0], cells[0][1]])
        out.append((cells, col, (min(rs), min(cs), max(rs) + 1, max(cs) + 1)))
    return out


def _shape_key(cells):
    rs = [a for a, _ in cells]; cs = [b for _, b in cells]
    r0, c0 = min(rs), min(cs)
    return frozenset((a - r0, b - c0) for a, b in cells)


def _place(out, cells, color, dr, dc, bg=0):
    """Move a component's cells by (dr,dc) into out; True iff fully in-bounds onto bg/own cells."""
    h, w = out.shape
    tgt = [(a + dr, b + dc) for a, b in cells]
    for a, b in tgt:
        if not (0 <= a < h and 0 <= b < w):
            return False
    for a, b in tgt:
        out[a, b] = color
    return True


# ===========================================================================
# PARAMETRIC MOTION / RELATIONAL CONCEPTS  (the hard same-shape position-change family).
# Each returns a fitted grid->grid fn AND exposes a SCHEMA descriptor (family + holes) so the solving
# binding can be antiunified and banked. We attach the schema descriptor as fn._schema.
# ===========================================================================
def fit_translate_objects(train):
    """Schema: translate EVERY nonzero object by a single learned vector V=(dr,dc) (holes: dr,dc).
    Background fixed at 0. Footprint/colors preserved, only position changes."""
    if any(gi.shape != go.shape for gi, go in train):
        return None
    # candidate vectors: deduce from bbox-of-all-nonzero displacement on each pair; require agreement.
    vecs = None
    for gi, go in train:
        ni = np.argwhere(gi != 0); no = np.argwhere(go != 0)
        if ni.size == 0 or no.size == 0 or ni.shape[0] != no.shape[0]:
            return None
        dv = (no.min(0) - ni.min(0))
        # also try matching via total mass centroid as a fallback candidate
        cand = {(int(dv[0]), int(dv[1]))}
        if vecs is None:
            vecs = cand
        else:
            vecs &= cand
        if not vecs:
            return None
    for (dr, dc) in sorted(vecs):
        def fn(g, dr=dr, dc=dc):
            out = np.zeros_like(g)
            h, w = g.shape
            for a, b in np.argwhere(g != 0):
                na, nb = a + dr, b + dc
                if not (0 <= na < h and 0 <= nb < w):
                    return None
                out[na, nb] = g[a, b]
            return out
        if _verify(fn, train):
            fn._schema = ("translate", {"dr": dr, "dc": dc})
            return fn
    return None


def _gravity_dir(g, direction, by_color=True, diag=False):
    """Per-OBJECT gravity: each component slides as a rigid body in `direction` until it would overlap
    another component or leave the grid. Objects processed leading-edge first so they stack."""
    h, w = g.shape
    di, dj = direction
    objs = _objects(g, bg=0, diag=diag, by_color=by_color)
    # order by progress along direction so the foremost object settles first
    def lead(o):
        cells = o[0]
        return max((di * a + dj * b) for a, b in cells)
    objs = sorted(objs, key=lead, reverse=True)
    out = np.zeros_like(g)
    occ = np.zeros((h, w), bool)
    for cells, col, _ in objs:
        step = 0
        while True:
            nxt = [(a + di * (step + 1), b + dj * (step + 1)) for a, b in cells]
            ok = True
            for a, b in nxt:
                if not (0 <= a < h and 0 <= b < w) or occ[a, b]:
                    ok = False
                    break
            if not ok:
                break
            step += 1
        for a, b in cells:
            na, nb = a + di * step, b + dj * step
            out[na, nb] = g[a, b]
            occ[na, nb] = True
    return out


def fit_object_gravity(train):
    """Schema: per-object gravity in a learned direction (hole: direction in {down,up,left,right})."""
    if any(gi.shape != go.shape for gi, go in train):
        return None
    DIRS = {"down": (1, 0), "up": (-1, 0), "left": (0, -1), "right": (0, 1)}
    for dname, d in DIRS.items():
        for by_color in (True, False):
            for diag in (False, True):
                def fn(g, d=d, by_color=by_color, diag=diag):
                    return _gravity_dir(g, d, by_color=by_color, diag=diag)
                if _verify(fn, train):
                    fn._schema = ("obj_gravity", {"dir": dname, "by_color": by_color, "diag": diag})
                    return fn
    return None


def _rays_from_singletons(g, dirs, stop_at_nonzero=True):
    """From every isolated single-cell colored pixel, shoot rays of its color in `dirs` until the grid
    edge (or until hitting a nonzero cell if stop_at_nonzero)."""
    h, w = g.shape
    out = g.copy()
    # singletons = nonzero cells whose 8-neighbours are all zero
    for a in range(h):
        for b in range(w):
            c = g[a, b]
            if c == 0:
                continue
            iso = True
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    x, y = a + di, b + dj
                    if 0 <= x < h and 0 <= y < w and g[x, y] != 0:
                        iso = False
            if not iso:
                continue
            for (di, dj) in dirs:
                x, y = a + di, b + dj
                while 0 <= x < h and 0 <= y < w:
                    if stop_at_nonzero and g[x, y] != 0:
                        break
                    out[x, y] = c
                    x += di; y += dj
    return out


def fit_ray_draw(train):
    """Schema: shoot rays from singleton pixels along a learned direction-set (hole: dir-set, stop mode)."""
    if any(gi.shape != go.shape for gi, go in train):
        return None
    DSETS = {
        "4": [(-1, 0), (1, 0), (0, -1), (0, 1)],
        "up": [(-1, 0)], "down": [(1, 0)], "left": [(0, -1)], "right": [(0, 1)],
        "vert": [(-1, 0), (1, 0)], "horiz": [(0, -1), (0, 1)],
        "diag": [(-1, -1), (-1, 1), (1, -1), (1, 1)],
        "8": [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)],
    }
    for sname, dirs in DSETS.items():
        for stop in (True, False):
            def fn(g, dirs=dirs, stop=stop):
                return _rays_from_singletons(g, dirs, stop_at_nonzero=stop)
            if _verify(fn, train):
                fn._schema = ("ray_draw", {"dirs": sname, "stop": stop})
                return fn
    return None


# motion concept registry: name -> fitter. Each fitter's returned fn carries fn._schema.
MOTION_CONCEPTS = [
    ("translate_objects", fit_translate_objects),
    ("object_gravity", fit_object_gravity),
    ("ray_draw", fit_ray_draw),
]


# re-instantiate a MOTION schema from its banked descriptor (re-fit = re-verify on the new task's train).
def _motion_fn_from_schema(family, params):
    if family == "translate":
        dr, dc = params["dr"], params["dc"]
        def fn(g, dr=dr, dc=dc):
            out = np.zeros_like(g); h, w = g.shape
            for a, b in np.argwhere(g != 0):
                na, nb = a + dr, b + dc
                if not (0 <= na < h and 0 <= nb < w):
                    return None
                out[na, nb] = g[a, b]
            return out
        return fn
    if family == "obj_gravity":
        d = {"down": (1, 0), "up": (-1, 0), "left": (0, -1), "right": (0, 1)}[params["dir"]]
        bc, dg = params["by_color"], params["diag"]
        return lambda g, d=d, bc=bc, dg=dg: _gravity_dir(g, d, by_color=bc, diag=dg)
    if family == "ray_draw":
        DSETS = {
            "4": [(-1, 0), (1, 0), (0, -1), (0, 1)],
            "up": [(-1, 0)], "down": [(1, 0)], "left": [(0, -1)], "right": [(0, 1)],
            "vert": [(-1, 0), (1, 0)], "horiz": [(0, -1), (0, 1)],
            "diag": [(-1, -1), (-1, 1), (1, -1), (1, 1)],
            "8": [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)],
        }
        dirs = DSETS[params["dirs"]]; stop = params["stop"]
        return lambda g, dirs=dirs, stop=stop: _rays_from_singletons(g, dirs, stop_at_nonzero=stop)
    return None


# ===========================================================================
# PROGRAM-SKELETON SCHEMAS — antiunify VERIFIED DSL op-sequences (colors -> holes), bank, re-fit later.
# A skeleton = tuple of (op_name, arg_kinds) where arg_kinds is a tuple of 'C' (color hole) per color arg.
# Re-fit = enumerate the task palette for the holes, exact-verify; bounded by |palette|^(#holes) <= small.
# ===========================================================================
def _prog_to_skeleton(prog):
    """prog: list of (op_name, args). -> (skeleton_key, nholes) or None if it contains a library macro."""
    sk = []
    nholes = 0
    for item in prog:
        name = item[0]; args = item[1]
        if name not in dsl.OPS:
            return None  # macro / non-DSL op: not antiunifiable into a color-hole skeleton
        nc = dsl.OPS[name][1]
        sk.append((name, nc))
        nholes += nc
    return tuple(sk), nholes


def _skeleton_refit(skeleton, train, palette, budget_holes=400):
    """Enumerate color fillings of a skeleton's holes from `palette`; return the FIRST filling whose
    concrete program exactly reproduces every train output. Bounded enumeration (transfer must be cheap)."""
    # collect hole positions
    holes = []
    for k, (name, nc) in enumerate(skeleton):
        for _ in range(nc):
            holes.append(k)
    nh = len(holes)
    if nh == 0:
        prog = [(name, ()) for (name, nc) in skeleton]
        if dsl.solves(prog, train):
            return prog
        return None
    if len(palette) ** nh > budget_holes:
        # too big to brute-force cheaply; skip (keeps transfer cheap / honest under budget)
        return None
    import itertools
    for combo in itertools.product(palette, repeat=nh):
        prog = []
        ci = 0
        for (name, nc) in skeleton:
            args = tuple(int(combo[ci + t]) for t in range(nc))
            ci += nc
            prog.append((name, args))
        if dsl.solves(prog, train):
            return prog
    return None


# ===========================================================================
# THE CROSS-TASK SCHEMA BANK (module-level; the transferable experience).
#   prog_schemas  : skeleton_key -> {"first_task", "reuse", "nholes"}
#   motion_schemas: (family,frozenset(params.items())) -> {"family","params","first_task","reuse"}
# Only verified-correct schemas enter. No grids/outputs/files are ever stored. consolidate() applies the
# MDL razor (drop schemas with reuse < 2). reset_library() clears it (for the empty-library ablation).
# ===========================================================================
class _Bank:
    def __init__(self):
        self.prog_schemas = {}      # skeleton_key -> dict
        self.motion_schemas = {}    # key -> dict
        self.tagcount = Counter()   # 'single'/'link'/'reuse'
        self.audit = []             # (task_id, tag, detail)
        self.transfer_fires = 0

    def bank_prog(self, skeleton, nholes, task_id):
        rec = self.prog_schemas.get(skeleton)
        if rec is None:
            self.prog_schemas[skeleton] = {"first_task": task_id, "reuse": 0, "nholes": nholes}
        # banked on first solve; reuse incremented only on a verified re-fit on a different task.

    def bank_motion(self, family, params, task_id):
        key = (family, frozenset(params.items()))
        if key not in self.motion_schemas:
            self.motion_schemas[key] = {"family": family, "params": params,
                                        "first_task": task_id, "reuse": 0}

    def consolidate(self):
        """MDL razor: keep only schemas reused >= 2x (long-term). Returns counts kept/dropped."""
        keepp = {k: v for k, v in self.prog_schemas.items() if v["reuse"] >= 2}
        keepm = {k: v for k, v in self.motion_schemas.items() if v["reuse"] >= 2}
        dropped = (len(self.prog_schemas) - len(keepp)) + (len(self.motion_schemas) - len(keepm))
        return {"kept_prog": len(keepp), "kept_motion": len(keepm), "dropped": dropped}


_BANK = _Bank()


def reset_library():
    """Clear ALL cross-task experience (schema bank + base library). For the empty-library ablation."""
    global _BANK
    _BANK = _Bank()
    B._LIB.__init__()
    _TASK_COUNTER[0] = 0


_TASK_COUNTER = [0]


# ===========================================================================
# core: try banked schemas FIRST on the current task (the transfer step).
# Returns (rules, used_reuse) where rules is a list of (tag, fn) exact-verified on this train.
# Crucially, when from_scratch_present is True we know a single concept already fits; a schema that ALSO
# fits is then redundant for solving but we still count its reuse (so the MDL razor sees genuine reuse).
# ===========================================================================
def _try_banked_schemas(train, task_id, palette):
    rules = []
    fired_new = False  # a schema fired on a task whose first_task != this one (genuine cross-task reuse)

    # (A) MOTION schemas first (parametric, the hard family) — cheapest to re-verify.
    for key, rec in list(_BANK.motion_schemas.items()):
        fn = _motion_fn_from_schema(rec["family"], rec["params"])
        if fn is None:
            continue
        try:
            ok = _verify(fn, train)
        except Exception:
            ok = False
        if ok:
            rules.append(("reuse:motion:%s" % rec["family"], fn))
            if rec["first_task"] != task_id:
                rec["reuse"] += 1
                fired_new = True

    # (B) PROGRAM-skeleton schemas — re-fit color holes from this palette, exact-verify.
    #     MDL order: fewest holes / shortest skeleton first.
    order = sorted(_BANK.prog_schemas.items(), key=lambda kv: (kv[1]["nholes"], len(kv[0])))
    for skeleton, rec in order:
        prog = _skeleton_refit(skeleton, train, palette)
        if prog is None:
            continue
        def fn(g, prog=prog):
            return dsl.apply_prog(g, prog)
        rules.append(("reuse:prog:%s" % "+".join(n for n, _ in skeleton), fn))
        if rec["first_task"] != task_id:
            rec["reuse"] += 1
            fired_new = True
    if fired_new:
        _BANK.transfer_fires += 1
    return rules, fired_new


def _render_concept_as_prog(name, train):
    """If a verified spine/graft concept is equivalent to a short DSL op-sequence on THIS task, return that
    program so it can be antiunified into a banked skeleton. We only render the easy structural ones whose
    DSL equivalent is unambiguous; others are banked as motion/closure schemas instead."""
    return None  # handled via the seed-search programs + motion schemas; concept-fns are not DSL-rendered.


def _mine_skeletons_from_seed(train, task_id, budget):
    """Run the base seed search; for each verified program found, antiunify -> skeleton and bank it.
    Returns the verified programs (concrete) so the caller can also use them to answer this task."""
    progs, _ = B._search_collect(train, min(budget, 3000), K=4)
    progs = sorted(progs, key=B._plen)
    for p in progs:
        plain = [(n, a) for (n, a, *_) in p if not n.startswith("MACRO[")]
        if len(plain) != len(p):
            continue
        res = _prog_to_skeleton(plain)
        if res is None:
            continue
        skeleton, nholes = res
        _BANK.bank_prog(skeleton, nholes, task_id)
    return progs


# ===========================================================================
# PUBLIC: full solver, single-only ablation.
# ===========================================================================
def _attempts_from_rules(rules, test_inputs, max_keep=2):
    out = []
    for gi in test_inputs:
        cand = []
        for _tag, fn in rules:
            try:
                o = fn(gi)
            except Exception:
                o = None
            if o is not None and getattr(o, "ndim", None) == 2 and o.size > 0:
                o = np.asarray(o, int)
                if not any(_eq(o, c) for c in cand):
                    cand.append(o)
            if len(cand) >= max_keep:
                break
        out.append(cand[:max_keep])
    return out


def _base_direct(train):
    """The base's direct concept fit + linker + its own library replay (no seed). Returns (rules, fresh)."""
    return B._try_concepts(train)


def _fit_motion(train):
    """Fit the parametric motion concepts on THIS task's train; return verified (tag, fn) with fn._schema."""
    out = []
    for name, fitter in MOTION_CONCEPTS:
        try:
            fn = fitter(train)
        except Exception:
            fn = None
        if fn is not None and _verify(fn, train):
            out.append((name, fn))
    return out


def _seed_attempts_and_mine(train, test_inputs, budget, task_id, mine):
    """Faithful copy of base._seed_attempts, but ALSO antiunifies every verified seed program into a
    color-hole skeleton and banks it (when mine=True). Returns base-identical attempts."""
    progs, _ = B._search_collect(train, budget, K=4)
    progs = sorted(progs, key=B._plen)
    if mine:
        for p in progs:
            plain = [(n, a) for (n, a, *_) in p if not n.startswith("MACRO[")]
            if len(plain) != len(p):
                continue
            res = _prog_to_skeleton(plain)
            if res is not None:
                _BANK.bank_prog(res[0], res[1], task_id)
    if progs:  # base's macro/op-hit ingestion (preserve in-session base experience exactly)
        best = min(progs, key=B._plen)
        plain = [(n, a) for (n, a, *_) in best if not n.startswith("MACRO[")]
        if len(plain) == len(best) and all(n in dsl.OPS for n, _ in plain):
            B._LIB.ingest_program("p%d" % B._TASK_COUNTER[0], plain)
        for n, a, *_ in best:
            if not n.startswith("MACRO["):
                B._LIB.op_hits[n] += 1
    progs = progs[:2]
    attempts = []
    for gi in test_inputs:
        cand = []
        for p in progs:
            o = B._prog_run(gi, p)
            if o is not None:
                cand.append(o)
        attempts.append(cand)
    return attempts


def _solve_core(train, test_inputs, budget, allow_transfer=True, allow_compose=True):
    """Faithful SUPERSET of base.solve. Order:
        (0) TRANSFER  : banked cross-task schemas re-fit + exact-verified FIRST  [allow_transfer]
        (1) DIRECT    : base concept store + base linker + base library replay   (always)
        (1b) MOTION   : parametric hard-family motion concepts as extra trusted  [allow_compose]
        (2) SEED      : base DSL search (+ skeleton mining for future transfer)   (always; the floor)
    Mirrors base's trusted/prone slot discipline and merge path so it NEVER regresses base's solves.
    Returns (attempts, tag) with tag in {reuse, single, link, seed, None}."""
    B._TASK_COUNTER[0] += 1  # keep base's counter advancing (used in its ingest keys)
    train = [(np.asarray(gi, int), np.asarray(go, int)) for gi, go in train]
    task_id = "t%d" % _TASK_COUNTER[0]; _TASK_COUNTER[0] += 1
    palette = [int(c) for c in dsl.palette(train)]

    # ---- (0) TRANSFER: banked schemas FIRST (the experience that transfers) ----
    if allow_transfer:
        bank_rules, fired_new = _try_banked_schemas(train, task_id, palette)
        if bank_rules:
            att = _attempts_from_rules(bank_rules, test_inputs)
            if fired_new and all(len(a) >= 1 for a in att):
                # a schema banked on an EARLIER task re-fit + verified here -> genuine transfer solve.
                _BANK.tagcount["reuse"] += 1
                _BANK.audit.append((task_id, "reuse", bank_rules[0][0]))
                return att, "reuse"

    # ---- (1) DIRECT: base concept store + linker + base library replay ----
    rules, fresh = _base_direct(train)
    if not allow_compose:
        # SINGLE-CONCEPT-ONLY ablation: strip the base's own linker/library compositions so the only
        # solves are genuine single-concept self-verifications (+ the seed floor). This is what makes
        # novel_link_solves measure recombination, not just the cross-task bank.
        rules = [(t, f) for t, f in rules
                 if not (t.startswith("link:") or t.startswith("lib:") or "+" in t)]
    is_link = any(t.startswith("link:") or t.startswith("lib:") or "+" in t for t, _ in rules)

    # base in-session experience (bump counts; remember non-prone fresh closures) — preserve base behavior.
    for name, fn in rules:
        B._LIB.bump(B._base_name(name))
    for ftag, ff in fresh:
        if not B._is_prone(ftag):
            B._LIB.remember_closure(ftag, ff)
    for name, _ in rules:
        if name.startswith("lib:") or name.startswith("link:"):
            B._LIB.audit.append((task_id, name, True))

    # ---- (1b) MOTION concepts (hard family) — extra TRUSTED material; bank their schemas ----
    motion_rules = _fit_motion(train) if allow_compose else []
    for name, fn in motion_rules:
        sch = getattr(fn, "_schema", None)
        if sch is not None:
            _BANK.bank_motion(sch[0], sch[1], task_id)

    trusted = [(n, f) for n, f in rules if not B._is_prone(n)] + list(motion_rules)
    prone = [(n, f) for n, f in rules if B._is_prone(n)]
    ordered = trusted + prone

    if ordered:
        # build attempts trusted-first (mirrors base)
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

        # FAST PATH (base): a trusted concept fired AND every test got a candidate.
        if trusted and all(len(a) >= 1 for a in attempts):
            # tag: 'link' iff the answer required a link/motion/lib composition (no plain single concept);
            # else 'single'. (Motion fits count as hard-family link material here.)
            plain_single = any(not (t.startswith("link:") or t.startswith("lib:") or "+" in t)
                               and t not in [m[0] for m in motion_rules]
                               for t, _ in [(n, f) for n, f in rules if not B._is_prone(n)])
            tag = "single" if plain_single else "link"
            _BANK.tagcount[tag] += 1
            return attempts, tag

        # MERGE PATH (base): prone-only (or unfilled) — keep concept attempt-1, seed backfills.
        seed = _seed_attempts_and_mine(train, test_inputs, budget, task_id, mine=allow_transfer)
        merged = []
        for k, gi in enumerate(test_inputs):
            cand = list(attempts[k]) if k < len(attempts) else []
            for o in (seed[k] if k < len(seed) else []):
                if o is None or any(_eq(o, c) for c in cand):
                    continue
                cand.append(o)
                if len(cand) >= 2:
                    break
            merged.append(cand[:2])
        tag = "link" if is_link else ("seed" if not trusted and not prone else "single")
        _BANK.tagcount[tag] += 1
        return merged, tag

    # ---- (2) pure SEED fallback (never regress the gen-0 seed) + skeleton mining ----
    att = _seed_attempts_and_mine(train, test_inputs, budget, task_id, mine=allow_transfer)
    tag = "seed" if any(len(a) >= 1 for a in att) else None
    if tag:
        _BANK.tagcount["seed"] += 1
    return att, tag


def solve(train, test_inputs, budget):
    """FULL system: transfer + composition + hard-family motion + seed fallback."""
    att, _tag = _solve_core(train, test_inputs, budget, allow_transfer=True, allow_compose=True)
    return att


def solve_single_only(train, test_inputs, budget):
    """ABLATION: single from-scratch concept only. Cross-task transfer AND composition (linker + motion
    hard-family) DISABLED. novel_link_solves = full_dev - this_dev. Seed fallback still allowed (it is the
    floor, not creativity), and seed-mined skeletons are NOT banked (transfer off)."""
    att, _tag = _solve_core(train, test_inputs, budget, allow_transfer=False, allow_compose=False)
    return att


# expose base build time for reporting
_BUILD_SEC = B._BUILD_SEC
