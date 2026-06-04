#!/usr/bin/env python3
"""Gen-1 candidate (mutation operator #2: IN-SESSION EXPERIENCE LIBRARY).

THESIS SERVED: creativity = a CONCEPT STORE filled by EXPERIENCE + a LINKER that recombines concepts
in novel ways, filtered by exact verify. This solver adds a *module-level experience library* that grows
as solve() is called across a run: it EXTRACTS reusable abstractions from train-verified solutions
(named concept-fires + recurring DSL op-subsequences via antiunification), CONSOLIDATES them under an MDL
razor (keep a macro only if it compresses across >=2 solutions or a concept fires >=1 time), and REUSES
those abstractions as HIGH-PRIORITY concepts on LATER tasks. Later tasks benefit from earlier ones.

Two experience channels are present:
  * BAKED-IN: a small bank of *concept families* (learned per task from the current train pairs, then
    exact-verified) — these are the priors. They were chosen by studying REAL bucket-A failures.
  * IN-SESSION: the growing library (this file's module-level state) of concept-fire records + DSL macros
    mined from prior verified solutions, replayed first on later tasks.

The LINKER recombines: concept families are tried in MDL order; library macros are spliced into the DSL
search as length-1 "super-ops" (functional repurposing of a subsequence proven elsewhere). Every candidate
is exact-verified on the CURRENT task's train pairs before being trusted — pure self-supervision.

INTEGRITY: solve() reads ONLY (a) the current task's train pairs it is handed, (b) module-level state
accumulated from PRIOR solve() calls in THIS run (verified-correct solutions only), (c) nothing from disk,
no test outputs, no network, no LLM. Build-time setup uses only tiny self-generated synthetic checks.

Run/imported with /data/llm/.venv/bin/python."""
import sys, heapq
from collections import deque, Counter
import numpy as np

ARC = "/data/Windows-files/Documents/airfoil/incubation/arc"
if ARC not in sys.path:
    sys.path.insert(0, ARC)
import dsl

META = {"name": "exp_library_v1",
        "desc": "in-session experience library: concept families + antiunified DSL macros mined from "
                "verified solutions, replayed high-priority on later tasks; exact-verified linker"}

# =====================================================================================================
# MODULE-LEVEL IN-SESSION EXPERIENCE LIBRARY  (persists across solve() calls within one run)
# =====================================================================================================
class Library:
    def __init__(self):
        # concept-fire tally: concept_name -> how many tasks it has solved this run
        self.concept_hits = Counter()
        # ordered list of concept names by how useful they've been (most-useful first) — the REUSE order
        self.concept_order = []
        # mined DSL macros: name -> (op-subsequence tuple). Each is a length-1 "super-op" for later search.
        self.macros = {}
        # provenance: macro_name -> set of task_ids whose verified program contained the subsequence
        self.macro_src = {}
        # raw store of verified DSL programs (list of (op,args)) for antiunification across solutions
        self.solved_progs = []
        # reuse audit: list of (task_id, what_fired, was_it_a_library_item)
        self.audit = []

    def note_concept(self, name):
        self.concept_hits[name] += 1
        # keep concept_order sorted by hit count desc (stable-ish) — this is the priority the LINKER uses
        self.concept_order = [c for c, _ in self.concept_hits.most_common()]

    def ingest_program(self, task_id, prog):
        """A train-verified DSL PROGRAM (list of (op,args)) was found. Mine recurring op-subsequences as
        macros under an MDL razor: a subsequence becomes a named macro only once it recurs across >=2
        distinct solved tasks (compresses future search)."""
        self.solved_progs.append((task_id, prog))
        if len(prog) < 2:
            return
        # antiunification-lite: enumerate contiguous op-NAME subsequences (length 2..len), ignoring color
        # args so the abstraction is a *shape* reusable across palettes (functional repurposing).
        seqs = set()
        names = [op for op, _ in prog]
        for L in range(2, len(names) + 1):
            for s in range(0, len(names) - L + 1):
                seqs.add(tuple(names[s:s + L]))
        for seq in seqs:
            self.macro_src.setdefault(seq, set()).add(task_id)
        # consolidate: promote any subsequence seen in >=2 distinct tasks to a live macro
        for seq, srcs in self.macro_src.items():
            if len(srcs) >= 2:
                mname = "MACRO[" + ">".join(seq) + "]"
                self.macros[mname] = seq
                self.macro_src[mname] = srcs

    def macro_ops(self):
        """Return [(macro_name, fn)] where fn applies the op-name subsequence with NO color args.
        These are used for arg-free ops only; parametric subsequences are skipped (kept simple/sound)."""
        out = []
        for mname, seq in self.macros.items():
            if all(dsl.OPS[op][1] == 0 for op in seq):
                def make(seq):
                    def fn(g):
                        for op in seq:
                            g = dsl.OPS[op][0](g)
                        return g
                    return fn
                out.append((mname, make(seq)))
        return out


LIB = Library()

# =====================================================================================================
# CONCEPT FAMILIES  (the BAKED-IN priors — each learns its parameters from the current train pairs,
# then is exact-verified before use). Each returns a *predictor* g->g or None if it can't fit train.
# These were derived by studying real unsolved (bucket-A) ARC-AGI-1 training tasks.
# =====================================================================================================

def _patch_mats(a, k=1):
    h, w = a.shape
    pad = np.full((h + 2 * k, w + 2 * k), -1, int)
    pad[k:k + h, k:k + w] = a
    return [[pad[i:i + 2 * k + 1, j:j + 2 * k + 1] for j in range(w)] for i in range(h)]


def _canon8(m):
    """Canonical key of a patch over the 8 dihedral symmetries (rot/reflect invariant)."""
    best = None
    r = m
    for _ in range(4):
        for t in (r, r[:, ::-1]):
            key = t.tobytes()
            if best is None or key < best:
                best = key
        r = np.rot90(r)
    return best


def concept_local_rule(train):
    """CONCEPT 'local-rule': output[i,j] = f(3x3 neighborhood of input), as a learned lookup table that is
    rotation/reflection INVARIANT (the invariance is the generalization lever). Unknown patches copy input.
    Captures color-maps, denoising, neighbor-count textures, dotted-cross stamping, etc."""
    if any(a.shape != b.shape for a, b in train):
        return None
    mp = {}
    for a, b in train:
        pm = _patch_mats(a, 1)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                key = _canon8(pm[i][j]); y = int(b[i, j])
                if key in mp and mp[key] != y:
                    return None
                mp[key] = y

    def f(a):
        out = a.copy(); pm = _patch_mats(a, 1)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                key = _canon8(pm[i][j])
                if key in mp:
                    out[i, j] = mp[key]
        return out
    return f


def concept_local_rule_plain(train):
    """CONCEPT 'local-rule-plain': same as above but ORIENTED (no symmetry canonicalization). Catches
    direction-sensitive local rules the invariant version would wrongly merge."""
    if any(a.shape != b.shape for a, b in train):
        return None
    mp = {}
    for a, b in train:
        h, w = a.shape
        pad = np.full((h + 2, w + 2), -1, int); pad[1:1 + h, 1:1 + w] = a
        for i in range(h):
            for j in range(w):
                key = pad[i:i + 3, j:j + 3].tobytes(); y = int(b[i, j])
                if key in mp and mp[key] != y:
                    return None
                mp[key] = y

    def f(a):
        h, w = a.shape; out = a.copy()
        pad = np.full((h + 2, w + 2), -1, int); pad[1:1 + h, 1:1 + w] = a
        for i in range(h):
            for j in range(w):
                key = pad[i:i + 3, j:j + 3].tobytes()
                if key in mp:
                    out[i, j] = mp[key]
        return out
    return f


def concept_fractal(train):
    """CONCEPT 'fractal-self-tile': out is (h*h, w*w); block (i,j) = the input grid where a[i,j] passes a
    learned 0/nonzero condition, else zeros. (e.g. 007bbfb7.)"""
    for a, b in train:
        h, w = a.shape
        if b.shape != (h * h, w * w):
            return None
    for mode in ("nz", "z"):
        def apply(a, mode=mode):
            h, w = a.shape; out = np.zeros((h * h, w * w), int)
            for i in range(h):
                for j in range(w):
                    on = (a[i, j] != 0) if mode == "nz" else (a[i, j] == 0)
                    if on:
                        out[i * h:(i + 1) * h, j * w:(j + 1) * w] = a
            return out
        if all(np.array_equal(apply(a), b) for a, b in train):
            return apply
    return None


def concept_mirror_tile(train):
    """CONCEPT 'mirror-tile': out tiles the input by an integer (rh, rw); each tile is the input optionally
    flipped v/h. Per-tile flips inferred from the first train pair and verified across all."""
    a0, b0 = train[0]
    if a0.shape == b0.shape:
        return None
    if b0.shape[0] % a0.shape[0] or b0.shape[1] % a0.shape[1]:
        return None
    rh, rw = b0.shape[0] // a0.shape[0], b0.shape[1] // a0.shape[1]
    if rh * rw < 2:
        return None
    h0, w0 = a0.shape
    flips = [[None] * rw for _ in range(rh)]
    for ti in range(rh):
        for tj in range(rw):
            blk = b0[ti * h0:(ti + 1) * h0, tj * w0:(tj + 1) * w0]
            found = None
            for fv in (False, True):
                for fh in (False, True):
                    t = a0[::-1, :] if fv else a0
                    t = t[:, ::-1] if fh else t
                    if np.array_equal(t, blk):
                        found = (fv, fh)
            if found is None:
                return None
            flips[ti][tj] = found

    def f(a):
        h, w = a.shape
        if any(np.array_equal(a, None) for _ in ()):
            pass
        rows = []
        for ti in range(rh):
            cols = []
            for tj in range(rw):
                fv, fh = flips[ti][tj]
                t = a[::-1, :] if fv else a
                t = t[:, ::-1] if fh else t
                cols.append(t)
            rows.append(np.concatenate(cols, 1))
        return np.concatenate(rows, 0)
    return f


def concept_sym_repair(train):
    """CONCEPT 'symmetry-repair': a single noise color N occludes a globally symmetric grid; restore each
    N-cell from a symmetric copy (h/v/180/transpose). (e.g. 496994bd-style restoration.)"""
    if any(a.shape != b.shape for a, b in train):
        return None
    Ns = set()
    for a, b in train:
        d = a != b
        Ns |= set(a[d].tolist())
    if len(Ns) != 1:
        return None
    N = Ns.pop()

    def repair(a):
        out = a.copy(); h, w = a.shape
        syms = [a[::-1, :], a[:, ::-1], a[::-1, ::-1]]
        if h == w:
            syms += [a.T, a.T[::-1, :], a.T[:, ::-1]]
        for _ in range(4):
            mask = out == N
            if not mask.any():
                break
            for s in syms:
                m = mask & (s != N)
                out[m] = s[m]
                mask = out == N
        return out
    return repair


def _components(g):
    h, w = g.shape; seen = np.zeros_like(g, bool); cs = []
    for i in range(h):
        for j in range(w):
            if g[i, j] != 0 and not seen[i, j]:
                comp = []; q = deque([(i, j)]); seen[i, j] = True
                while q:
                    x, y = q.popleft(); comp.append((x, y))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        a, b = x + dx, y + dy
                        if 0 <= a < h and 0 <= b < w and g[a, b] != 0 and not seen[a, b]:
                            seen[a, b] = True; q.append((a, b))
                cs.append(comp)
    return cs


def concept_object_recolor_by_size(train):
    """CONCEPT 'recolor-by-size': each 4-connected nonzero object is recolored solid by a learned function
    of its cell-count (size -> color). Background untouched. A 'concept used outside its usual role':
    object size becomes a key into a color map."""
    if any(a.shape != b.shape for a, b in train):
        return None
    mp = {}
    for a, b in train:
        for comp in _components(a):
            outc = set(int(b[x, y]) for x, y in comp)
            if len(outc) != 1:
                return None
            oc = outc.pop(); sz = len(comp)
            if sz in mp and mp[sz] != oc:
                return None
            mp[sz] = oc
        bg = a == 0
        if not np.array_equal(a[bg], b[bg]):
            return None

    def f(a):
        out = a.copy()
        for comp in _components(a):
            sz = len(comp)
            if sz in mp:
                for x, y in comp:
                    out[x, y] = mp[sz]
        return out
    return f


def concept_colormap(train):
    """CONCEPT 'global-colormap': a single consistent per-color substitution (degenerate local rule, but
    cheapest in MDL terms so tried first)."""
    if any(a.shape != b.shape for a, b in train):
        return None
    mp = {}
    for a, b in train:
        for x, y in zip(a.ravel(), b.ravel()):
            x, y = int(x), int(y)
            if x in mp and mp[x] != y:
                return None
            mp[x] = y

    def f(a):
        out = a.copy()
        for x, y in mp.items():
            out[a == x] = y
        return out
    return f


# Concept registry in DEFAULT MDL order (cheaper / simpler concepts first). The LINKER reorders by
# in-session experience: concepts that have fired before are promoted to the front.
CONCEPTS = [
    ("global-colormap", concept_colormap),
    ("recolor-by-size", concept_object_recolor_by_size),
    ("symmetry-repair", concept_sym_repair),
    ("fractal-self-tile", concept_fractal),
    ("mirror-tile", concept_mirror_tile),
    # the symmetry-INVARIANT local rule precedes the oriented one: invariance = fewer effective params =
    # simpler by MDL AND generalizes better (oriented version often fits train but misses unseen patches).
    ("local-rule", concept_local_rule),
    ("local-rule-plain", concept_local_rule_plain),
]


def _verify(f, train):
    """Exact-verify a predictor on ALL train pairs (shape + values)."""
    try:
        for a, b in train:
            o = f(a)
            if o is None or o.shape != b.shape or not np.array_equal(o, b):
                return False
        return True
    except Exception:
        return False


# =====================================================================================================
# DSL SEARCH (seed best-first) + library-macro splicing
# =====================================================================================================
def _gdist(a, b):
    if a is None:
        return 3.0
    if a.shape != b.shape:
        return 1.0 + abs(a.size - b.size) / max(a.size, b.size, 1)
    return float((a != b).mean())


def _instantiate(pal, macro_ops):
    colors = [c for c in pal if c != 0]
    insts = []
    # library MACROS first (length-1 super-ops) — high-priority reuse of proven subsequences
    for mname, fn in macro_ops:
        insts.append((mname, (), fn))
    for name, (_fn, nc) in dsl.OPS.items():
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


def search_collect(train, B, macro_ops, W=25, max_len=3, K=4):
    insts = _instantiate(dsl.palette(train), macro_ops)
    ins = [gi for gi, _ in train]; tgt = [go for _, go in train]
    start = sum(_gdist(a, b) for a, b in zip(ins, tgt)) / len(ins)
    heap = [(start, 0, [], ins)]; ctr = 1; nexec = 0; found = []
    while heap and nexec < B:
        _s, _c, prog, outs = heapq.heappop(heap)
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
    n = 0
    for name, a, *_ in p:
        n += 1 + len(a)
        if name.startswith("MACRO["):
            n -= 0  # macros count as 1 op (reuse is cheap by MDL) — keeps them competitive
    return n


def _prog_run(gi, prog):
    g = gi
    try:
        for inst in prog:
            g = _apply_inst(g, inst)
        return g
    except Exception:
        return None


# =====================================================================================================
# solve()
# =====================================================================================================
_TASK_COUNTER = [0]


def solve(train, test_inputs, budget):
    tid = "t%d" % _TASK_COUNTER[0]; _TASK_COUNTER[0] += 1
    candidates = []   # list of predictor functions g->g that PASSED exact train-verify, MDL order
    fired_label = None
    fired_is_library = False

    # ---- (1) CONCEPT FAMILIES, reordered by IN-SESSION EXPERIENCE (REUSE) ----------------------------
    # The linker tries concepts that have fired on PRIOR tasks first (the experience library's priority),
    # then the rest in default MDL order. This is the core "later tasks benefit from earlier" mechanism.
    order = list(CONCEPTS)
    rank = {n: i for i, n in enumerate(LIB.concept_order)} if LIB.concept_order else {}
    if rank:
        order.sort(key=lambda c: rank.get(c[0], len(rank)))
    for cname, cfn in order:
        try:
            f = cfn(train)
        except Exception:
            f = None
        if f is not None and _verify(f, train):
            candidates.append((cname, f))
            if fired_label is None:
                fired_label = cname
                fired_is_library = cname in rank
            if len(candidates) >= 2:  # 2-attempt budget: keep the two best MDL-ordered verified concepts
                break

    # ---- (2) DSL SEARCH with library MACROS spliced in (super-ops) -----------------------------------
    dsl_progs = []
    if not candidates:
        macro_ops = LIB.macro_ops()
        progs, _ = search_collect(train, budget, macro_ops, K=4)
        if progs:
            progs = sorted(progs, key=_plen)[:2]
            dsl_progs = progs
            used_macro = any(any(n.startswith("MACRO[") for n, *_ in p) for p in progs)
            if fired_label is None:
                fired_label = "dsl-search" + ("+macro" if used_macro else "")
                fired_is_library = used_macro

    # ---- (3) INGEST into the experience library (verified-correct solutions only) --------------------
    if candidates:
        LIB.note_concept(fired_label)
        LIB.audit.append((tid, fired_label, fired_is_library))
    elif dsl_progs:
        # store the shortest verified DSL program for antiunification / macro mining
        best = min(dsl_progs, key=_plen)
        plain = [(n, a) for (n, a, *_) in best if not n.startswith("MACRO[")]
        if all(n in dsl.OPS for n, _ in plain) and len(plain) == len(best):
            LIB.ingest_program(tid, plain)
        LIB.note_concept(fired_label)
        LIB.audit.append((tid, fired_label, fired_is_library))

    # ---- (4) PRODUCE ATTEMPTS (up to 2 per test input, best-first MDL order) --------------------------
    attempts = []
    for gi in test_inputs:
        cand = []
        for _cname, f in candidates[:2]:
            try:
                o = f(gi)
                if o is not None:
                    cand.append(o)
            except Exception:
                pass
        for p in dsl_progs[:2]:
            o = _prog_run(gi, p)
            if o is not None:
                cand.append(o)
        attempts.append(cand[:2])
    return attempts


# Self-test entry point for quick sanity (not run on import).
if __name__ == "__main__":
    a = np.array([[3, 1, 2], [3, 1, 2], [3, 1, 2]])
    b = np.array([[4, 5, 6], [4, 5, 6], [4, 5, 6]])
    f = concept_colormap([(a, b)])
    print("colormap self-test:", np.array_equal(f(a), b))
