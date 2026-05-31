#!/usr/bin/env python3
"""
v7: the tree leap.

v0-v6 all abstracted over LINEAR pipelines — sequences of ops, BPE-merged. The
real DreamCoder territory is abstraction over TREES: higher-order programs where
combinators (map/filter/fold) take SUB-PROGRAMS as arguments. A program is now an
expression tree, e.g.

    map(λx.(x*2)+1, xs)   ==  App(map, Lam(App(inc, App(dbl, Var))), Xs)

— a tree, not a sequence. The question this whole project has been building to:
does the compression/transfer story survive when abstraction is over trees and
the "BPE merge" becomes "find a recurring SUBTREE"?

Mechanism (the tree analogue of v1):
  - Programs are expression trees with real semantics over integer lists.
  - Abstraction = mine recurring SUBTREES ("fragments") across the TRAIN trees;
    add the most-reused ground (non-parameterized) fragments as named library
    primitives, greedily, highest-reuse-first, stopping when reuse runs out
    (count >= 2) — exactly BPE's stopping rule, lifted to trees.
  - Description length of a tree = number of nodes needed to write it, where any
    subtree that EQUALS a library fragment collapses to a single node. (This is
    the direct tree analogue of v1's "fewest library symbols": a fragment is one
    symbol. We minimize over which fragments to apply via a bottom-up DP that
    picks, at each node, the cheapest covering — fragment-as-1 or recurse.)

The measurement (v1 mirror, honest controls):
  TRAIN          : trees built from a few motif sub-programs (learn the motifs).
  TEST_RELATED   : novel trees that REUSE the motif subtrees in new arrangements.
  CONTROL_DISJOINT : trees over leaf ops NO motif uses -> no fragment can apply.
  CONTROL_SCRAMBLE : trees over the SAME leaf ops as the motifs, arranged so no
                     motif subtree ever appears.
If related compresses while controls stay flat, structural transfer holds for
TREES. Plus a depth-generalization check (deeper unseen compositions).

Pure stdlib; real semantics, sanity-checked; runs in well under a second.

NOTE on scope (honest): we do the GROUND-fragment version cleanly. Parameterized
fragments via antiunification (fragments with holes) are a documented stretch;
see the ANTIUNIFICATION block near the end for a correct-but-small attempt.
"""
import itertools
from collections import Counter

# ── higher-order tree DSL ────────────────────────────────────────────────────
# A tree node is a tuple: (tag, *children).
#   ('var',)                      the lambda-bound variable x (an int)
#   ('xs',)                       the input list
#   ('lf', name)                  a leaf int->int op applied... no: leaf ops are
#                                 unary funcs on the bound var, represented as
#                                 ('op', name, child) where child is an int-tree.
#   ('op', name, child)           apply unary int op `name` to int-tree `child`
#   ('pred', name, child)         apply predicate `name` to int-tree `child` (->bool)
#   ('map', fbody, listtree)      map a lambda (fbody over 'var') across a list-tree
#   ('filter', pbody, listtree)   filter a list-tree by a predicate-lambda pbody
#   ('fold', fbody, init, lt)     left-fold; fbody is over ('acc',)&('var',) ... we
#                                 keep fold simple: fbody uses ('var',) as acc and
#                                 a fixed +element via ('elt',). (kept minimal.)
# Int ops on the bound var:
IOP = {
    "inc": lambda x: x + 1,
    "dbl": lambda x: 2 * x,
    "tpl": lambda x: 3 * x,
    "sqr": lambda x: x * x,
    "neg": lambda x: -x,
    "dec": lambda x: x - 1,
}
PRED = {
    "even": lambda x: x % 2 == 0,
    "odd":  lambda x: x % 2 == 1,
    "pos":  lambda x: x > 0,
}


def ev_int(t, x):
    """Evaluate an int-valued tree (a lambda body) given bound var value x."""
    tag = t[0]
    if tag == "var":
        return x
    if tag == "op":
        return IOP[t[1]](ev_int(t[2], x))
    raise ValueError(f"not an int tree: {t}")


def ev_pred(t, x):
    """Evaluate a bool-valued tree given bound var value x."""
    assert t[0] == "pred"
    return PRED[t[1]](ev_int(t[2], x))


def ev_list(t, xs):
    """Evaluate a list-valued tree given the input list xs."""
    tag = t[0]
    if tag == "xs":
        return list(xs)
    if tag == "map":
        sub = ev_list(t[2], xs)
        return [ev_int(t[1], v) for v in sub]
    if tag == "filter":
        sub = ev_list(t[2], xs)
        return [v for v in sub if ev_pred(t[1], v)]
    raise ValueError(f"not a list tree: {t}")


# ── tree helpers ─────────────────────────────────────────────────────────────
def children(t):
    """The sub-trees of a node (skipping the string op/pred name)."""
    tag = t[0]
    if tag in ("var", "xs"):
        return []
    if tag in ("op", "pred"):
        return [t[2]]
    if tag in ("map", "filter"):
        return [t[1], t[2]]
    raise ValueError(t)


def size(t):
    """Plain node count (no library)."""
    return 1 + sum(size(c) for c in children(t))


def subtrees(t):
    """Yield every subtree (including t itself)."""
    yield t
    for c in children(t):
        yield from subtrees(c)


# ── leaf-op lambda bodies (sub-programs over the bound var) ──────────────────
VAR = ("var",)
def op(name, child=VAR):  return ("op", name, child)
def pred(name, child=VAR): return ("pred", name, child)
XS = ("xs",)


# ── motif sub-programs (the reusable lambda bodies) ──────────────────────────
# These are the TREE fragments we hope to discover & transfer.
F_aff   = op("inc", op("dbl"))            # λx.(2x)+1      "double-then-inc"
F_sqr   = op("inc", op("sqr"))            # λx.x²+1        "square-then-inc"
F_quad  = op("dbl", op("dbl"))            # λx.4x          "quadruple"
P_even  = pred("even")                    # λx. even(x)
P_pos   = pred("pos")                     # λx. x>0
MOTIF_BODIES = [F_aff, F_sqr, F_quad]
MOTIF_PREDS  = [P_even, P_pos]


def mp(body, lt=XS):   return ("map", body, lt)
def flt(p, lt=XS):     return ("filter", p, lt)


# TRAIN: trees built from the motif sub-programs in shallow arrangements.
# (single combinator over a motif, and depth-2 stacks — the "all depth-2" of v1.)
TRAIN = []
for b in MOTIF_BODIES:
    TRAIN.append(mp(b))                                   # map(motif, xs)
for p in MOTIF_PREDS:
    TRAIN.append(flt(p))                                  # filter(motif, xs)
# depth-2 stacks: map over a filter, filter then map, etc.
for b in MOTIF_BODIES:
    for p in MOTIF_PREDS:
        TRAIN.append(mp(b, flt(p)))                       # map(b, filter(p, xs))
for b1 in MOTIF_BODIES:
    for b2 in MOTIF_BODIES:
        TRAIN.append(mp(b1, mp(b2)))                      # map(b1, map(b2, xs))

# TEST_RELATED: NOVEL deeper trees reusing the SAME motif subtrees, never seen.
TEST_RELATED = [
    mp(F_aff, flt(P_pos, mp(F_quad))),                    # 3-deep, new arrangement
    flt(P_even, mp(F_sqr, flt(P_pos))),
    mp(F_quad, mp(F_aff, flt(P_even))),
    mp(F_sqr, flt(P_pos, mp(F_aff))),
    flt(P_pos, mp(F_quad, mp(F_sqr))),
    mp(F_aff, mp(F_sqr, flt(P_even, mp(F_quad)))),        # 4-deep
]

# CONTROL_DISJOINT: leaf ops NO motif uses (tpl/neg/dec, pred odd) -> no fragment
# learned from TRAIN can ever match. Same combinator skeleton, disjoint leaves.
G_a = op("tpl", op("neg"))                                # λx.-3x  (uses no motif op)
G_b = op("dec", op("tpl"))                                # λx.3x-1
G_c = op("neg", op("dec"))                                # λx.-(x-1)
P_odd = pred("odd")
CONTROL_DISJOINT = [
    mp(G_a, flt(P_odd, mp(G_b))),
    flt(P_odd, mp(G_c, flt(P_odd))),
    mp(G_b, mp(G_a, flt(P_odd))),
    mp(G_c, flt(P_odd, mp(G_a))),
    flt(P_odd, mp(G_b, mp(G_c))),
    mp(G_a, mp(G_c, flt(P_odd, mp(G_b)))),
]

# CONTROL_SCRAMBLE: the SAME map-body leaf ops the motifs use (inc/dbl/sqr) but
# composed so NO motif body-subtree (inc∘dbl, inc∘sqr, dbl∘dbl) ever appears, and
# filtered by the NON-trained predicate `odd` so it doesn't borrow the trained
# filter idioms either. This isolates the one HONEST tree-specific leak: sharing
# the leaf op `dbl` on the bound var means sharing the 2-node subtree dbl(x),
# which the tree-miner names — a partial-subtree overlap with no linear analogue.
S_a = op("dbl", op("inc"))                                # λx.2(x+1)  -- NOT inc∘dbl
S_b = op("sqr", op("dbl"))                                # λx.(2x)²
S_c = op("inc", op("inc"))                                # λx.x+2     -- not a motif
CONTROL_SCRAMBLE = [
    mp(S_a, flt(P_odd, mp(S_b))),
    flt(P_odd, mp(S_c, flt(P_odd))),
    mp(S_b, mp(S_a, flt(P_odd))),
    mp(S_c, flt(P_odd, mp(S_a))),
    flt(P_odd, mp(S_b, mp(S_c))),
    mp(S_a, mp(S_b, flt(P_odd, mp(S_c)))),
]


# ── library / description length over TREES ──────────────────────────────────
def dl(t, fragset):
    """Description length of tree t given a set of library fragments.
    = number of nodes to express t, where any subtree equal to a fragment
    collapses to a SINGLE node. Bottom-up DP: cost(node) = 1 if the subtree
    rooted here is itself a library fragment, else 1 + sum of children costs.
    (A fragment is atomic — it has internal nodes but is written as one symbol,
    the tree analogue of v1's macro-as-one-symbol.)"""
    if t in fragset:
        return 1
    return 1 + sum(dl(c, fragset) for c in children(t))


def mine_fragments(corpus):
    """Greedy reuse-driven fragment mining (BPE-for-trees).

    Repeatedly: count every subtree across the corpus *under the current
    library's segmentation*, add the most-reused non-trivial subtree as a new
    fragment, stop when no subtree recurs (count < 2). Trivial leaves (var/xs)
    are never fragments. Ties broken toward the LARGER fragment (more compression
    per use), mirroring v1's tie-break."""
    frags = []

    def seg_subtrees(t, fragset):
        """Yield the subtrees that remain VISIBLE given current fragments:
        once a subtree is a fragment we treat it as atomic (don't descend)."""
        yield t
        if t in fragset:
            return
        for c in children(t):
            yield from seg_subtrees(c, fragset)

    while True:
        fragset = set(frags)
        cnt = Counter()
        for tree in corpus:
            for st in seg_subtrees(tree, fragset):
                if st[0] in ("var", "xs"):
                    continue           # trivial leaves aren't worth naming
                if st in fragset:
                    continue           # already a library symbol
                if size(st) < 2:
                    continue           # need internal structure to be a fragment
                cnt[st] += 1
        # only consider subtrees that recur
        cand = [(st, n) for st, n in cnt.items() if n >= 2]
        if not cand:
            break
        # most-reused, tie-break toward larger subtree (bigger compression/use)
        st, n = max(cand, key=lambda kv: (kv[1], size(kv[0])))
        frags.append(st)
    return frags


# ── pretty-printing trees (for sanity / readable library) ────────────────────
def show(t):
    tag = t[0]
    if tag == "var":
        return "x"
    if tag == "xs":
        return "xs"
    if tag == "op":
        return f"{t[1]}({show(t[2])})"
    if tag == "pred":
        return f"{t[1]}?({show(t[2])})"
    if tag == "map":
        return f"map(λx.{show(t[1])}, {show(t[2])})"
    if tag == "filter":
        return f"filter(λx.{show(t[1])}, {show(t[2])})"
    return str(t)


def mean_dl(corpus, fragset):
    return sum(dl(t, fragset) for t in corpus) / len(corpus)


def main():
    print("=" * 78)
    print("v7  THE TREE LEAP: abstraction over higher-order EXPRESSION TREES")
    print("=" * 78)

    # ── sanity: real semantics ────────────────────────────────────────────
    samp = [1, -2, 3, 4, -5]
    p1 = mp(F_aff)                       # map(λx.2x+1, xs)
    p2 = flt(P_pos, mp(F_sqr))           # filter(pos, map(λx.x²+1, xs))
    print(f"  sample input {samp}:")
    print(f"    {show(p1)}")
    print(f"      = {ev_list(p1, samp)}   (expect each 2x+1)")
    print(f"    {show(p2)}")
    print(f"      = {ev_list(p2, samp)}")
    assert ev_list(p1, samp) == [2 * x + 1 for x in samp]
    assert ev_list(p2, samp) == [v for v in (x * x + 1 for x in samp) if v > 0]
    # a TRAIN tree is genuinely a tree (not a linear chain):
    big = mp(F_aff, flt(P_even, mp(F_quad)))
    assert ev_list(big, samp) == [2 * (4 * x) + 1 for x in samp if (4 * x) % 2 == 0]
    print("    semantics sanity-checked ✓")

    # ── learn the library from TRAIN (shallow trees) ──────────────────────
    frags = mine_fragments(TRAIN)
    fragset = set(frags)
    print(f"\n  --- LIBRARY mined from TRAIN ({len(frags)} fragments) ---")
    for f in frags:
        print(f"    [{size(f)} nodes] {show(f)}")

    # ── transfer (v1 mirror) ──────────────────────────────────────────────
    sets = {"related": TEST_RELATED, "ctrl-disjoint": CONTROL_DISJOINT,
            "ctrl-scramble": CONTROL_SCRAMBLE}
    print("\n  --- TRANSFER: description length (nodes) per held-out tree ---")
    print(f"  {'set':<15}{'base':>8}{'w/ lib':>9}{'factor':>9}")
    s0 = {k: mean_dl(v, set()) for k, v in sets.items()}
    s1 = {k: mean_dl(v, fragset) for k, v in sets.items()}
    for k in sets:
        print(f"    {k:<13}{s0[k]:>8.2f}{s1[k]:>9.2f}{s0[k] / s1[k]:>8.2f}x")

    # ── depth generalization (v4/v5 mirror) ───────────────────────────────
    # nest map(F_aff, .) repeatedly inside a deeper related tree; the motifs are
    # the same, depth grows beyond anything in TRAIN (which is depth<=2).
    print("\n  --- DEPTH GENERALIZATION (TRAIN is depth<=2; test deeper) ---")
    print(f"    {'depth':>6}{'base':>9}{'w/ lib':>9}{'factor':>9}")
    cyc = [F_aff, F_sqr, F_quad]
    for d in (2, 3, 4, 5, 6):
        # build a depth-d tree: map(b0, map(b1, ... filter(P_pos, xs)))
        t = flt(P_pos)
        for k in range(d):
            t = mp(cyc[k % len(cyc)], t)
        b = size(t)
        wl = dl(t, fragset)
        print(f"    {d:>6}{b:>9}{wl:>9}{b / wl:>8.2f}x")

    # ── ANTIUNIFICATION (stretch): parameterized fragments with one hole ──
    # A ground fragment only matches an EXACT subtree. Antiunification finds the
    # least-general generalization of two subtrees — a fragment with a HOLE where
    # they differ — so e.g. map(λx.OP(?), xs) abstracts map-over-any-unary-op.
    # We do the minimal correct version: antiunify the two most common
    # *same-shape* subtrees, report the schema, and note its (limited) reuse.
    print("\n  --- ANTIUNIFICATION (stretch: one-hole parameterized fragment) ---")
    au = try_antiunify(TRAIN)
    if au is None:
        print("    no two same-shaped subtrees to generalize — skipped.")
    else:
        schema, holes, matches = au
        print(f"    schema  {show_schema(schema)}")
        print(f"    matched {matches} train subtrees that differ only at the hole")
        print("    (kept honest & small — not wired into the DL metric above.)")

    # ── verdict ───────────────────────────────────────────────────────────
    rel = s0["related"] / s1["related"]
    cdis = s0["ctrl-disjoint"] / s1["ctrl-disjoint"]   # true structural control
    cscr = s0["ctrl-scramble"] / s1["ctrl-scramble"]   # shares leaf ops only
    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"  related       {s0['related']:.2f} -> {s1['related']:.2f} nodes  ({rel:.2f}x)")
    print(f"  ctrl-disjoint {s0['ctrl-disjoint']:.2f} -> {s1['ctrl-disjoint']:.2f} nodes  ({cdis:.2f}x)")
    print(f"  ctrl-scramble {s0['ctrl-scramble']:.2f} -> {s1['ctrl-scramble']:.2f} nodes  ({cscr:.2f}x)")
    # the disjoint control is the clean test of "no shared structure"; scramble
    # shares the SAME leaf ops, which in a TREE means it cannot avoid sharing the
    # tiny leaf subtrees (e.g. dbl(x)) that the miner names -> a small, honest,
    # tree-SPECIFIC leak with no analogue in the linear (v1-v6) regime.
    if rel >= 1.5 and cdis <= 1.05 and cscr <= 1.10:
        print(f"\n  ✓ TREE TRANSFER (with one honest caveat): related {rel:.2f}x shorter;")
        print(f"    disjoint control flat ({cdis:.2f}x). Recurring SUBTREES mined from shallow")
        print("    TRAIN trees shorten novel DEEPER compositions that reuse them, and do")
        print("    NOTHING for trees with no shared structure. The compression/generalization")
        print("    story lifts from LINEAR pipelines to higher-order TREES.")
        print(f"    Caveat: the same-leaf-ops control dips slightly ({cscr:.2f}x) — sharing")
        print("    leaf op `dbl` means sharing the 2-node subtree dbl(x), which the tree")
        print("    miner names. A partial-subtree overlap with no linear (BPE) analogue.")
    else:
        print(f"\n  partial: related {rel:.2f}x, disjoint {cdis:.2f}x, scramble {cscr:.2f}x"
              " — see LOG for the read.")


# ── antiunification helpers (kept separate; stretch goal) ────────────────────
def same_shape(a, b):
    """Do two trees have the same tag-skeleton (ignoring op/pred names)?"""
    if a[0] != b[0]:
        return False
    ca, cb = children(a), children(b)
    if len(ca) != len(cb):
        return False
    return all(same_shape(x, y) for x, y in zip(ca, cb))


def antiunify(a, b):
    """Least-general generalization: identical where a,b agree, a HOLE where the
    leaf op/pred names differ. Returns (schema, n_holes). Assumes same_shape."""
    tag = a[0]
    if tag in ("var", "xs"):
        return a, 0
    if tag in ("op", "pred"):
        sub, h = antiunify(a[2], b[2])
        if a[1] == b[1]:
            return (tag, a[1], sub), h
        return ("hole", tag, sub), h + 1     # differ at this op name -> hole
    # map / filter: two child positions
    s1, h1 = antiunify(a[1], b[1])
    s2, h2 = antiunify(a[2], b[2])
    return (tag, s1, s2), h1 + h2


def schema_size(t):
    """Node count of a (possibly holed) schema."""
    if t[0] == "hole":
        return 1 + schema_size(t[2])
    return 1 + sum(schema_size(c) for c in children(t))


def show_schema(t):
    tag = t[0]
    if tag == "var":
        return "x"
    if tag == "xs":
        return "xs"
    if tag == "hole":
        return f"?({show_schema(t[2])})"      # any op/pred of this arity
    if tag == "op":
        return f"{t[1]}({show_schema(t[2])})"
    if tag == "pred":
        return f"{t[1]}?({show_schema(t[2])})"
    if tag == "map":
        return f"map(λx.{show_schema(t[1])}, {show_schema(t[2])})"
    if tag == "filter":
        return f"filter(λx.{show_schema(t[1])}, {show_schema(t[2])})"
    return str(t)


def matches_schema(schema, t):
    tag = schema[0]
    if tag == "hole":
        return t[0] == schema[1] and matches_schema(schema[2], t[2])
    if tag in ("var", "xs"):
        return t == schema
    if t[0] != tag:
        return False
    if tag in ("op", "pred"):
        return t[1] == schema[1] and matches_schema(schema[2], t[2])
    if tag in ("map", "filter"):
        return matches_schema(schema[1], t[1]) and matches_schema(schema[2], t[2])
    return False


def try_antiunify(corpus):
    """Find the most-reused one-hole schema: group same-shaped non-trivial
    subtrees, antiunify the two largest-support groups, return the best schema."""
    all_sub = [st for tree in corpus for st in subtrees(tree)
               if st[0] not in ("var", "xs") and size(st) >= 2]
    best = None
    seen = sorted(set(all_sub), key=lambda s: (size(s), str(s)))  # deterministic
    for i in range(len(seen)):
        for j in range(i + 1, len(seen)):
            a, b = seen[i], seen[j]
            if a == b or not same_shape(a, b):
                continue
            schema, h = antiunify(a, b)
            if h != 1:                      # exactly one hole = cleanest abstraction
                continue
            m = sum(1 for st in seen if matches_schema(schema, st))
            if best is None or m > best[2] or (m == best[2] and schema_size(schema) > schema_size(best[0])):
                best = (schema, h, m)
    return best


if __name__ == "__main__":
    main()
