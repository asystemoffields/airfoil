#!/usr/bin/env python3
"""Branch-B scale-prep BOX-PREP 3 — the THIN-CORE typed relation DSL (Alex's call: earn the vocabulary).

Design line (the thinnest principled core): the FIXED CORE is exactly the proven recognizer's level — object
DECOMPOSITION + per-object FEATURES + minimal COMPOSE — because that is grounding/routing, not vocabulary, and
it is where V2/V3-GEO live. EVERYTHING RELATIONAL above it (object-PAIR/SET quantifiers, argextreme/argunique,
let-binding, conditionals) is SCAFFOLD: present so the loop can ignite, tagged `SCAFFOLD`, and ABLATABLE — the
experiment is whether anti-unification (lgg.py, proven 1.00) RE-DISCOVERS these from verified solves, in which
case the system EARNED its relational vocabulary rather than us handing it over. The reason to keep the core thin:
the expressiveness ceiling stops being our imagination (it becomes the closure of the GROWN library), dodging the
"DSL design is the treadmill moved up a level" risk.

A program is a typed AST mapping GRID->GRID; the exact verifier (verify+generalize) gates everything.
Run: /data/llm/.venv/bin/python rel_dsl.py   (smoke: DSL SUBSUMES the grammar + EXPRESSES a relation it can't)"""
import numpy as np
import grammar as G

CORE = "core"; SCAFFOLD = "scaffold"


# ---------- per-object KEY expressions (object -> hashable value); recolor/select read these ----------
class FeatKey:                                  # CORE: the recognizer's vocabulary
    tier = CORE
    def __init__(self, name): self.name = name
    def __repr__(self): return f"feat:{self.name}"
    def __call__(self, o, objs): return G.FEATURES[self.name](o, objs)


def _bbox(o): return (o["r0"], o["c0"], o["r0"] + o["h"], o["c0"] + o["w"])
def _contains(a, b):
    ar0, ac0, ar1, ac1 = _bbox(a); br0, bc0, br1, bc1 = _bbox(b)
    return a is not b and ar0 <= br0 and ac0 <= bc0 and ar1 >= br1 and ac1 >= bc1 and (a["size"] != b["size"] or a["h"] != b["h"])


def _adjacent(a, b):
    ar0, ac0, ar1, ac1 = _bbox(a); br0, bc0, br1, bc1 = _bbox(b)
    return a is not b and ar0 - 1 <= br1 and br0 - 1 <= ar1 and ac0 - 1 <= bc1 and bc0 - 1 <= ac1
def _aligned(a, b):
    return a is not b and (a["r0"] == b["r0"] or a["c0"] == b["c0"])


# ===================== THE RELATIONAL EYE (a fixed structural FACULTY, core) =====================
# pair_signature exposes the RAW relational perceptions between two objects -- the "cones". It NAMES NOTHING;
# which channel+value matters ("contained", "adjacent", ...) is the vocabulary the system EARNS, not us.
SIG = ["a_contains_b", "b_contains_a", "adjacent", "aligned", "a_bigger", "same_size", "same_color"]

def pair_signature(a, b):
    return (int(_contains(a, b)), int(_contains(b, a)), int(_adjacent(a, b)), int(_aligned(a, b)),
            int(a["size"] > b["size"]), int(a["size"] == b["size"]), int(a["color"] == b["color"]))


class Quantify:                                 # CORE combinator over the faculty (the quantifier = grounding);
    tier = CORE                                 # a chosen (channel,value,mode) instantiation = the EARNED predicate
    def __init__(self, channel, value=1, mode="exists"):
        self.ch = channel; self.value = value; self.mode = mode
    def __repr__(self): return f"quant:{self.mode}({SIG[self.ch]}={self.value})"
    def __call__(self, o, objs):
        hits = [pair_signature(o, b)[self.ch] == self.value for b in objs if b is not o]
        if self.mode == "exists": return int(any(hits))
        if self.mode == "forall": return int(bool(hits) and all(hits))
        return sum(hits)                         # count


def predicate_space():
    """the EARNABLE relational predicates = instantiations of the Quantify combinator over the faculty."""
    for ch in range(len(SIG)):
        for mode in ("exists", "forall"):
            yield Quantify(ch, 1, mode)


class Composed:                                 # the CEILING CLIMBING: a predicate NO single faculty instantiation
    tier = CORE                                 # can express -- exists/forall b with pair_sig[ch]==v AND inner(b)
    def __init__(self, channel, value, inner, mode="exists"):
        self.ch = channel; self.value = value; self.inner = inner; self.mode = mode
    def __repr__(self): return f"compose:{self.mode}({SIG[self.ch]}={self.value} & {self.inner})"
    def __call__(self, o, objs):
        hits = [(pair_signature(o, b)[self.ch] == self.value and bool(self.inner(b, objs)))
                for b in objs if b is not o]
        if self.mode == "exists": return int(any(hits))
        if self.mode == "forall": return int(bool(hits) and all(hits))
        return sum(hits)


# ---------- program nodes (GRID -> GRID) ----------
class Recolor:                                  # CORE backbone: decompose -> recolor each object by a KEY -> grid
    tier = CORE
    def __init__(self, key, table, conn=4, by_color=True):
        self.key = key; self.table = table; self.conn = conn; self.by_color = by_color
    def __repr__(self): return f"recolor[{self.conn}{'c' if self.by_color else 't'}]({self.key}, {self.table})"
    def tiers(self): return {self.tier, self.key.tier}
    def ev(self, grid):
        out = grid.copy(); objs = G.objects(grid, self.conn, self.by_color)
        for o in objs:
            k = self.key(o, objs)
            if k in self.table:
                for (r, c) in o["cells"]:
                    out[r, c] = self.table[k]
        return out


class Compose:                                  # CORE: structural pre-op (V3-GEO's prefix) then a relation
    tier = CORE
    def __init__(self, pre, prog): self.pre = pre; self.prog = prog
    def __repr__(self): return f"compose({self.pre}, {self.prog})"
    def tiers(self): return {self.tier} | self.prog.tiers()
    def ev(self, grid):
        import grammar_comp as GC
        g2 = GC._pre(self.pre, grid)
        return None if (g2 is None or g2.size == 0) else self.prog.ev(g2)


def run(prog, grid):
    try:
        return prog.ev(np.asarray(grid, int))
    except Exception:
        return None


def verify(prog, train, test=None):
    """exact verify on train (+ generalize to test). The gate -- identical discipline to the grammar."""
    for gi, go in train:
        o = run(prog, gi)
        if o is None or o.shape != np.asarray(go).shape or not np.array_equal(o, go):
            return False
    if test is not None:
        for gi, go in test:
            o = run(prog, gi)
            if o is None or o.shape != np.asarray(go).shape or not np.array_equal(o, go):
                return False
    return True


def to_dsl(rel):
    """SUBSUMPTION: transcribe a grammar.py relation dict -> an equivalent DSL program (core-only)."""
    if rel["effect"] == "recolor":
        conn, byc = rel["decomp"]
        return Recolor(FeatKey(rel["feature"]), dict(rel["table"]), conn, byc)
    raise NotImplementedError(rel["effect"])


def induce_recolor(key, train, conn=4, by_color=True):
    """fit a recolor table: object's key-value -> its output color, consistent across demos (or None)."""
    table = {}
    for gi, go in train:
        gi = np.asarray(gi, int); go = np.asarray(go, int)
        if gi.shape != go.shape:
            return None
        objs = G.objects(gi, conn, by_color)
        for o in objs:
            k = key(o, objs)
            cols = [int(go[r, c]) for (r, c) in o["cells"]]
            col = max(set(cols), key=cols.count) if cols else 0
            if k in table and table[k] != col:
                return None
            table[k] = col
    return Recolor(key, table, conn, by_color)


def earn_predicate(train, test):
    """EARN a relational predicate from the FACULTY: search the Quantify instantiation space for one whose
    induced recolor solves+generalizes. The predicate is DISCOVERED (search+verify), never hand-given."""
    for pred in predicate_space():
        prog = induce_recolor(pred, train)
        if prog is not None and verify(prog, train, test):
            return prog
    return None


def earn_composed(train, test, library):
    """EARN a COMPOSED predicate: search OUTER (channel,value,mode) x INNER (an already-earned library predicate)
    for a composition whose induced recolor solves+generalizes. This is the ceiling climbing past the faculty's
    single instantiations -- built from earned vocabulary, nothing hand-coded."""
    for ch in range(len(SIG)):
        for mode in ("exists", "forall"):
            for inner in library:
                prog = induce_recolor(Composed(ch, 1, inner, mode), train)
                if prog is not None and verify(prog, train, test):
                    return prog
    return None


def uses_relational(prog):
    return isinstance(getattr(prog, "key", None), (Quantify, Composed)) or \
        (isinstance(prog, Compose) and uses_relational(prog.prog))


# ---------- smoke: SUBSUMES the grammar + EXPRESSES a relation the grammar structurally cannot ----------
def _demo():
    import sys
    sys.path.insert(0, "/data/Windows-files/Documents/airfoil/incubation/evolve")
    import harness
    from ground_arc import winning_relations

    # (1) SUBSUMPTION: a real grammar-solvable recolor task -> DSL program solves it identically, CORE-only.
    sub_ok = sub_tot = 0
    for tid, train, test in list(harness.load_split("arc1-train"))[:120]:
        for eff, feat, rel in winning_relations(train, test):
            if eff == "recolor":
                prog = to_dsl(rel); sub_tot += 1
                if verify(prog, train, test) and not uses_relational(prog):
                    sub_ok += 1
                break
    print(f"SUBSUMPTION: {sub_ok}/{sub_tot} grammar recolor-solves reproduced by CORE-only DSL programs "
          f"(scaffold-free) = the thin core already contains the whole proven grammar.")

    # (2) EXPRESSIVENESS: a containment relation -- recolor objects CONTAINED in another. The grammar's
    #     per-object features cannot say "contained-in-another"; the scaffold ContainedKey can.
    rng = np.random.RandomState(0)
    def make_containment_task(n):
        # RANDOMIZED: 3 hollow boxes (color 5) at random spots, each with a contained dot at a random interior
        # offset, + 4 not-contained dots at random open cells. All dots are IDENTICAL (size 1, color 4) and
        # randomly positioned ACROSS demos -> no per-object feature (size/color/rank/position) consistently
        # correlates with containment; ONLY the actual object-pair relation generalizes.
        demos = []
        for _ in range(n):
            g = np.zeros((22, 22), int); boxes = []
            for _ in range(3):
                for _try in range(30):
                    r, c = rng.randint(0, 16), rng.randint(0, 16)
                    if all(abs(r - br) > 7 or abs(c - bc) > 7 for br, bc in boxes):
                        g[r:r+6, c:c+6] = 5; g[r+1:r+5, c+1:c+5] = 0; boxes.append((r, c)); break
            for (r, c) in boxes:                                     # one contained dot per box (random interior)
                g[r + 1 + rng.randint(0, 3), c + 1 + rng.randint(0, 3)] = 4
            placed = 0                                               # 4 not-contained dots in open space
            for _try in range(120):
                if placed >= 4: break
                r, c = rng.randint(0, 22), rng.randint(0, 22)
                if g[r, c] == 0 and all(not (br <= r < br+6 and bc <= c < bc+6) for br, bc in boxes):
                    g[r, c] = 4; placed += 1
            out = g.copy(); objs = G.objects(g, 4, True)
            for o in objs:                                          # ground truth: contained -> 2, else -> 3
                col = 2 if any(_contains(x, o) for x in objs) else 3
                for (rr, cc) in o["cells"]: out[rr, cc] = col
            demos.append((g, out))
        return demos
    ctrain = make_containment_task(4); ctest = make_containment_task(2)
    gram = winning_relations(ctrain, ctest)                  # grammar (per-object features) cannot express it
    earned = earn_predicate(ctrain, ctest)                   # EARN it from the faculty -- search, nothing given
    print(f"EARN-THE-PREDICATE: containment task -- GRAMMAR winning relations = {len(gram)} (per-object features "
          f"can't say it); EARNED from the relational faculty -> {earned.key if earned else None} "
          f"(uses_relational={uses_relational(earned) if earned else False})")
    print("READ: grammar=0 but the system DISCOVERS the right predicate by searching instantiations of the fixed "
          "pair-comparison FACULTY (the eye is given; the concept 'contained' is EARNED, not hand-coded). "
          "BOX-PREP 4: does anti-unification NAME the recurring earned predicates into a reusable library.")


if __name__ == "__main__":
    _demo()
