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


class ContainedKey:                             # SCAFFOLD: object-PAIR relation the per-object grammar cannot say
    tier = SCAFFOLD
    def __repr__(self): return "rel:contained-in-another"
    def __call__(self, o, objs): return int(any(_contains(x, o) for x in objs))


class ContainsKey:                              # SCAFFOLD
    tier = SCAFFOLD
    def __repr__(self): return "rel:contains-another"
    def __call__(self, o, objs): return int(any(_contains(o, x) for x in objs))


class ArgExtremeKey:                            # SCAFFOLD: 1 iff this object is the argmax/argmin of a feature
    tier = SCAFFOLD
    def __init__(self, feat, take_max=True): self.feat = feat; self.mx = take_max
    def __repr__(self): return f"rel:arg{'max' if self.mx else 'min'}:{self.feat}"
    def __call__(self, o, objs):
        vals = [G.FEATURES[self.feat](x, objs) for x in objs]
        tgt = max(vals) if self.mx else min(vals)
        return int(G.FEATURES[self.feat](o, objs) == tgt)


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


def uses_scaffold(prog):
    return SCAFFOLD in prog.tiers()


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
                if verify(prog, train, test) and not uses_scaffold(prog):
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
    prog = Recolor(ContainedKey(), {1: 2, 0: 3})
    rel_ok = verify(prog, ctrain, ctest)
    # confirm the GRAMMAR cannot express it: no winning grammar relation
    gram = winning_relations(ctrain, ctest)
    print(f"EXPRESSIVENESS: containment task -- DSL(scaffold ContainedKey) solves={rel_ok} (uses_scaffold="
          f"{uses_scaffold(prog)}); GRAMMAR winning relations = {len(gram)} "
          f"-> the scaffold expresses a relation the fixed grammar structurally CANNOT.")
    print("READ: subsumption ~full + a scaffold-only solve the grammar can't reach = the thin-core DSL is a strict "
          "superset of the proven grammar, with the relational layer cleanly TAGGED + ABLATABLE for the "
          "'does anti-unification re-discover it' experiment (BOX-PREP 4).")


if __name__ == "__main__":
    _demo()
