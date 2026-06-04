#!/usr/bin/env python3
"""Vine RL loop — the REWARD (brick 1). R = SOLVE x GEN x NOVELTY, multiplicative so no factor can be traded:
the creativity bar IS the loss weight (verifier-as-reward expert iteration, NOT temporal RL).

  SOLVE   in {0,1} = exact verify on TRAIN demos.
  GEN     in {0,1} = exact verify on HELD-OUT test pair(s) -- a spurious train-only fit scores 0 -> ZERO gradient
                     (kills the documented spurious-consistency tail).
  NOVELTY = 1.0 iff the INVENTION GATE fires: the hand-authored grammar genuinely can't express the task
                     (winning_relations==0) AND the solve uses an EARNED concept (relational/substrate/gesture)
                     AND it is NOT a single-library lookup for this task (forces RECOMBINATION, not retrieval);
            else ALPHA (verified-but-grammar-expressible / retrieved -> kept small, not zero).

Reuses rel_dsl.verify + ground_arc.winning_relations + effect_faculty.verify_effect VERBATIM. Pure CPU."""
import sys
import rel_dsl as D
import substrate_eye as SE
from effect_faculty import verify_effect, Gesture

sys.path.insert(0, "/data/Windows-files/Documents/airfoil/incubation/evolve")
from ground_arc import winning_relations

ALPHA = 0.2


def is_earned(prog):
    """does the solve use an EARNED concept (relational / substrate sense / gesture), not a bare feature recolor?"""
    if isinstance(prog, Gesture):
        return True
    k = getattr(prog, "key", None)
    return isinstance(k, (D.Quantify, D.Composed, SE.SubQuantify))


def _verify(prog, pairs):
    return verify_effect(prog, pairs) if isinstance(prog, Gesture) else D.verify(prog, pairs)


def single_library_hit(train, test, library):
    """can a SINGLE accumulated library concept solve this task directly? (retrieval, not recombination)."""
    for c in library:
        if isinstance(c, Gesture):
            if verify_effect(c, train) and verify_effect(c, test):
                return True
        else:
            p = D.induce_recolor(c, train)
            if p is not None and D.verify(p, train) and D.verify(p, test):
                return True
    return False


def reward(prog, train, test, library=()):
    if prog is None:
        return 0.0
    solve = 1 if _verify(prog, train) else 0
    gen = 1 if _verify(prog, test) else 0
    if solve == 0 or gen == 0:
        return 0.0
    try:
        gram0 = len(winning_relations(train, test)) == 0
    except Exception:
        gram0 = False
    invented = gram0 and is_earned(prog) and not single_library_hit(train, test, library)
    return float(solve * gen * (1.0 if invented else ALPHA))


def _smoke():
    from grow_library import make_containment_task
    from open_loop import open_solve
    # invented (containment: grammar=0, relational, not a library hit)
    tr, te = make_containment_task(4), make_containment_task(2)
    prog, _k, _c = open_solve(tr, te)
    r_inv = reward(prog, tr, te, library=[])
    # grammar-expressible (recolor by a per-object feature) -> ALPHA
    import grammar as G, numpy as np
    rng = np.random.RandomState(0)

    def recolor_by_size(n):
        ds = []
        for _ in range(n):
            g = np.zeros((12, 12), int)
            for s, (r, c) in zip((1, 2, 3), [(1, 1), (1, 8), (8, 1)]):
                g[r:r+s, c:c+s] = 4
            out = g.copy()
            for o in G.objects(g, 4, True):
                for (rr, cc) in o["cells"]:
                    out[rr, cc] = {1: 2, 4: 3, 9: 5}[o["size"]]
            ds.append((g, out))
        return ds
    tr2, te2 = recolor_by_size(3), recolor_by_size(2)
    prog2, _k2, _c2 = open_solve(tr2, te2)
    r_gram = reward(prog2, tr2, te2, library=[])
    print(f"REWARD smoke: invented-containment R={r_inv} (expect 1.0) | grammar-expressible recolor R={r_gram} (expect {ALPHA})")
    print("READ: R = SOLVE x GEN x NOVELTY, multiplicative -- the creativity bar is the loss weight, Goodhart-resistant.")


if __name__ == "__main__":
    _smoke()
