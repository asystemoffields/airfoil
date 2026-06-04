#!/usr/bin/env python3
"""Vine — ANALOGICAL ADAPTATION (Alex's "this looks like THINGY which needed STUFF -> try THINGY-modded + STUFF-
modded"), built on lgg.py's proven schema engine (Hole / antiunify / instantiate). Per the wr36mpy4u synthesis.

A SCHEMA = an earned solution with some slots lifted to typed Holes (a moddable THINGY). STUFF-modded = refill the
holes on a new task (induce+verify, = lgg.instantiate). THINGY-modded = a structural edit (sibling-swap / compose /
add-step) -- the missing half. Retrieve the nearest schema, adapt, verify, anti-unify the winner back -> the library
climbs the generality ladder. Stage 1 here = the NAMED falsifier (STUFF-mod sanity on Vine's EARNED cell solutions):
antiunify Invariance(mirror_h)+Invariance(mirror_v) -> Invariance(.MAP) -> re-instantiate a THIRD axis (rot180/diag)
the parents never saw, with FAR fewer induce-calls than blind. Run: /data/llm/.venv/bin/python schema_adapt.py"""
from collections import namedtuple
import numpy as np
import cell_evolve as CE
from test_generativity import make_inv_task

Hole = namedtuple("Hole", ["sort"])
def _h(x): return isinstance(x, Hole)
MAPS = CE.generative_maps()


def earn_inv(train, test):
    """earn a concrete cell-invariance solution {map, occluder} (or None) -- a ground (hole-free) schema."""
    for name, mapfn in MAPS.items():
        for N in range(10):
            if CE._earns_inv(train, test, name, mapfn, N):
                return {"kind": "invariance", "map": name, "occluder": N}
    return None


def antiunify_inv(s1, s2):
    """lift differing slots to typed Holes (lgg.antiunify, on cell-invariance schemas) -> the moddable THINGY."""
    return {"kind": "invariance",
            "map":      s1["map"]      if s1["map"]      == s2["map"]      else Hole("MAP"),
            "occluder": s1["occluder"] if s1["occluder"] == s2["occluder"] else Hole("COLOR")}


def instantiate_inv(schema, train, test):
    """STUFF-modded: refill the holes on a new task (MAP from the generative map-set, COLOR from 0-9), verify.
    Returns (ground_schema|None, n_induce_calls)."""
    maps = list(MAPS.items()) if _h(schema["map"]) else [(schema["map"], MAPS[schema["map"]])]
    occs = list(range(10)) if _h(schema["occluder"]) else [schema["occluder"]]
    n = 0
    for name, mapfn in maps:
        for N in occs:
            n += 1
            if CE._earns_inv(train, test, name, mapfn, N):
                return {"kind": "invariance", "map": name, "occluder": N}, n
    return None, n


BLIND = {"kind": "invariance", "map": Hole("MAP"), "occluder": Hole("COLOR")}   # fully-holed = from-scratch
PARENT = {"mirror_h": lambda r, c, H, W: (r, W-1-c), "mirror_v": lambda r, c, H, W: (H-1-r, c)}
HELD = {"rot180": lambda r, c, H, W: (H-1-r, W-1-c), "diag": lambda r, c, H, W: (c, r)}


def named_test(trials=10):
    print("STAGE 1 — NAMED falsifier (STUFF-mod): antiunify two earned invariance solutions -> Invariance(.MAP) ->")
    print("          re-instantiate a THIRD axis the parents never saw, cheaper than blind.\n")
    for held_name, held_map in HELD.items():
        solved = 0; lgg_calls = []; blind_calls = []
        for _ in range(trials):
            # earn the two parents (mirror_h, mirror_v) -- they share occluder by construction (make_inv_task occ=9)
            p1 = earn_inv(make_inv_task(4, PARENT["mirror_h"]), make_inv_task(2, PARENT["mirror_h"]))
            p2 = earn_inv(make_inv_task(4, PARENT["mirror_v"]), make_inv_task(2, PARENT["mirror_v"]))
            if not p1 or not p2:
                continue
            schema = antiunify_inv(p1, p2)                       # -> Invariance(.MAP, occluder=9)
            tr, te = make_inv_task(4, held_map), make_inv_task(2, held_map)
            ground, n_lgg = instantiate_inv(schema, tr, te)      # STUFF-mod: refill MAP on the unseen axis
            _, n_blind = instantiate_inv(BLIND, tr, te)          # from-scratch
            if ground is not None:
                solved += 1; lgg_calls.append(n_lgg); blind_calls.append(n_blind)
        ratio = (np.median(blind_calls) / max(1, np.median(lgg_calls))) if lgg_calls else 0
        sch = antiunify_inv({"kind": "invariance", "map": "mirror_h", "occluder": 9},
                            {"kind": "invariance", "map": "mirror_v", "occluder": 9})
        print(f"  held-out '{held_name}': re-instantiated {solved}/{trials}  | induce-calls blind {np.median(blind_calls) if blind_calls else 0:.0f} "
              f"vs schema {np.median(lgg_calls) if lgg_calls else 0:.0f}  ({ratio:.1f}x cheaper)")
    print(f"\n  parent schema after anti-unify: {sch}  (MAP lifted to a hole, occluder kept)")
    print("READ: the antiunified schema re-instantiates an axis the parents never saw (generalizes across the MAP "
          "sort) AND cheaper than blind = the schema/retrieve/refill engine works on Vine's EARNED cell solutions. "
          "Next: THINGY-modded structural ops (compose/add-step) on near-miss families = the make-or-break.")


# ---- STAGE 2: THINGY-modded COMPOSE (the expressiveness-ADDING operator) on a near-miss family ----
def _cmap(inters, tgts):
    table = {}
    for inter, tgt in zip(inters, tgts):
        if inter is None or inter.shape != tgt.shape:
            return None
        for a, b in zip(inter.ravel(), tgt.ravel()):
            a, b = int(a), int(b)
            if a in table and table[a] != b:
                return None
            table[a] = b
    return table


def _apply_cmap(g, table):
    return np.vectorize(lambda v: table.get(int(v), int(v)))(np.asarray(g, int))


def make_symcolor_task(n, mapfn, cmap):
    """near-miss: output = colormap(complete_M(input)) -- symmetry-completion THEN a recolor. The PURE invariance
    earner fires 0 (output is recolored); only COMPOSE(invariance, colormap) solves."""
    base_demos = make_inv_task(n, mapfn)
    return [(corrupt, _apply_cmap(base, cmap)) for (corrupt, base) in base_demos]


def adapt_compose(schema, train, test):
    """THINGY-mod COMPOSE: wrap the retrieved invariance schema with a colormap; refill both holes; verify."""
    maps = list(MAPS.items()) if _h(schema["map"]) else [(schema["map"], MAPS[schema["map"]])]
    occs = list(range(10)) if _h(schema["occluder"]) else [schema["occluder"]]
    n = 0
    for name, mapfn in maps:
        for N in occs:
            n += 1
            inters = [CE._apply_invariance(gi, name, mapfn, N) for gi, _ in train]
            tab = _cmap(inters, [np.asarray(go, int) for _, go in train])
            if tab is None:
                continue
            ok = all((lambda x: x is not None and np.array_equal(_apply_cmap(x, tab), np.asarray(go, int)))(
                CE._apply_invariance(gi, name, mapfn, N)) for gi, go in test)
            if ok:
                return {"compose": [f"invariance:{name}", "colormap"], "occluder": N, "table": tab}, n
    return None, n


def compose_test(trials=12):
    print("\nSTAGE 2 — THINGY-mod COMPOSE (make-or-break): near-miss symmetry o colormap (pure invariance fires 0).")
    print("          adapt = retrieve invariance schema (occluder known) + compose colormap;  blind = from-scratch.\n")
    rng = np.random.RandomState(7)
    AX = {"mirror_h": PARENT["mirror_h"], "mirror_v": PARENT["mirror_v"], "rot180": HELD["rot180"]}
    a_solved = b_solved = b_at_budget = pure_fires = 0; tot = 0
    a_calls, b_calls = [], []
    for _ in range(trials):
        axis = list(AX.values())[rng.randint(0, 3)]
        cmap = {0: 0, 9: 9}
        perm = list(rng.permutation([1, 2, 3, 4, 5, 6]) + 0)
        for i, col in enumerate([1, 2, 3, 4, 5, 6]):
            cmap[col] = int(perm[i])
        tr, te = make_symcolor_task(4, axis, cmap), make_symcolor_task(2, axis, cmap)
        tot += 1
        if earn_inv(tr, te) is not None:                                # the PURE invariance earner should fire 0
            pure_fires += 1
        retrieved = {"kind": "invariance", "map": Hole("MAP"), "occluder": 9}  # retrieved THINGY (occluder from parent)
        ga, na = adapt_compose(retrieved, tr, te)
        gb, nb = adapt_compose(BLIND, tr, te)                          # blind = both holes open (from-scratch)
        if ga is not None:
            a_solved += 1; a_calls.append(na)
        if gb is not None:
            b_solved += 1; b_calls.append(nb)
            if na and nb <= na:
                b_at_budget += 1
    print(f"  pure invariance earner fires: {pure_fires}/{tot} (want 0 -- confirms it's a genuine near-miss)")
    print(f"  ADAPT (compose) solved: {a_solved}/{tot}   median induce-calls {np.median(a_calls) if a_calls else 0:.0f}")
    print(f"  BLIND solved: {b_solved}/{tot}   median induce-calls {np.median(b_calls) if b_calls else 0:.0f}")
    print(f"  BLIND at adapt's budget: {b_at_budget}/{tot}")
    go = a_solved >= int(0.67 * tot) and pure_fires == 0 and a_solved > b_at_budget
    print(f"\n  STAGE 2: {'GO' if go else 'NO-GO'} -- THINGY-mod compose {'ADDS expressiveness adapt-cheaper than blind' if go else 'did not beat blind at equal budget'}")
    print("READ: GO = compose solves near-miss tasks the pure earner can't, and cheaper than from-scratch -> the "
          "analogical layer adds real reach. This is the 'bend a known schema to fit' working on composed effects.")


if __name__ == "__main__":
    named_test()
    compose_test()
