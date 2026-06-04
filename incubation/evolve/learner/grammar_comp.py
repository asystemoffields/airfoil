#!/usr/bin/env python3
"""Branch-B composition — COMPOSED relations: a structural PRE-OP, then a feature-relation.

A composed relation = (pre_op, base_relation): apply a deterministic structural transform (geometry / crop /
scale / tile) to the grid, THEN apply a (decomposition, feature, effect) relation from grammar.py. This:
  * multiplies the relation space past enumerable size (|PRE_OPS| x |rtypes| x decomps), the regime where a
    learned proposer is NECESSARY rather than tidy; and
  * naturally creates SIZE-CHANGING / reshaping tasks (crop->recolor turns input into a smaller output) — the
    cross-shape family our same-shape recognizer was weak on.

KEY: the recognizer stays clean. At solve time we ENUMERATE the ~12 cheap pre-ops; for each we transform the
inputs and let the consistency-recognizer read the feature relevance on the transformed demos -> top-K ->
induce -> exact-verify. So the learner navigates the EXPENSIVE feature x effect x decomp factor; pre-ops are
cheap enumeration. Run: /data/llm/.venv/bin/python grammar_comp.py  (prints a smoke test)."""
import sys
import numpy as np
import grammar as G

sys.path.insert(0, "/data/Windows-files/Documents/airfoil/incubation/arc")
import dsl

# structural pre-ops. OBJECT-PRESERVING ones (geometry + content-crop) keep each object's feature identity
# intact while reshaping/repositioning, so the SAME feature-relation composes cleanly. (scale/tile/downscale
# destroy/duplicate objects and break feature induction -> excluded from v1 composition.)
PRE_OPS = ["identity", "reflect_h", "reflect_v", "rot90", "rot180", "rot270", "transpose", "crop_content"]


def _pre(name, g):
    if name == "identity":
        return g
    try:
        return dsl.OPS[name][0](g)
    except Exception:
        return None


def apply_composed(comp, g):
    """comp = (pre_name, base_rel). pre-transform, then apply the feature-relation."""
    pre_name, base_rel = comp
    g2 = _pre(pre_name, np.asarray(g, int))
    if g2 is None or g2.size == 0:
        return None
    return G.apply_relation(base_rel, g2)


def induce_composed(pre_name, effect, decomp, feature, train):
    """transform inputs by pre_op, induce the base relation on (pre(input), output), return composed or None."""
    train2 = []
    for gi, go in train:
        g2 = _pre(pre_name, gi)
        if g2 is None or g2.size == 0:
            return None
        train2.append((g2, go))
    rel = G.induce(effect, decomp, feature, train2)
    if rel is None:
        return None
    comp = (pre_name, rel)
    # verify end-to-end on the ORIGINAL pairs
    for gi, go in train:
        out = apply_composed(comp, gi)
        if out is None or out.shape != go.shape or not np.array_equal(out, go):
            return None
    return comp


# ---------------------------------------------------------------------------
# curriculum: sample (pre_op, base relation), render demos on random grids.
# ---------------------------------------------------------------------------
def sample_composed(rng, pre_name=None, base_rtype=None, n_demos=4):
    if pre_name is None:
        pre_name = PRE_OPS[rng.randint(0, len(PRE_OPS))]
    types = [t for t in G.all_rtypes() if t != "colormap"]   # composition over object-relations
    if base_rtype is None:
        base_rtype = types[rng.randint(0, len(types))]
    effect, dp, feat = base_rtype.split("|")
    decomp = (int(dp[0]), dp[1] == "c")
    palette = [c for c in range(1, 10)]; rng.shuffle(palette)
    demos = []; rel = None
    # sample the relation's params ONCE; recolor table ACCUMULATES across demos (consistency); select mode fixed.
    if effect == "recolor":
        target = [c for c in range(1, 10) if c not in palette[:4]]; rng.shuffle(target)
        base_spec = {"effect": "recolor", "decomp": decomp, "feature": feat, "table": {}, "_lazy": (feat, target)}
    else:
        mode = G.SELECT_MODES[rng.randint(0, len(G.SELECT_MODES))]
    for _ in range(n_demos):
        for _try in range(12):
            h = rng.randint(8, 16); w = rng.randint(8, 16)
            g = G._rand_grid_with_objects(rng, rng.randint(3, 6), palette[:5], h, w)
            g2 = _pre(pre_name, g)
            if g2 is None or g2.size == 0 or len(G.objects(g2, decomp[0], decomp[1])) < 2:
                continue
            if effect == "recolor":
                base = G._materialize(base_spec, g2, rng)     # accumulates the table -> consistent across demos
            else:
                base = {"effect": "select", "decomp": decomp, "feature": feat, "mode": mode}
            if base is None:
                continue
            out = G.apply_relation(base, g2)
            if out is None or out.size == 0 or np.array_equal(out, g):
                continue
            demos.append((g, out)); rel = base
            if effect == "recolor": base_spec = base          # carry the accumulated table forward
            break
        else:
            return None
    comp = (pre_name, rel)
    for gi, go in demos:
        out = apply_composed(comp, gi)
        if out is None or out.shape != go.shape or not np.array_equal(out, go):
            return None
    return demos, (pre_name, base_rtype)


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    n_types = len(PRE_OPS) * (len(G.all_rtypes()) - 1)
    print(f"composed space: {len(PRE_OPS)} pre-ops x {len(G.all_rtypes())-1} base relations = {n_types} composed types")
    ok = tried = reshape = 0; seen = set()
    for i in range(500):
        out = sample_composed(np.random.RandomState(1000 + i)); tried += 1
        if out is None:
            continue
        demos, (pre_name, brt) = out
        # round-trip: recover via induce_composed (enumerate pre-ops would do this; here check the true pre-op)
        eff, dp, ft = brt.split("|"); dec = (int(dp[0]), dp[1] == "c")
        comp = induce_composed(pre_name, eff, dec, ft, demos)
        if comp is not None:
            ok += 1; seen.add((pre_name, brt))
        if demos[0][0].shape != demos[0][1].shape:
            reshape += 1
    print(f"curriculum round-trip: {ok}/{tried} composed tasks induce-recoverable; {len(seen)} distinct composed types")
    print(f"SIZE-CHANGING (reshaping) tasks: {reshape}/{tried} -> composition reaches the cross-shape family")
    for s in range(50):
        ex = sample_composed(np.random.RandomState(3 + s), pre_name="crop_content")
        if ex is not None:
            d, lab = ex
            print(f"\nexample composed task: pre={lab[0]} base={lab[1]}; demo[0] {d[0][0].shape}->{d[0][1].shape} (reshaped)")
            break
