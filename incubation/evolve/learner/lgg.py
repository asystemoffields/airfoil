#!/usr/bin/env python3
"""Branch-B anti-unification — Mode-2 PRECONDITION test (the cheap, falsifiable first experiment).

A verified relation is a native grammar.py dict. antiunify(R1,R2) lifts the slots where two verified relations
DIFFER into typed HOLEs -> a generative SCHEMA. The key slot is the recolor TABLE (the per-task-fitted part):
a schema like recolor|(4,True)|rank_size|?TABLE is a GENERATOR — re-instantiate it on a NEW task by routing the
TABLE hole to ONE G.induce() call (fits the table fresh + exact-verifies). The hypothesis (per the design):
LGG is NOT an expressiveness unlock (schema fillers come from the same grammar), but it IS an efficiency organ —
the schema re-instantiates a held-out third task's relation with FAR fewer induce-calls than blind enumeration.
This proves the mechanism + the precondition for the too-big-to-enumerate regime where it cashes out.

GO threshold: canonical 08ed6ac7/67385a82/6e82a1ae triple re-instantiates 3/3 leave-one-out, AND across all
>=2-distinct-table recolor buckets the held-out task re-instantiates >=0.80 with induce-call ratio >=10x vs blind.
Run: /data/llm/.venv/bin/python lgg.py"""
import sys, time
from collections import namedtuple, defaultdict
import numpy as np
import grammar as G
import ground_arc as GA   # winning_relations, generalizes

Hole = namedtuple("Hole", ["sort"])
def _h(x): return isinstance(x, Hole)


def antiunify(R1, R2):
    """least-general-generalization of two verified relation dicts. v0: same effect required."""
    if R1["effect"] != R2["effect"]:
        return None
    eff = R1["effect"]
    if eff == "colormap":
        return {"effect": "colormap", "table": Hole("TABLE")}
    d1, d2 = R1["decomp"], R2["decomp"]
    decomp = (d1[0] if d1[0] == d2[0] else Hole("CONN"),
              d1[1] if d1[1] == d2[1] else Hole("BOOL"))
    feat = R1["feature"] if R1["feature"] == R2["feature"] else Hole("FEAT")
    s = {"effect": eff, "decomp": decomp, "feature": feat}
    if eff == "recolor":
        s["table"] = R1["table"] if R1.get("table") == R2.get("table") else Hole("TABLE")
    else:  # select
        s["mode"] = R1["mode"] if R1.get("mode") == R2.get("mode") else Hole("MODE")
    return s


def instantiate_and_verify(schema, train, test=None):
    """fill atom holes by enumeration; route TABLE/MODE holes to G.induce(); exact-verify (+generalize).
    Returns (relation|None, n_induce_calls)."""
    eff = schema["effect"]
    if eff == "colormap":
        rel = G.induce("colormap", None, None, train)
        ok = rel is not None and (test is None or GA.generalizes(rel, test))
        return (rel if ok else None), 1
    d = schema["decomp"]; feat = schema["feature"]
    conns = [4, 8] if _h(d[0]) else [d[0]]
    bys = [True, False] if _h(d[1]) else [d[1]]
    feats = G.FEATURE_NAMES if _h(feat) else [feat]
    n = 0
    for c in conns:
        for b in bys:
            for f in feats:
                n += 1
                rel = G.induce(eff, (c, b), f, train)   # TABLE/MODE hole fitted+verified inside induce
                if rel is not None and (test is None or GA.generalizes(rel, test)):
                    return rel, n
    return None, n


def blind_count(train, test):
    """skeleton-agnostic blind enumeration of the grammar; induce-calls to solve the task. -> (solved, n)."""
    n = 1
    rel = G.induce("colormap", None, None, train)
    if rel is not None and GA.generalizes(rel, test):
        return True, n
    for eff in ("recolor", "select"):
        for dec in G.DECOMPS:
            for f in G.FEATURE_NAMES:
                n += 1
                rel = G.induce(eff, dec, f, train)
                if rel is not None and GA.generalizes(rel, test):
                    return True, n
    return False, n


def main():
    t0 = time.time()
    # 1) bucket every grammar-solvable ARC-1-train relation by skeleton (effect, decomp, feature)
    buckets = defaultdict(list)   # skeleton -> list of (tid, train, test, rel)
    for tid, train, test in GA.harness.load_split("arc1-train"):
        for eff, feat, rel in GA.winning_relations(train, test):
            if eff == "colormap":
                continue
            sk = (eff, rel["decomp"], feat)
            buckets[sk].append((tid, train, test, rel))
    # keep buckets with >=2 tasks whose induced params DIFFER (distinct relations to anti-unify)
    multi = {}
    for sk, items in buckets.items():
        distinct = []
        seen = set()
        for it in items:
            key = str(it[3].get("table", it[3].get("mode")))
            if key not in seen:
                seen.add(key); distinct.append(it)
        if len(distinct) >= 2:
            multi[sk] = distinct
    print(f"grammar-solvable recolor/select buckets with >=2 distinct-param tasks: {len(multi)}  [{time.time()-t0:.0f}s]")

    # 2) leave-one-out: antiunify 2 parents -> schema -> re-instantiate the held-out third; vs blind
    tri = ("08ed6ac7", "67385a82", "6e82a1ae")
    succ = 0; tot = 0; ratios = []; tri_succ = 0; tri_tot = 0
    for sk, items in multi.items():
        for i in range(len(items)):
            held = items[i]; parents = [items[j] for j in range(len(items)) if j != i][:2]
            if len(parents) < 2:
                continue
            schema = antiunify(parents[0][3], parents[1][3])
            if schema is None:
                continue
            rel, n_lgg = instantiate_and_verify(schema, held[1], held[2])
            solved_b, n_blind = blind_count(held[1], held[2])
            tot += 1; succ += int(rel is not None)
            if rel is not None and solved_b:
                ratios.append(n_blind / max(1, n_lgg))
            if held[0] in tri:
                tri_tot += 1; tri_succ += int(rel is not None)
    print(f"\nLGG leave-one-out re-instantiation (held-out third task), {tot} cases across {len(multi)} buckets:")
    print(f"  re-instantiate+verify+generalize success: {succ}/{tot} = {succ/max(1,tot):.2f}")
    print(f"  induce-call ratio (blind / LGG) on shared solves: median {np.median(ratios) if ratios else 0:.0f}x  (mean {np.mean(ratios) if ratios else 0:.0f}x)")
    print(f"  CANONICAL triple (08ed6ac7/67385a82/6e82a1ae) leave-one-out: {tri_succ}/{tri_tot}")
    print("READ: success>=0.80 AND ratio>=10x = LGG mints a per-task GENERATOR that re-fits cheaply -> the "
          "efficiency precondition holds; cash it in the too-big-to-enumerate composed regime next. (Expected: "
          "0 NEW beyond_gen6 here -- LGG is navigation/efficiency, NOT expressiveness.)")


if __name__ == "__main__":
    main()
