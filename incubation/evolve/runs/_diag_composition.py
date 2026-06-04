#!/usr/bin/env python3
"""DECISIVE DIAGNOSTIC for the airfoil creativity campaign (gen-4 composition non-transfer).

QUESTION: is composition's non-transfer a SELECTION problem (a generalizing depth>=2 train-consistent
composition EXISTS in the gen4_01 alphabet space, but the beam picks a non-generalizing one) or an
EXPRESSIVENESS/SEARCH problem (no generalizing composition exists/is findable -> a prior can't help)?

METHOD (see CAMPAIGN.md task):
  - Reuse gen4_01's relational alphabet + fitted concepts (imported verbatim).
  - Probe set = arc1-eval tasks gen2_base MISSES where a depth>=2 train-consistent composition EXISTS.
  - For each probe task: enumerate ALL train-consistent depth<=3 compositions over the alphabet
    (generous budget). Check whether ANY generalizes to the held-out test output (exact match on ALL
    test pairs). Bucket:
        SELECTION-FIXABLE : >=1 train-consistent composition generalizes
        ABSENT            : zero train-consistent compositions generalize
  - For SELECTION-FIXABLE: is the generalizing program prior-distinguishable? (shortest? top-k by a
    simplicity/commonness prior? or indistinguishable from spurious train-consistent ones?)

INTEGRITY: held-out test OUTPUTS are used ONLY in the final generalization check, never to steer search.
Bounded: representative probe set if full enumeration is too slow (reported).
Run with /data/llm/.venv/bin/python from .../incubation/evolve.
"""
import sys, os, json, time, heapq, argparse
from collections import deque, Counter
import numpy as np

HERE = "/data/Windows-files/Documents/airfoil/incubation/evolve"
ARC = "/data/Windows-files/Documents/airfoil/incubation/arc"
for p in (HERE, os.path.join(HERE, "cand"), ARC):
    if p not in sys.path:
        sys.path.insert(0, p)

import dsl
import importlib.util
_spec = importlib.util.spec_from_file_location("g4", os.path.join(HERE, "cand", "gen4_01_relational-depth.py"))
G4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G4)

TRAIN_DIR = "/data/arc/data/training"
EVAL_DIR = "/data/arc/data/evaluation"


def _eq(a, b):
    return a is not None and getattr(a, "shape", None) == getattr(b, "shape", None) and np.array_equal(a, b)


# ---------------------------------------------------------------------------
# EXHAUSTIVE-ish enumeration of train-consistent compositions over the gen4_01 alphabet.
# We do a breadth-first expansion keeping the FRONTIER as (program, outputs-on-train-inputs), deduped
# by output signature so we don't re-expand identical states. We collect EVERY program (depth in
# [min_depth, max_depth]) whose outputs match ALL train targets exactly. Budget = max #verb-executions.
# ---------------------------------------------------------------------------
def enumerate_consistent(train, alphabet, tgts, max_depth=3, budget=40000,
                         frontier_cap=4000, time_limit=30.0, no_repeat_adjacent=False):
    ins = [gi for gi, _ in train]
    found = []                      # list of program name-tuples that verify on train
    found_set = set()
    nexec = 0
    t0 = time.time()
    # frontier: list of (prog_names, outputs). start with identity state (depth 0).
    frontier = [((), ins)]
    seen_sigs = set()
    seen_sigs.add(tuple(o.tobytes() for o in ins))
    truncated = False
    for depth in range(1, max_depth + 1):
        nxt = []
        nxt_sigs = set()
        for prog_names, outs in frontier:
            for name, fn in alphabet:
                if no_repeat_adjacent and prog_names and prog_names[-1] == name:
                    continue
                if nexec >= budget or (time.time() - t0) > time_limit:
                    truncated = True
                    break
                try:
                    nouts = [fn(o) for o in outs]
                except Exception:
                    nouts = None
                nexec += 1
                if nouts is None or any(o is None for o in nouts):
                    continue
                # all must be 2d arrays
                ok = True
                for o in nouts:
                    if getattr(o, "ndim", None) != 2 or o.size == 0:
                        ok = False
                        break
                if not ok:
                    continue
                new_prog = prog_names + (name,)
                # verify against train targets exactly?
                if all(o.shape == t.shape and np.array_equal(o, t) for o, t in zip(nouts, tgts)):
                    if new_prog not in found_set:
                        found_set.add(new_prog)
                        found.append(new_prog)
                    # a verified state is a goal; do not expand it further (its children are >train length
                    # and would not be minimal; but we DO allow other depths to reach it independently)
                    continue
                sig = tuple(o.tobytes() for o in nouts)
                if sig in seen_sigs or sig in nxt_sigs:
                    continue
                nxt_sigs.add(sig)
                nxt.append((new_prog, nouts))
            if truncated:
                break
        # cap frontier size to keep memory/time bounded (keep states closest to target)
        if len(nxt) > frontier_cap:
            def stval(item):
                _pn, outs = item
                return sum(G4._dist(o, t) for o, t in zip(outs, tgts)) / len(tgts)
            nxt.sort(key=stval)
            nxt = nxt[:frontier_cap]
            truncated = True
        seen_sigs |= nxt_sigs
        frontier = nxt
        if truncated or nexec >= budget or (time.time() - t0) > time_limit:
            # we still finished the current depth's verification scan above to the extent budget allowed
            if depth >= max_depth:
                break
            # if we hit budget mid-way we stop expanding deeper
            if nexec >= budget or (time.time() - t0) > time_limit:
                break
    return found, nexec, (time.time() - t0), truncated


def names_to_fns(prog_names, amap):
    return [(n, amap[n]) for n in prog_names]


def run_prog_names(g, prog_names, amap):
    cur = g
    try:
        for n in prog_names:
            cur = amap[n](cur)
            if cur is None or getattr(cur, "ndim", None) != 2 or cur.size == 0:
                return None
        return np.asarray(cur, int)
    except Exception:
        return None


def generalizes(prog_names, amap, test_pairs):
    """Exact match on ALL test pairs (held-out outputs used ONLY here)."""
    for ti, to in test_pairs:
        o = run_prog_names(ti, prog_names, amap)
        if not _eq(o, to):
            return False
    return True


# ---------------------------------------------------------------------------
# A-PRIORI PRIOR for distinguishability analysis.
# We rank train-consistent programs by simplicity priors that DO NOT see the test output:
#   (a) length (shortest first) — pure Occam.
#   (b) a "commonness" weighted length: each verb has a base cost; geometry/recolor are cheaper/common,
#       exotic object ops cost more. Sum of costs, tie-broken by length then lexicographic.
# A generalizing program is PRIOR-DISTINGUISHABLE if it is uniquely the top-ranked (or tied-top) by the
# prior; we also report its rank and whether it shares the minimal length with spurious programs.
# ---------------------------------------------------------------------------
VERB_COST = {}
def verb_cost(name):
    if name in VERB_COST:
        return VERB_COST[name]
    # cheaper = more "common"/generic. geometry & const recolor cheap; object/relational ops pricier.
    c = 3.0
    if name.startswith(("reflect", "rot", "transpose")):
        c = 1.0
    elif name.startswith(("recolor_", "swap_")):
        c = 1.5
    elif name.startswith(("gravity_", "shift_")):
        c = 2.0
    elif name.startswith(("ray_", "connect_", "gravfill_")):
        c = 2.5
    elif name.startswith(("moveobj_",)):
        c = 3.0
    elif name in ("complete_sym", "fill_holes_each", "crop_content"):
        c = 2.5
    VERB_COST[name] = c
    return c


def prog_cost(prog_names):
    return sum(verb_cost(n) for n in prog_names)


def analyze_distinguishability(consistent, gen_set, amap):
    """consistent: list of program-name-tuples (all train-consistent, depth>=2 only considered here).
    gen_set: subset that generalize. Return a dict characterizing whether a prior ranks a generalizer top.
    """
    # restrict to depth>=2 (compositions); single-step are not 'compositions'
    comps = [p for p in consistent if len(p) >= 2]
    gen_comps = [p for p in gen_set if len(p) >= 2]
    n_comp = len(comps)
    n_gen_comp = len(gen_comps)
    res = {"n_consistent_comps": n_comp, "n_generalizing_comps": n_gen_comp}
    if not gen_comps:
        res["distinguishable"] = None
        return res
    # rank by length then cost (Occam + commonness)
    def keyA(p):  # shortest-first
        return (len(p), prog_cost(p), p)
    def keyB(p):  # commonness-cost-first
        return (prog_cost(p), len(p), p)
    sortedA = sorted(comps, key=keyA)
    sortedB = sorted(comps, key=keyB)
    rankA = min(sortedA.index(p) for p in gen_comps)
    rankB = min(sortedB.index(p) for p in gen_comps)
    min_len = min(len(p) for p in comps)
    gen_min_len = min(len(p) for p in gen_comps)
    n_at_min_len = sum(1 for p in comps if len(p) == min_len)
    n_gen_at_min_len = sum(1 for p in gen_comps if len(p) == min_len)
    # top-of-prior: is a generalizer the unique #1 by length-prior?
    best_len_progs = [p for p in comps if len(p) == min_len]
    gen_is_unique_shortest = (gen_min_len == min_len and n_at_min_len == 1 and n_gen_at_min_len == 1)
    gen_among_shortest = (gen_min_len == min_len)
    res.update({
        "rank_by_length_prior": rankA,        # 0 = a generalizer is top-ranked
        "rank_by_cost_prior": rankB,
        "min_comp_len": min_len,
        "gen_min_comp_len": gen_min_len,
        "n_comps_at_min_len": n_at_min_len,
        "n_gen_comps_at_min_len": n_gen_at_min_len,
        "gen_is_unique_shortest": bool(gen_is_unique_shortest),
        "gen_among_shortest": bool(gen_among_shortest),
        # distinguishable if a generalizer sits within top-k of SOME simple prior AND is not buried among
        # a huge tie of spurious equally-simple programs.
        "distinguishable_topk": bool(min(rankA, rankB) < 5),
        "distinguishable_shortest_bucket": bool(gen_among_shortest and n_at_min_len <= 5),
    })
    return res


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def process_task(tid, data_dir, max_depth=3, budget=40000, time_limit=30.0, frontier_cap=4000):
    tr, te = dsl.load_task(os.path.join(data_dir, tid + ".json"))
    train = [(np.asarray(gi, int), np.asarray(go, int)) for gi, go in tr]
    test_pairs = [(np.asarray(gi, int), np.asarray(go, int)) for gi, go in te]
    tgts = [go for _, go in train]
    alphabet, bg, colors = G4.build_alphabet(train)
    amap = dict(alphabet)
    consistent, nexec, dt, truncated = enumerate_consistent(
        train, alphabet, tgts, max_depth=max_depth, budget=budget,
        frontier_cap=frontier_cap, time_limit=time_limit)
    # split by depth
    comps = [p for p in consistent if len(p) >= 2]
    singles = [p for p in consistent if len(p) == 1]
    # generalization check (held-out outputs used ONLY here)
    gen_all = [p for p in consistent if generalizes(p, amap, test_pairs)]
    gen_comps = [p for p in gen_all if len(p) >= 2]
    gen_singles = [p for p in gen_all if len(p) == 1]
    dist = analyze_distinguishability(consistent, gen_all, amap)
    return {
        "task_id": tid,
        "alphabet_size": len(alphabet),
        "n_consistent": len(consistent),
        "n_consistent_comps": len(comps),
        "n_consistent_singles": len(singles),
        "has_comp": len(comps) > 0,
        "n_generalizing": len(gen_all),
        "n_generalizing_comps": len(gen_comps),
        "n_generalizing_singles": len(gen_singles),
        "comp_generalizes": len(gen_comps) > 0,
        "any_generalizes": len(gen_all) > 0,
        "example_gen_comp": list(gen_comps[0]) if gen_comps else None,
        "example_comps": [list(p) for p in comps[:5]],
        "dist": dist,
        "nexec": nexec, "search_sec": round(dt, 2), "truncated": truncated,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["eval", "train"], default="eval")
    ap.add_argument("--max_depth", type=int, default=3)
    ap.add_argument("--budget", type=int, default=40000)
    ap.add_argument("--time_limit", type=float, default=30.0)
    ap.add_argument("--frontier_cap", type=int, default=4000)
    ap.add_argument("--limit", type=int, default=0, help="cap #tasks (0=all misses)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = json.load(open(os.path.join(HERE, "runs", "gen2_base_solved.json")))
    if args.split == "eval":
        data_dir = EVAL_DIR
        missed = base["arc1_eval"]["missed"]
    else:
        data_dir = TRAIN_DIR
        missed = base["arc1_train"]["missed"]
    missed = sorted(missed)
    if args.limit:
        missed = missed[:args.limit]

    print(f"[{args.split}] processing {len(missed)} gen2_base-missed tasks "
          f"(depth<={args.max_depth}, budget={args.budget}, tl={args.time_limit}s)", flush=True)

    results = []
    t0 = time.time()
    for i, tid in enumerate(missed):
        try:
            r = process_task(tid, data_dir, max_depth=args.max_depth, budget=args.budget,
                             time_limit=args.time_limit, frontier_cap=args.frontier_cap)
        except Exception as e:
            r = {"task_id": tid, "error": f"{type(e).__name__}: {e}"}
        results.append(r)
        if (i + 1) % 10 == 0:
            hc = sum(1 for x in results if x.get("has_comp"))
            cg = sum(1 for x in results if x.get("comp_generalizes"))
            print(f"  [{i+1}/{len(missed)}] has_comp={hc} comp_generalizes={cg} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    out = args.out or os.path.join(HERE, "runs", f"_diag_composition_{args.split}.json")
    json.dump({"args": vars(args), "results": results}, open(out, "w"))
    print("WROTE", out, flush=True)

    # ---- summary ----
    probe = [r for r in results if r.get("has_comp")]
    sel = [r for r in probe if r.get("comp_generalizes")]
    absent = [r for r in probe if not r.get("comp_generalizes")]
    n_err = sum(1 for r in results if "error" in r)
    n_trunc = sum(1 for r in probe if r.get("truncated"))
    print("\n===== SUMMARY [%s] =====" % args.split, flush=True)
    print(f"missed tasks processed     : {len(results)} (errors: {n_err})")
    print(f"PROBE SET (depth>=2 train-consistent composition EXISTS): {len(probe)}")
    print(f"  truncated searches in probe set: {n_trunc} (ceiling may undercount)")
    print(f"SELECTION-FIXABLE (>=1 generalizing composition): {len(sel)}")
    print(f"ABSENT (zero generalizing compositions)         : {len(absent)}")
    # distinguishability among selection-fixable
    dshort = sum(1 for r in sel if r["dist"].get("distinguishable_shortest_bucket"))
    dtopk = sum(1 for r in sel if r["dist"].get("distinguishable_topk"))
    duniq = sum(1 for r in sel if r["dist"].get("gen_is_unique_shortest"))
    print(f"  of SELECTION-FIXABLE: distinguishable_shortest_bucket={dshort} "
          f"distinguishable_topk={dtopk} unique_shortest={duniq}")
    # also: tasks where SINGLE concept generalizes (the known eval_beyond_base route)
    n_single_gen = sum(1 for r in results if r.get("n_generalizing_singles", 0) > 0)
    print(f"(context) tasks where a depth-1 single concept generalizes: {n_single_gen}")
    print("SELECTION-FIXABLE task ids:", [r["task_id"] for r in sel])


if __name__ == "__main__":
    main()
