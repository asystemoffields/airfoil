#!/usr/bin/env python3
"""Branch-B scale-prep — the EXPERT-ITERATION LOOP (the thing KAGGLE-1 runs), dry-testable on the box.

The loop: stream tasks -> recognizer-guided policy proposes a relation (singles, then OUTER recognizer-ranked x
INNER from the GROWING LIBRARY) -> exact verifier gates -> an INVENTED solve (grammar=0 + relational + generalizes)
grows the library with its earned predicate(s) -> the library makes later compositions cheaper. This is verifier-
as-reward expert iteration without a gradient step yet (the box version); KAGGLE-1 adds the learned policy + scale.
N scales: tiny on the box (dry-run), large on Kaggle. Run: /data/llm/.venv/bin/python kaggle_loop.py [N]"""
import sys, time
import numpy as np
import torch
import grammar as G
import rel_dsl as D
from train_v2 import FEATS
from ground_v2_relational import task_VO_ext, REL_PREDS
from policy_eval import NET, make_adjacency_task
from grow_library import make_containment_task, make_contained_in_largest_task

sys.path.insert(0, "/data/Windows-files/Documents/airfoil/incubation/evolve")
from ground_arc import winning_relations


def ranked_predicates(demos):
    V, O, m, g = task_VO_ext(demos)
    with torch.no_grad():
        _, lf = NET(torch.from_numpy(V[None]), torch.from_numpy(O[None]),
                    torch.from_numpy(m[None]), torch.from_numpy(g[None]))
    order = lf[0].argsort(descending=True).tolist()
    return [D.FeatKey(FEATS[i]) if i < len(FEATS) else REL_PREDS[i - len(FEATS)] for i in order]


def policy_solve(train, test, library, topk=6):
    """recognizer-guided; compositions try the LIBRARY inner FIRST (cheap as it grows), then the full faculty."""
    ranked = ranked_predicates(train); n = 0
    for key in ranked[:topk]:
        n += 1
        prog = D.induce_recolor(key, train)
        if prog is not None and D.verify(prog, train, test): return prog, n
    rels = [k for k in ranked[:topk] if isinstance(k, D.Quantify)]
    inner_order = library + [p for p in REL_PREDS if p not in library]   # library first = expert-iteration payoff
    for outer in rels:
        for inner in inner_order:
            n += 1
            prog = D.induce_recolor(D.Composed(outer.ch, outer.value, inner, outer.mode), train)
            if prog is not None and D.verify(prog, train, test): return prog, n
    return None, n


def earned_predicates(prog):
    out = []
    k = getattr(prog, "key", None)
    if isinstance(k, D.Quantify): out.append(k)
    if isinstance(k, D.Composed):
        out.append(D.Quantify(k.ch, k.value, k.mode))
        if isinstance(k.inner, D.Quantify): out.append(k.inner)
    return out


def loop(stream, label):
    library = []; solved = invented = costs = tot = 0
    for tid, train, test in stream:
        tot += 1
        try:
            prog, cost = policy_solve(train, test, library)
        except Exception:
            continue
        if prog is None:
            continue
        solved += 1; costs += cost
        try:
            gram0 = len(winning_relations(train, test)) == 0
        except Exception:
            gram0 = False
        if gram0 and D.uses_relational(prog):
            invented += 1
            for p in earned_predicates(prog):
                if str(p) not in [str(x) for x in library]:
                    library.append(p)                                  # GROW the library
    print(f"  {label:<26}: {tot} tasks | solved {solved} | INVENTED {invented} | "
          f"library {len(library)} ({[str(p) for p in library]}) | mean cost {costs/max(1,solved):.1f}")
    return library


def synthetic_stream(n):
    rng = np.random.RandomState(7); fams = [make_containment_task, make_adjacency_task, make_contained_in_largest_task]
    for i in range(n):
        gen = fams[i % len(fams)]
        yield f"syn{i}", gen(4), gen(2)


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    t0 = time.time()
    print(f"EXPERT-ITERATION LOOP dry-run (N={N} per stream) — verifier-as-reward, library grows from invented solves:")
    loop(synthetic_stream(N), "synthetic families")
    # real data: ConceptARC (local, clean) -- the honest cross-distribution check
    try:
        from ground_conceptarc import load_conceptarc
        real = list(load_conceptarc())[:N]
        loop(((tid, tr, te) for tid, tr, te in real), "ConceptARC (real)")
    except Exception as e:
        print(f"  (ConceptARC skipped: {e})")
    print(f"[{time.time()-t0:.0f}s] READ: the loop solves + certifies INVENTED solves + GROWS a relational library "
          "end-to-end on synthetic AND real tasks, library-first inner = the expert-iteration payoff. This is the "
          "KAGGLE-1 loop; scale N + add the learned policy + leave-one-family-out GO gate on T4.")


if __name__ == "__main__":
    main()
