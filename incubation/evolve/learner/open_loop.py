#!/usr/bin/env python3
"""Vine — the OPEN-ENDED unified loop: the recognizer routes the SELF-EVOLVED senses + the hand earns gestures.

V2 is fed the full open sense vocabulary -- regular features + fixed relational channels + the SUBSTRATE channels
(a.p OP b.q, exists/forall) -- and ranks them all (it's feature-agnostic, proven to scale). The policy tries the
top-K as recolor keys, then composes the top relational ones, then reaches for the complete-basis hand (move/copy).
So senses AND gestures are earned + recognizer-routed in one loop. Re-measure: synthetic (one task per faculty)
+ real BARC -- does open-ended faculties move coverage off the recolor-only ~8%? Run: /data/llm/.venv/bin/python open_loop.py [N]"""
import sys, time
import numpy as np
import torch
import grammar as G
import rel_dsl as D
import substrate_eye as SE
from train_v2 import task_VO, FEATS, T
from train_v2_hardened import V2H
from effect_faculty import earn_effect, make_align_task, make_copy_task
from grow_library import make_containment_task

sys.path.insert(0, "/data/Windows-files/Documents/airfoil/incubation/evolve")
from ground_arc import winning_relations
from ground_barc import load_barc_tasks

REGULAR = [D.FeatKey(f) for f in FEATS]
FIXED = list(D.predicate_space())
SUBSTRATE = [SE.SubQuantify(ch, m) for ch in SE.substrate_channels(cross=False) for m in ("exists", "forall")]
ALL_PREDS = REGULAR + FIXED + SUBSTRATE                     # the open sense vocabulary V2 routes
NET = V2H(); NET.load_state_dict(torch.load("learner_v2h.pt")); NET.eval()


def task_VO_open(demos):
    _, O, mask, gvec = task_VO(demos)
    V = np.zeros((T, len(ALL_PREDS)), np.float32); n = 0
    for gi, go in demos:
        objs = G.objects(gi, 4, True)
        for o in objs:
            if n >= T: break
            V[n] = [float(p(o, objs)) for p in ALL_PREDS]; n += 1
    return V, O, mask, gvec


def ranked(demos):
    V, O, m, g = task_VO_open(demos)
    with torch.no_grad():
        _, lf = NET(torch.from_numpy(V[None]), torch.from_numpy(O[None]),
                    torch.from_numpy(m[None]), torch.from_numpy(g[None]))
    return [ALL_PREDS[i] for i in lf[0].argsort(descending=True).tolist()]


def open_solve(train, test, topk=8):
    r = ranked(train); n = 0
    for key in r[:topk]:                                   # recolor, V2-ranked over the OPEN sense vocabulary
        n += 1
        prog = D.induce_recolor(key, train)
        if prog is not None and D.verify(prog, train, test): return prog, "recolor", n
    outers = [k for k in r[:topk] if isinstance(k, D.Quantify)]
    inners = [k for k in r[:topk] if isinstance(k, (D.Quantify, SE.SubQuantify))]
    for outer in outers:                                   # composition (fixed-relational outer x earned inner)
        for inner in inners:
            n += 1
            prog = D.induce_recolor(D.Composed(outer.ch, outer.value, inner, outer.mode), train)
            if prog is not None and D.verify(prog, train, test): return prog, "compose", n
    try:
        eff = earn_effect(train, test)                     # the complete-basis hand (move/copy)
    except Exception:
        eff = None
    if eff is not None: return eff, "gesture", n
    return None, None, n


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    t0 = time.time()
    print("OPEN-ENDED LOOP — one task per faculty (recognizer routes earned senses + gestures):")
    fams = [("containment (fixed channel)", make_containment_task), ("share-height (SUBSTRATE sense)", SE.make_share_height_task),
            ("align (MOVE gesture)", make_align_task), ("copy (COPY gesture)", make_copy_task)]
    for name, gen in fams:
        ok = 0; via = "-"
        for _ in range(10):
            prog, kind, _c = open_solve(gen(4), gen(2))
            ok += int(prog is not None); via = kind if prog is not None else via
        print(f"  {name:<30}: solved {ok}/10  via '{via}'")

    rec = sub = ges = tot = 0
    for tid, train, test, meta in load_barc_tasks(limit=N):
        tot += 1
        try:
            prog, kind, _c = open_solve(train, test)
        except Exception:
            continue
        if kind in ("recolor", "compose"):
            rec += 1
            if isinstance(getattr(prog, "key", None), SE.SubQuantify): sub += 1
        elif kind == "gesture":
            ges += 1
    print(f"\nRE-MEASURE on BARC ARC-Heavy (N={N} -> {tot}) [{time.time()-t0:.0f}s]:")
    print(f"  recolor/compose: {rec} (of which via a SUBSTRATE sense: {sub})   gestures: {ges}   = {rec+ges} total")
    print("READ: the open loop routes earned senses (substrate) + gestures (move/copy) in one recognizer-guided "
          "pass. Coverage delta off the recolor-only ~8% shows whether open-ended FACULTIES help on real data, or "
          "whether the gesture vocabulary (still align-only) needs broadening next (measure, then grow).")


if __name__ == "__main__":
    main()
