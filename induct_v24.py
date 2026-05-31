#!/usr/bin/env python3
"""
v24: close the recognition gap. v23 showed the scaffold is a broad 3-8x amplifier of a
small LLM vs its single-pass self, BUT +airfoil only TIED blind search (captured just
42-73% of the oracle ceiling) — the uplift was search-driven, not recognition-driven.
The model's single-shot category-naming is too noisy to beat undirected search.

This tests two recognition upgrades, reusing v23's domains/harness/tasks (same seeds, so
BLIND and ORACLE are identical to v23):

  - UNION-VOTE : sample the recognizer K times at temperature, UNION the named categories.
                 Recall-oriented — more likely to INCLUDE the needed category (the failure
                 mode was excluding it). Cost: a broader op-set -> shallower reachable depth.
  - +FALLBACK  : narrowed search first; if it finds no verified program, fall back to BLIND
                 full-DSL search. Provably never worse than blind; keeps the cheap narrowed
                 win when recognition is right.

Question: does combo move from ~blind toward oracle (combo > blind = the model's breadth
finally ADDING over search)?  Run with /data/llm/.venv/bin/python.
"""
import re
import time
from collections import Counter
from statistics import mean

import induct_v23 as V

K_VOTES = 3
TEMP = 0.7
N_COMBO = 150


def _recognize_once(dom, train, temp):
    cats = list(dom.cats)
    tag = "/no_think\n" if V._QWEN else ""
    u = (f"An input->output puzzle over {dom.name}. Operation categories: "
         f"{', '.join(cats)}. Which categories are used to transform input to output? "
         f"Reply with only the category words that apply.\n" + dom.render(train))
    p = f"<|im_start|>user\n{tag}{u}<|im_end|>\n<|im_start|>assistant\n"
    try:
        txt = V._LLM(p, max_tokens=40, temperature=temp, stop=["<|im_end|>"])["choices"][0]["text"]
    except ValueError:
        return set()
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S).lower()
    return {c for c in cats if c in txt}


def vote_counter(dom, train, k=K_VOTES, temp=TEMP):
    cnt = Counter()
    for _ in range(k):
        for c in _recognize_once(dom, train, temp):
            cnt[c] += 1
    return cnt


def opset_from(dom, cnt, thresh):
    ops = []
    for c in dom.cats:
        if cnt[c] >= thresh:
            ops += dom.cats[c]
    return list(dict.fromkeys(ops))


def solve(dom, train, opset, fallback):
    sol = V.solve_budget(train, opset, dom.ops, dom.eq, dom.ok)
    if sol is None and fallback:                      # never worse than blind
        sol = V.solve_budget(train, dom.all_ops, dom.ops, dom.eq, dom.ok)
    return sol


# vote ONCE per task, evaluate majority (>=2/3, precision) and union (>=1/3, recall),
# each with/without blind-fallback — all from the same K samples (no extra LLM cost).
def arm_multi(dom, tasks, k=K_VOTES):
    keys = [("maj", False), ("maj", True), ("uni", False), ("uni", True)]
    res = {key: 0 for key in keys}
    sizes = {"maj": [], "uni": []}
    THRESH = {"maj": 2, "uni": 1}
    for t in tasks:
        cnt = vote_counter(dom, t["train"], k)
        opsets = {lab: opset_from(dom, cnt, THRESH[lab]) for lab in ("maj", "uni")}
        for lab in ("maj", "uni"):
            sizes[lab].append(len(opsets[lab]))
        for lab, fb in keys:
            sol = solve(dom, t["train"], opsets[lab], fb)
            if sol and all(dom.eq(V.predict(sol, ti, dom.ops, dom.ok), to) for ti, to in t["test"]):
                res[(lab, fb)] += 1
    return res, {lab: mean(s) if s else 0 for lab, s in sizes.items()}


# v23 single-shot +AIRFOIL (temp 0.2, k=1) for reference, by model+domain
V23_SINGLE = {
    "360M": {"lists": 0.7, "strings": 2.0, "numbers": 16.0},
    "1.7B": {"lists": 27.3, "strings": 56.7, "numbers": 31.3},
}


def main():
    flush = lambda *a: print(*a, flush=True)
    flush("=" * 88)
    flush(f"v24  CLOSE THE RECOGNITION GAP — union-vote (K={K_VOTES}, T={TEMP}) + blind-fallback")
    flush("=" * 88)
    tasks = {d.name: V.make_tasks(d, N_COMBO, seed=1234 + i) for i, d in enumerate(V.DOMAINS)}
    floors = {}
    flush("  search baselines (same tasks/seeds as v23):")
    for d in V.DOMAINS:
        b, n = V.arm_blind(d, tasks[d.name])
        o, _ = V.arm_oracle(d, tasks[d.name])
        floors[d.name] = (b / n * 100, o / n * 100)
        flush(f"    {d.name:<8} BLIND {b/n*100:4.1f}%   ORACLE {o/n*100:4.1f}%")

    for name, path, is_qwen in V.MODELS:
        t0 = time.time()
        flush(f"\n----- {name} -----")
        V.load(path, is_qwen)
        for d in V.DOMAINS:
            bl, orc = floors[d.name]
            n = len(tasks[d.name])
            res, sz = arm_multi(d, tasks[d.name])
            pct = {key: res[key] / n * 100 for key in res}
            single = V23_SINGLE[name][d.name]
            best_fb = max(pct[("maj", True)], pct[("uni", True)])
            flush(f"  {d.name:<8} blind {bl:4.1f} | single {single:4.1f} | "
                  f"MAJ {pct[('maj',False)]:4.1f}(ops~{sz['maj']:.0f}) MAJ+FB {pct[('maj',True)]:4.1f} | "
                  f"UNI {pct[('uni',False)]:4.1f}(ops~{sz['uni']:.0f}) UNI+FB {pct[('uni',True)]:4.1f} | "
                  f"oracle {orc:4.1f} -> best vs-blind {best_fb-bl:+5.1f} capture {best_fb/orc*100 if orc else 0:4.0f}%")
        V.unload()
        flush(f"  ({name} done in {time.time()-t0:.0f}s)")

    flush("\n" + "=" * 88)
    flush("RESULT")
    flush("=" * 88)
    flush("  MAJ (precision, tight op-set) vs UNI (recall, broad) vs v23 single-shot; +FB =")
    flush("  blind-fallback (never worse than blind). Did better recognition push combo ABOVE")
    flush("  blind toward oracle = the model's breadth finally ADDING over search? [LOG after.]")


if __name__ == "__main__":
    main()
