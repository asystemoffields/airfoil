#!/usr/bin/env python3
"""
v25: can the model SELECT? (the crux). v24 showed the 1.7B OVER-includes — asked "which
categories apply?", it names nearly the whole DSL, so narrowing fails and combo=blind. The
model has the breadth but won't pick the relevant few. v25 forces a choice and measures
ranking quality directly.

Change: the recognizer RANKS categories most->least relevant; we take a forced top-K
(K=1,2,3). Two measurements:
  - recall@K : fraction of tasks where the TRUE program's categories are ALL within the
               model's top-K. This is a DIRECT, search-free test of relevance-ranking.
  - accuracy : top-K narrowed search, verified, with/without a blind fallback.

Reads, decisively:
  - recall@K high + accuracy approaches oracle  -> the model CAN select when forced;
    selection-pressure was the missing ingredient (a shippable knob).
  - recall@K low                                -> the model genuinely can't rank task-
    relevance; pivot to the LLM as a PROGRAM proposer, not a category recognizer.

Also fixes v23/v24's set-order noise: all op-sets are ordered by the DSL index, so search
order is deterministic. Run with /data/llm/.venv/bin/python.
"""
import re
import time
from statistics import mean

import induct_v23 as V

N = 150


def order(dom, ops):
    idx = {op: i for i, op in enumerate(dom.all_ops)}
    return sorted(set(ops), key=lambda o: idx[o])


def cats_to_ops(dom, cats):
    ops = []
    for c in cats:
        ops += dom.cats.get(c, [])
    return order(dom, ops)


def oracle_ops(dom, prog):
    return cats_to_ops(dom, sorted({dom.op2cat[op] for op in prog}))


def true_cats(dom, prog):
    return {dom.op2cat[op] for op in prog}


def rank_categories(dom, train):
    """Return the model's category list, ranked most->least relevant (by first mention)."""
    cats = list(dom.cats)
    tag = "/no_think\n" if V._QWEN else ""
    u = (f"An input->output puzzle over {dom.name}. Operation categories: {', '.join(cats)}. "
         f"Rank them from MOST to LEAST likely to be used in this transformation, most "
         f"relevant FIRST, comma-separated.\n" + dom.render(train))
    p = f"<|im_start|>user\n{tag}{u}<|im_end|>\n<|im_start|>assistant\n"
    try:
        txt = V._LLM(p, max_tokens=60, temperature=0.2, stop=["<|im_end|>"])["choices"][0]["text"]
    except ValueError:
        return []
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S).lower()
    pos = {c: txt.find(c) for c in cats if c in txt}
    return sorted(pos, key=lambda c: pos[c])   # ranked by first appearance


def solve(dom, train, opset, fallback):
    sol = V.solve_budget(train, opset, dom.ops, dom.eq, dom.ok)
    if sol is None and fallback:
        sol = V.solve_budget(train, order(dom, dom.all_ops), dom.ops, dom.eq, dom.ok)
    return sol


def run(dom, tasks):
    Ks = [1, 2, 3]
    recall = {k: 0 for k in Ks}
    acc = {(k, fb): 0 for k in Ks for fb in (False, True)}
    for t in tasks:
        ranked = rank_categories(dom, t["train"])
        tc = true_cats(dom, t["prog"])
        for k in Ks:
            topk = ranked[:k]
            if tc.issubset(set(topk)):
                recall[k] += 1
            opset = cats_to_ops(dom, topk)
            for fb in (False, True):
                sol = solve(dom, t["train"], opset, fb)
                if sol and all(dom.eq(V.predict(sol, ti, dom.ops, dom.ok), to) for ti, to in t["test"]):
                    acc[(k, fb)] += 1
    return recall, acc


def main():
    flush = lambda *a: print(*a, flush=True)
    flush("=" * 92)
    flush("v25  CAN THE MODEL SELECT? — forced ranked top-K recognition (+ recall@K, the direct test)")
    flush("=" * 92)
    tasks = {d.name: V.make_tasks(d, N, seed=1234 + i) for i, d in enumerate(V.DOMAINS)}

    # deterministic baselines (ordered op-sets — no set-order noise)
    base = {}
    flush("  deterministic baselines + avg #true-categories/task:")
    for d in V.DOMAINS:
        ts = tasks[d.name]
        bl = sum(1 for t in ts if (lambda s: s and all(d.eq(V.predict(s, ti, d.ops, d.ok), to) for ti, to in t["test"]))(
            V.solve_budget(t["train"], order(d, d.all_ops), d.ops, d.eq, d.ok)))
        orc = sum(1 for t in ts if (lambda s: s and all(d.eq(V.predict(s, ti, d.ops, d.ok), to) for ti, to in t["test"]))(
            V.solve_budget(t["train"], oracle_ops(d, t["prog"]), d.ops, d.eq, d.ok)))
        ncat = mean(len(true_cats(d, t["prog"])) for t in ts)
        base[d.name] = (bl / N * 100, orc / N * 100)
        flush(f"    {d.name:<8} BLIND {bl/N*100:4.1f}%   ORACLE {orc/N*100:4.1f}%   (avg {ncat:.2f} true cats/task)")

    for name, path, is_qwen in V.MODELS:
        t0 = time.time()
        flush(f"\n----- {name} -----")
        V.load(path, is_qwen)
        for d in V.DOMAINS:
            bl, orc = base[d.name]
            rec, acc = run(d, tasks[d.name])
            r = {k: rec[k] / N * 100 for k in rec}
            a = {key: acc[key] / N * 100 for key in acc}
            flush(f"  {d.name:<8} blind {bl:4.1f} oracle {orc:4.1f} | "
                  f"recall@1/2/3 {r[1]:3.0f}/{r[2]:3.0f}/{r[3]:3.0f}% | "
                  f"top1 {a[(1,False)]:4.1f}/{a[(1,True)]:4.1f} top2 {a[(2,False)]:4.1f}/{a[(2,True)]:4.1f} "
                  f"top3 {a[(3,False)]:4.1f}/{a[(3,True)]:4.1f} (raw/+FB)")
        V.unload()
        flush(f"  ({name} done in {time.time()-t0:.0f}s)")

    flush("\n" + "=" * 92)
    flush("RESULT")
    flush("=" * 92)
    flush("  recall@K = can the model RANK the true categories into the top K (direct, search-free).")
    flush("  If recall@2 is high but accuracy stays ~blind -> depth/budget, not selection. If recall@K")
    flush("  is low -> the model can't rank task-relevance -> pivot to LLM-as-program-proposer. [LOG.]")


if __name__ == "__main__":
    main()
