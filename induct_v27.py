#!/usr/bin/env python3
"""
v27: can the scaffold make a BETTER standalone LLM? (in-context expert-iteration proxy)

The working organ across v22-v26 is search+verify — it produces correct solutions the
model can't produce alone. Those verified solutions are FREE, perfect supervision. The
"better LLM" question: if we feed them back, does the model get better at solving tasks
ALONE? Gradient fine-tuning needs external compute (7GB CPU can't), but in-context is a
cheap, decisive PROXY: if scaffold-verified worked-examples lift alone-accuracy on
held-out tasks, the supervision transfers and real distillation is worth setting up.

Protocol (per domain, reusing v23 tasks):
  - DEMO POOL and TEST set are DISJOINT (held-out).
  - HARVEST: run the DSL scaffold (search+verify) on the demo pool; keep only tasks it
    solves AND whose program is also correct on the held-out probe = verified demos
    (examples -> query -> correct answer). This is supervision the scaffold made itself.
  - TEST: model emits the answer directly (the v23 "alone" task), but with M in {0,2,4,8}
    scaffold-verified demos prepended in-context. M=0 = the v23 alone baseline.
  - Measure alone-accuracy vs M. Rising with M = scaffold supervision transfers (a better
    LLM is reachable by distillation); flat = in-context insufficient, need gradients.

Run with /data/llm/.venv/bin/python.
"""
import re
import time

import induct_v23 as V

N_POOL, N_TEST = 60, 60
MS = [0, 2, 4, 8]
SEED_OFFSET = 7000   # fresh tasks, disjoint from earlier runs


def fmt_io(dom, x, y):
    return f"{x!r} -> {y!r}"


def harvest(dom, pool):
    """Verified worked examples the scaffold produced itself (train-consistent AND correct
    on the held-out probe)."""
    demos = []
    for t in pool:
        prog = V.solve_budget(t["train"], dom.all_ops, dom.ops, dom.eq, dom.ok)
        if not prog:
            continue
        ti, to = t["test"][0]
        if dom.eq(V.predict(prog, ti, dom.ops, dom.ok), to):   # verified correct on the probe
            demos.append((t["train"], ti, to))
    return demos


def demo_block(dom, train, q, a=None):
    lines = "\n".join(fmt_io(dom, x, y) for x, y in train)
    tail = f"\nQ: {q!r}\nA: {a!r}" if a is not None else f"\nQ: {q!r}\nA:"
    return lines + tail


def solve_alone(dom, task, demos, m):
    tag = "/no_think\n" if V._QWEN else ""
    blocks = [demo_block(dom, tr, q, a) for (tr, q, a) in demos[:m]]
    ti, to = task["test"][0]
    blocks.append(demo_block(dom, task["train"], ti, None))
    u = (f"{tag}Each block: input->output examples, then a query Q and its answer A. "
         f"Infer the rule and give A for the LAST block. Reply with only the answer.\n\n"
         + "\n\n".join(blocks))
    p = f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\nA:"
    try:
        txt = V._LLM(p, max_tokens=40, temperature=0.2, stop=["\n\n", "\nQ:", "<|im_end|>"])["choices"][0]["text"]
    except ValueError:
        return False
    pred = dom.parse(re.sub(r"<think>.*?</think>", "", txt, flags=re.S))
    return dom.eq(pred, to)


def main():
    flush = lambda *a: print(*a, flush=True)
    flush("=" * 86)
    flush("v27  CAN THE SCAFFOLD TEACH? — in-context expert-iteration proxy (alone-accuracy vs M demos)")
    flush("=" * 86)
    pools, tests, demobank = {}, {}, {}
    for i, d in enumerate(V.DOMAINS):
        allt = V.make_tasks(d, N_POOL + N_TEST, seed=SEED_OFFSET + i)
        pools[d.name], tests[d.name] = allt[:N_POOL], allt[N_POOL:N_POOL + N_TEST]
        demobank[d.name] = harvest(d, pools[d.name])
        flush(f"  {d.name:<8} harvested {len(demobank[d.name])} verified demos from {N_POOL}-task pool; "
              f"{len(tests[d.name])} held-out test tasks")
    flush("")

    for name, path, is_qwen in V.MODELS:
        t0 = time.time()
        flush(f"----- {name} -----")
        V.load(path, is_qwen)
        for d in V.DOMAINS:
            demos, tt = demobank[d.name], tests[d.name]
            row = []
            for m in MS:
                mm = min(m, len(demos))
                ok = sum(solve_alone(d, t, demos, mm) for t in tt)
                row.append(f"M={m}:{ok/len(tt)*100:4.1f}%")
            flush(f"  {d.name:<8} alone-accuracy  " + "  ".join(row))
        V.unload()
        flush(f"  ({name} done in {time.time()-t0:.0f}s)\n")

    flush("=" * 86)
    flush("RESULT")
    flush("=" * 86)
    flush("  alone-accuracy RISING with M = the scaffold's verified examples teach the model in-")
    flush("  context -> gradient distillation would bake in a better STANDALONE LLM (green-light")
    flush("  the external fine-tune). FLAT = in-context insufficient; the signal needs weights.")


if __name__ == "__main__":
    main()
