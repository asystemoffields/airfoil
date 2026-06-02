#!/usr/bin/env python3
"""
v26: THE PIVOT — LLM as a PROGRAM proposer in its NATIVE representation (Python).

v22-v25 used the LLM to recognize/select from OUR invented op-taxonomy and it failed
(category-ranking is at chance — the model was never trained on our ontology). But
"write a function that maps these examples" is IN-DISTRIBUTION: code-from-examples is
everywhere in pretraining. So we query the model in the language its breadth is actually
stored in — code — and keep the verifier as the correctness ratchet.

Loop per task: prompt the model (few-shot) to write `def f(x): ...` from the train pairs;
SAMPLE K times; each candidate is SANDBOX-executed and VERIFIED on all train pairs; keep
the first that verifies; apply to held-out test. (Cheap-noisy-proposer + exact verifier =
v8, now with an in-distribution proposer.)

The unlock to test: a one-line `sorted(set(x))[::-1]` is a COMPRESSED deep composition that
the budgeted DSL search (v23-25) can't reach — so the model's code knowledge can directly
buy depth. Does code-propose+verify beat BOTH model-alone AND blind DSL search?

SANDBOX: exec with a whitelisted builtins dict (no __import__/open/eval), per-call SIGALRM
timeout, all exceptions swallowed. Tasks are trivial transforms; this is a local research run.
Run with /data/llm/.venv/bin/python.
"""
import builtins as _b
import re
import signal
import time

import induct_v23 as V

N = 100
K = 4
TEMP = 0.8

_SAFE = {n: getattr(_b, n) for n in [
    "len", "range", "sorted", "sum", "max", "min", "abs", "str", "int", "float", "bool",
    "list", "tuple", "set", "dict", "reversed", "enumerate", "map", "filter", "zip",
    "all", "any", "round", "divmod", "pow", "ord", "chr", "isinstance", "True", "False",
    "None"] if hasattr(_b, n)}


class _Timeout(Exception):
    pass


def _alarm(sig, frame):
    raise _Timeout()


def make_fn(code):
    ns = {}
    exec(code, {"__builtins__": _SAFE}, ns)        # restricted globals
    f = ns.get("f")
    if not callable(f):
        raise ValueError("no f")
    return f


def verify(f, pairs, dom, t=1.0):
    """All train pairs reproduced exactly, under a wall-clock timeout."""
    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, t)
    try:
        for x, y in pairs:
            if not dom.eq(f(x), y):
                return False
        return True
    except Exception:
        return False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def apply1(f, x, t=1.0):
    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, t)
    try:
        return f(x)
    except Exception:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


FEWSHOT = {
    "lists":   "f([3, 1, 2]) == [1, 2, 3]\nf([5, 0, 4]) == [0, 4, 5]\ndef f(x):\n    return sorted(x)",
    "strings": "f('hello world') == 'HELLO WORLD'\nf('foo bar') == 'FOO BAR'\ndef f(x):\n    return x.upper()",
    "numbers": "f(3) == 7\nf(5) == 11\ndef f(x):\n    return x * 2 + 1",
}


def examples_str(train):
    return "\n".join(f"f({x!r}) == {y!r}" for x, y in train)


def propose_code(dom, train, temp):
    tag = "/no_think\n" if V._QWEN else ""
    u = (f"{tag}You are given input->output examples. Study them, then write a Python "
         f"function f(x) that reproduces ALL of them. It may need several steps "
         f"(e.g. sorted(set(x))[::-1]). Reply with ONLY the function.\n\n"
         f"# example\n{FEWSHOT[dom.name]}\n\n# now\n{examples_str(train)}")
    p = f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\ndef f(x):"
    try:
        out = V._LLM(p, max_tokens=120, temperature=temp,
                     stop=["\nf(", "\ndef ", "<|im_end|>", "\n# ", "\n```", "\nprint"])["choices"][0]["text"]
    except ValueError:
        return None
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S)
    body = out if out.strip().startswith(("\n", " ")) or out.startswith("\n") else "\n" + out
    code = "def f(x):" + body
    # keep only the def + indented body lines
    lines, kept = code.split("\n"), []
    for i, ln in enumerate(lines):
        if i == 0 or ln.startswith((" ", "\t")) or ln.strip() == "":
            kept.append(ln)
        else:
            break
    return "\n".join(kept).rstrip()


def arm_code(dom, tasks, k=K):
    solved = verified_train = 0
    for t in tasks:
        winner = None
        for _ in range(k):
            code = propose_code(dom, t["train"], TEMP)
            if not code:
                continue
            try:
                f = make_fn(code)
            except Exception:
                continue
            if verify(f, t["train"], dom):
                winner = f
                break
        if winner is None:
            continue
        verified_train += 1
        if all(dom.eq(apply1(winner, ti), to) for ti, to in t["test"]):
            solved += 1
    return solved, verified_train, len(tasks)


# references (same task distribution): model-alone (v23) and blind DSL search
ALONE = {"360M": {"lists": 10.0, "strings": 6.7, "numbers": 10.0},
         "1.7B": {"lists": 10.0, "strings": 6.7, "numbers": 6.7}}
BLIND = {"lists": 27.3, "strings": 57.3, "numbers": 29.3}


def main():
    flush = lambda *a: print(*a, flush=True)
    flush("=" * 90)
    flush(f"v26  LLM-as-PROGRAM-PROPOSER (native Python) — sample K={K}, sandbox-exec + verify")
    flush("=" * 90)
    tasks = {d.name: V.make_tasks(d, N, seed=1234 + i) for i, d in enumerate(V.DOMAINS)}
    flush(f"  {N} tasks/domain (depth 2-3, held-out test). vs model-alone AND blind DSL search.\n")

    for name, path, is_qwen in V.MODELS:
        t0 = time.time()
        flush(f"----- {name} -----")
        V.load(path, is_qwen)
        for d in V.DOMAINS:
            s, vt, n = arm_code(d, tasks[d.name])
            al, bl = ALONE[name][d.name], BLIND[d.name]
            flush(f"  {d.name:<8} alone {al:4.1f} | blind {bl:4.1f} | "
                  f"CODE-PROPOSE {s/n*100:4.1f}% (train-verified {vt}/{n}={vt/n*100:.0f}%) "
                  f"-> vs-alone {s/n*100-al:+5.1f}  vs-blind {s/n*100-bl:+5.1f}")
        V.unload()
        flush(f"  ({name} done in {time.time()-t0:.0f}s)\n")

    flush("=" * 90)
    flush("RESULT")
    flush("=" * 90)
    flush("  vs-alone>0 = code+verify amplifies the model (expected). vs-blind>0 = the model's")
    flush("  CODE knowledge solves tasks undirected DSL search can't reach = the uplift flips")
    flush("  from search-driven to MODEL-driven. train-verified% = how often any of K samples")
    flush("  reproduced the train pairs (proposal hit-rate); solved% = of those, held-out correct.")


if __name__ == "__main__":
    main()
