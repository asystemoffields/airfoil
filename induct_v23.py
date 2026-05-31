#!/usr/bin/env python3
"""
v23: the TRANSFER SUITE — does the airfoil scaffold give a small LLM a BROAD
capability uplift across several distinct example->program domains? (Method
transfer, not one benchmark.)

Design (fixed after a unit test showed small DSLs leave no headroom): use a LARGE
DSL per domain so undirected search drowns — within a fixed node budget, blind
full-DSL search can only reach depth ~1 over ~30 ops, while a model that names the
right ~5-op category reaches depth 3. Breadth (which ops) -> affordable depth.
That is exactly where a model's broad latent knowledge should pay.

Four numbers per domain (held-out test inputs):
  - LLM ALONE   : model emits the answer directly                         -> floor
  - BLIND       : no LLM, full DSL, full node budget                      -> floor
  - LLM+AIRFOIL : model names concept CATEGORIES -> search those ops, verify
  - ORACLE      : narrow to the TRUE program's categories, then search    -> ceiling
Uplift = +AIRFOIL beats BOTH floors, consistently across domains. oracle-blind =
headroom narrowing creates; combo/oracle = fraction the LLM actually captures.

Equal compute: BLIND, COMBO, ORACLE all share the same NODE_BUDGET; they differ
only in WHICH ops they search, i.e. how that budget is spent (breadth vs depth).

Run with /data/llm/.venv/bin/python.
"""
import gc
import itertools
import random
import re
import time

NODE_BUDGET = 160   # ~ enough for a narrowed ~5-op set to reach depth 3 (5+25+125),
                    # but a 20-30 op full DSL only covers depth-1 + a sliver of depth-2.
MAXDEPTH = 4


# ===================== generic budgeted search (depth from breadth) =====================
def _run(prog, x, ops, ok):
    try:
        for nm in prog:
            x = ops[nm](x)
            if x is None or not ok(x):
                return None
        return x
    except Exception:
        return None


def solve_budget(train, opset, ops, eq, ok):
    if not opset:
        return None
    ins = [i for i, _ in train]
    outs = [o for _, o in train]
    tried = 0
    for d in range(1, MAXDEPTH + 1):
        if tried >= NODE_BUDGET:
            break
        for prog in itertools.product(opset, repeat=d):
            tried += 1
            if tried > NODE_BUDGET:
                break
            trans = [_run(prog, i, ops, ok) for i in ins]
            if any(t is None for t in trans):
                continue
            if all(eq(t, o) for t, o in zip(trans, outs)):
                return prog
    return None


def predict(prog, x, ops, ok):
    return _run(prog, x, ops, ok)


# ===================== domains =====================
class Domain:
    def __init__(self, name, ops, cats, render, parse, eq, ok, gen):
        self.name, self.ops, self.cats = name, ops, cats
        self.render, self.parse, self.eq, self.ok, self.gen = render, parse, eq, ok, gen
        self.all_ops = list(ops)
        self.op2cat = {op: c for c, ol in cats.items() for op in ol}


def _L():
    ops = {
        "reverse": lambda a: a[::-1], "sort": lambda a: sorted(a),
        "sort_desc": lambda a: sorted(a, reverse=True), "sort_abs": lambda a: sorted(a, key=abs),
        "rot_left": lambda a: a[1:] + a[:1], "rot_right": lambda a: a[-1:] + a[:-1],
        "unique": lambda a: list(dict.fromkeys(a)),
        "tail": lambda a: a[1:], "init": lambda a: a[:-1], "take2": lambda a: a[:2],
        "take3": lambda a: a[:3], "drop2": lambda a: a[2:],
        "double": lambda a: [x * 2 for x in a], "triple": lambda a: [x * 3 for x in a],
        "inc": lambda a: [x + 1 for x in a], "dec": lambda a: [x - 1 for x in a],
        "negate": lambda a: [-x for x in a], "square": lambda a: [x * x for x in a],
        "half_each": lambda a: [x // 2 for x in a], "plus10": lambda a: [x + 10 for x in a],
        "filter_even": lambda a: [x for x in a if x % 2 == 0],
        "filter_odd": lambda a: [x for x in a if x % 2],
        "filter_pos": lambda a: [x for x in a if x > 0],
        "filter_neg": lambda a: [x for x in a if x < 0],
        "sum1": lambda a: [sum(a)], "max1": lambda a: [max(a)] if a else [],
        "min1": lambda a: [min(a)] if a else [], "len1": lambda a: [len(a)],
        "cumsum": lambda a: [sum(a[:i + 1]) for i in range(len(a))],
        "diffs": lambda a: [a[i] - a[i - 1] for i in range(1, len(a))],
    }
    cats = {
        "order": ["reverse", "sort", "sort_desc", "sort_abs", "rot_left", "rot_right"],
        "dedup": ["unique"],
        "trim": ["tail", "init", "take2", "take3", "drop2"],
        "elementwise": ["double", "triple", "inc", "dec", "negate", "square", "half_each", "plus10"],
        "filter": ["filter_even", "filter_odd", "filter_pos", "filter_neg"],
        "aggregate": ["sum1", "max1", "min1", "len1"],
        "scan": ["cumsum", "diffs"],
    }
    render = lambda pairs: "\n".join(f"{list(i)} -> {list(o)}" for i, o in pairs)
    def parse(txt):
        lines = txt.strip().splitlines()
        nums = re.findall(r"-?\d+", lines[-1]) if lines else []
        return [int(x) for x in nums] if nums else None
    ok = lambda a: isinstance(a, list) and 0 <= len(a) <= 64 and all(-10**7 < x < 10**7 for x in a)
    gen = lambda rng: [rng.randint(-9, 9) for _ in range(rng.randint(4, 6))]
    return Domain("lists", ops, cats, render, parse, lambda a, b: a == b, ok, gen)


def _S():
    ops = {
        "upper": str.upper, "lower": str.lower, "capitalize": str.capitalize,
        "title": str.title, "swapcase": str.swapcase,
        "reverse": lambda s: s[::-1], "sort_chars": lambda s: "".join(sorted(s)),
        "strip": str.strip, "no_space": lambda s: s.replace(" ", ""),
        "dedup_space": lambda s: re.sub(r"\s+", " ", s),
        "no_digit": lambda s: re.sub(r"\d", "", s), "no_vowel": lambda s: re.sub(r"[aeiouAEIOU]", "", s),
        "no_punct": lambda s: re.sub(r"[^\w\s]", "", s), "only_alpha": lambda s: re.sub(r"[^A-Za-z]", "", s),
        "first_word": lambda s: s.split()[0] if s.split() else "",
        "last_word": lambda s: s.split()[-1] if s.split() else "",
        "first_char": lambda s: s[:1], "last_char": lambda s: s[-1:],
        "first3": lambda s: s[:3], "last3": lambda s: s[-3:],
        "add_excl": lambda s: s + "!", "dup": lambda s: s + s,
    }
    cats = {
        "case": ["upper", "lower", "capitalize", "title", "swapcase"],
        "order": ["reverse", "sort_chars"],
        "trim": ["strip", "no_space", "dedup_space"],
        "filter": ["no_digit", "no_vowel", "no_punct", "only_alpha"],
        "select": ["first_word", "last_word", "first_char", "last_char", "first3", "last3"],
        "affix": ["add_excl", "dup"],
    }
    render = lambda pairs: "\n".join(f"'{i}' -> '{o}'" for i, o in pairs)
    def parse(txt):
        ln = [l for l in txt.strip().splitlines() if l.strip()]
        if not ln:
            return None
        s = ln[-1].strip()
        m = re.search(r"'([^']*)'|\"([^\"]*)\"", s)
        return (m.group(1) if m.group(1) is not None else m.group(2)) if m else s
    ok = lambda a: isinstance(a, str) and len(a) <= 200
    WORDS = ["Hello World", "foo123 bar", "  Trim Me  ", "Quick Brown Fox",
             "data42 set", "MixED caSe", "the lazy dog", "abc XYZ 99", "go Cats go"]
    return Domain("strings", ops, cats, render, parse, lambda a, b: a == b, ok, lambda rng: rng.choice(WORDS))


def _N():
    ops = {
        "inc": lambda x: x + 1, "dec": lambda x: x - 1, "plus2": lambda x: x + 2,
        "plus5": lambda x: x + 5, "minus3": lambda x: x - 3, "minus5": lambda x: x - 5,
        "plus10": lambda x: x + 10, "minus10": lambda x: x - 10, "plus20": lambda x: x + 20,
        "plus50": lambda x: x + 50,
        "double": lambda x: x * 2, "triple": lambda x: x * 3, "times5": lambda x: x * 5,
        "times10": lambda x: x * 10, "square": lambda x: x * x, "cube": lambda x: x ** 3,
        "half": lambda x: x // 2,
        "negate": lambda x: -x, "abs": lambda x: abs(x),
        "mod2": lambda x: x % 2, "mod3": lambda x: x % 3, "mod10": lambda x: x % 10,
    }
    cats = {
        "additive": ["inc", "dec", "plus2", "plus5", "minus3", "minus5", "plus10", "minus10", "plus20", "plus50"],
        "multiplicative": ["double", "triple", "times5", "times10", "square", "cube", "half"],
        "sign": ["negate", "abs"], "modulo": ["mod2", "mod3", "mod10"],
    }
    render = lambda pairs: "\n".join(f"{i} -> {o}" for i, o in pairs)
    def parse(txt):
        m = re.findall(r"-?\d+", txt.strip().splitlines()[-1]) if txt.strip() else []
        return int(m[-1]) if m else None
    ok = lambda a: isinstance(a, int) and abs(a) < 10**9
    return Domain("numbers", ops, cats, render, parse, lambda a, b: a == b, ok, lambda rng: rng.randint(-12, 12))


DOMAINS = [_L(), _S(), _N()]


# ===================== task generation =====================
def make_tasks(dom, n, seed):
    rng = random.Random(seed)
    tasks, guard = [], 0
    while len(tasks) < n and guard < n * 80:
        guard += 1
        depth = rng.randint(2, 3)   # composition is the point; depth-1 is a free blind win
        prog = tuple(rng.choice(dom.all_ops) for _ in range(depth))
        pairs, good = [], True
        for _ in range(5):
            x = dom.gen(rng)
            y = predict(prog, x, dom.ops, dom.ok)
            if y is None or dom.eq(y, x):
                good = False
                break
            pairs.append((x, y))
        if not good or len({str(p) for p in pairs}) < 5:
            continue
        tasks.append({"prog": prog, "train": pairs[:3], "test": pairs[3:]})
    return tasks


def oracle_opset(dom, prog):
    cats = {dom.op2cat[op] for op in prog}
    return list(dict.fromkeys(op for c in cats for op in dom.cats[c]))


# ===================== the LLM =====================
_LLM, _QWEN = None, False


def load(path, is_qwen):
    global _LLM, _QWEN
    from llama_cpp import Llama
    _LLM = Llama(model_path=path, n_ctx=2048, n_threads=6, verbose=False)
    _QWEN = is_qwen


def unload():
    global _LLM
    _LLM = None
    gc.collect()


def chat(user, n):
    tag = "/no_think\n" if _QWEN else ""
    p = f"<|im_start|>user\n{tag}{user}<|im_end|>\n<|im_start|>assistant\n"
    try:
        txt = _LLM(p, max_tokens=n, temperature=0.2, stop=["<|im_end|>"])["choices"][0]["text"]
    except ValueError:
        return ""
    return re.sub(r"<think>.*?</think>", "", txt, flags=re.S)


def llm_recognize(dom, train):
    cats = list(dom.cats)
    u = (f"An input->output puzzle over {dom.name}. Operation categories: "
         f"{', '.join(cats)}. Which categories are used to transform input to output? "
         f"Reply with only the category words that apply.\n" + dom.render(train))
    txt = chat(u, 40).lower()
    ops = []
    for c in cats:
        if c in txt:
            ops += dom.cats[c]
    return list(dict.fromkeys(ops))


def llm_solve_direct(dom, train, test_in):
    shown = test_in if dom.name != "strings" else repr(test_in)
    u = ("Infer the rule from the examples, then output ONLY the result for the last "
         "input (no words).\n" + dom.render(train) + f"\nLAST INPUT: {shown}\nLAST OUTPUT:")
    return dom.parse(chat(u, 60))


# ===================== arms =====================
def _score(dom, tasks, opset_fn):
    ok = 0
    for t in tasks:
        sol = solve_budget(t["train"], opset_fn(t), dom.ops, dom.eq, dom.ok)
        if sol and all(dom.eq(predict(sol, ti, dom.ops, dom.ok), to) for ti, to in t["test"]):
            ok += 1
    return ok


def arm_blind(dom, tasks):
    return _score(dom, tasks, lambda t: dom.all_ops), len(tasks)


def arm_oracle(dom, tasks):
    return _score(dom, tasks, lambda t: oracle_opset(dom, t["prog"])), len(tasks)


def arm_alone(dom, tasks):
    ok = sum(dom.eq(llm_solve_direct(dom, t["train"], t["test"][0][0]), t["test"][0][1]) for t in tasks)
    return ok, len(tasks)


def arm_combo(dom, tasks):
    ok = named = 0
    for t in tasks:
        opset = llm_recognize(dom, t["train"])
        named += bool(opset)
        sol = solve_budget(t["train"], opset, dom.ops, dom.eq, dom.ok)
        if sol and all(dom.eq(predict(sol, ti, dom.ops, dom.ok), to) for ti, to in t["test"]):
            ok += 1
    return ok, len(tasks), named


MODELS = [
    ("360M", "/data/Windows-files/Documents/frontier-lab/models/SmolLM2-360M-Instruct-q8_0.gguf", False),
    ("1.7B", "/data/llm/models/qwen3-1.7b-pmra.gguf", True),
]
N_COMBO, N_ALONE = 150, 30


def main():
    flush = lambda *a: print(*a, flush=True)
    flush("=" * 84)
    flush("v23  TRANSFER SUITE — broad uplift from the airfoil scaffold across 3 domains")
    flush("=" * 84)
    tasks = {d.name: make_tasks(d, N_COMBO, seed=1234 + i) for i, d in enumerate(DOMAINS)}
    flush(f"  {N_COMBO} tasks/domain (depth 1-3, held-out test); node budget {NODE_BUDGET}; "
          f"DSL sizes {{{', '.join(f'{d.name}:{len(d.all_ops)}' for d in DOMAINS)}}}\n")

    floors = {}
    flush("  model-independent search baselines:")
    for d in DOMAINS:
        b, n = arm_blind(d, tasks[d.name])
        o, _ = arm_oracle(d, tasks[d.name])
        floors[d.name] = (b / n, o / n)
        flush(f"    {d.name:<8} BLIND {b}/{n}={b/n*100:4.1f}%   ORACLE(true-cat) {o}/{n}={o/n*100:4.1f}%"
              f"   headroom {(o-b)/n*100:+5.1f}")

    for name, path, is_qwen in MODELS:
        t0 = time.time()
        flush(f"\n----- {name} -----")
        load(path, is_qwen)
        for d in DOMAINS:
            a_ok, a_n = arm_alone(d, tasks[d.name][:N_ALONE])
            c_ok, c_n, named = arm_combo(d, tasks[d.name])
            bl, orc = floors[d.name]
            flush(f"  {d.name:<8} ALONE {a_ok}/{a_n}={a_ok/a_n*100:4.1f}% | "
                  f"+AIRFOIL {c_ok}/{c_n}={c_ok/c_n*100:4.1f}% (named {named}/{c_n}) | "
                  f"vs-alone {(c_ok/c_n-a_ok/a_n)*100:+5.1f}  vs-blind {(c_ok/c_n-bl)*100:+5.1f}  "
                  f"capture {c_ok/c_n/orc*100 if orc else 0:4.0f}% of oracle")
        unload()
        flush(f"  ({name} done in {time.time()-t0:.0f}s)")

    flush("\n" + "=" * 84)
    flush("RESULT")
    flush("=" * 84)
    flush("  BROAD uplift = +AIRFOIL beats BOTH floors (alone AND blind) across all 3 domains.")
    flush("  vs-blind>0 = breadth ADDS over undirected search; capture% = how much of the")
    flush("  perfect-recognizer ceiling the small model reaches. [honest LOG entry after.]")


if __name__ == "__main__":
    main()
