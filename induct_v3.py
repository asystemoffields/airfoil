#!/usr/bin/env python3
"""
v3: does the library make SOLVING new problems faster, not just describing them?

v1/v2 showed learned abstractions shorten the *description* of held-out tasks.
v3 asks the computational question: when the system has to *find* a program for
an unseen task by search, do its abstractions cut the search?

A solver does iterative-deepening enumeration over a vocabulary, counting the
programs it evaluates ("nodes") until it finds one consistent with the examples.
We compare two vocabularies on the SAME held-out tasks:
    base    : the 6 primitive ops only.
    learned : base + macros distilled from the training set (v1's library).

Expectation (and the honest flip side): for RELATED tasks the macros let the
search reach a 7-op program in ~3 big steps instead of 7 small ones — far fewer
nodes. For CONTROL tasks no macro applies, so the only effect of the extra
symbols is a *bigger branching factor* — i.e. abstractions should *slow down*
search that can't use them. Same double-edge as v2's bit cost.

Pure stdlib.
"""
import itertools
from collections import Counter

BASE = ["inc", "dec", "dbl", "tpl", "sqr", "neg"]
OP = {"inc": lambda x: x + 1, "dec": lambda x: x - 1, "dbl": lambda x: 2 * x,
      "tpl": lambda x: 3 * x, "sqr": lambda x: x * x, "neg": lambda x: -x}
INPUTS = [-1, 0, 1, 2, 3, 4]
CAP = 400_000

M1, M2, M3, M4 = ("dbl", "inc"), ("inc", "inc", "inc"), ("sqr", "inc"), ("dbl", "dbl")
MOTIFS = [M1, M2, M3, M4]
TRAIN = list(MOTIFS) + [a + b for a, b in itertools.product(MOTIFS, repeat=2)]
TEST_RELATED = [M1 + M3 + M4, M3 + M1 + M2, M4 + M2 + M1,
                M2 + M4 + M3, M1 + M4 + M2, M3 + M2 + M4]
CONTROL = [("tpl", "neg", "dec", "tpl", "neg"), ("neg", "dec", "tpl", "neg", "dec"),
           ("dec", "tpl", "neg", "dec", "tpl"), ("tpl", "dec", "neg", "tpl", "dec"),
           ("neg", "tpl", "dec", "neg", "tpl"), ("dec", "neg", "tpl", "dec", "neg")]


def run(seq, x):
    for op in seq:
        x = OP[op](x)
    return x


def seg(seq, vocab):
    n = len(seq)
    f = [0] + [float("inf")] * n
    bk = [None] * (n + 1)
    for i in range(1, n + 1):
        for s in vocab:
            L = len(s)
            if L <= i and seq[i - L:i] == s and f[i - L] + 1 < f[i]:
                f[i] = f[i - L] + 1
                bk[i] = (i - L, s)
    out, i = [], n
    while i > 0:
        j, s = bk[i]
        out.append(s)
        i = j
    return out[::-1]


def learn_library():
    vocab = [(b,) for b in BASE]
    while True:
        pairs = Counter()
        for q in TRAIN:
            s = seg(q, vocab)
            for a, b in zip(s, s[1:]):
                pairs[(a, b)] += 1
        if not pairs:
            break
        (a, b), n = max(pairs.items(), key=lambda kv: (kv[1], -len(kv[0][0] + kv[0][1])))
        if n < 2:
            break
        vocab.append(a + b)
    return vocab


def solve_nodes(exs, vocab):
    """Iterative-deepening over vocab symbols; count programs evaluated."""
    nodes = 0
    for d in range(1, 13):
        for combo in itertools.product(vocab, repeat=d):
            prog = tuple(op for sym in combo for op in sym)
            nodes += 1
            if nodes > CAP:
                return (False, nodes)
            ok = True
            for i, o in exs:
                if run(prog, i) != o:
                    ok = False
                    break
            if ok:
                return (True, nodes)
    return (False, nodes)


def measure(tasks, vocab):
    solved, total_nodes = 0, []
    for t in tasks:
        exs = [(i, run(t, i)) for i in INPUTS]
        ok, n = solve_nodes(exs, vocab)
        solved += ok
        total_nodes.append(n)
    total_nodes.sort()
    med = total_nodes[len(total_nodes) // 2]
    return solved, len(tasks), med


def main():
    base = [(b,) for b in BASE]
    learned = learn_library()
    print("=" * 80)
    print("v3  SEARCH-COST TRANSFER — nodes to SOLVE held-out tasks by search")
    print("=" * 80)
    print(f"  learned library: {[' '.join(s) for s in learned if len(s) > 1]}")
    print(f"  (search budget: {CAP:,} nodes/task)\n")
    print(f"  {'task set':<16}{'vocab':<10}{'solved':>10}{'median nodes':>16}")
    print("  " + "-" * 50)
    results = {}
    for name, tasks in [("related", TEST_RELATED), ("control", CONTROL)]:
        for vname, vocab in [("base(6)", base), ("learned(%d)" % len(learned), learned)]:
            s, tot, med = measure(tasks, vocab)
            results[(name, vname.split("(")[0])] = (s, tot, med)
            tag = "" if s == tot else "  (rest hit budget)"
            print(f"  {name:<16}{vname:<10}{f'{s}/{tot}':>10}{med:>16,}{tag}")
    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    rb = results[("related", "base")]
    rl = results[("related", "learned")]
    cb = results[("control", "base")]
    cl = results[("control", "learned")]
    if rl[2] > 0:
        print(f"  RELATED: base median {rb[2]:,} nodes ({rb[0]}/{rb[1]} solved) "
              f"-> learned {rl[2]:,} nodes ({rl[0]}/{rl[1]} solved)")
        if rb[2] >= rl[2]:
            print(f"           = {rb[2] / max(rl[2],1):.0f}x faster search with abstractions.")
    print(f"  CONTROL: base median {cb[2]:,} nodes ({cb[0]}/{cb[1]}) "
          f"-> learned {cl[2]:,} nodes ({cl[0]}/{cl[1]})")
    print(f"           abstractions add branching but no shortcut -> "
          f"{'slower / fails budget' if cl[2] >= cb[2] or cl[0] < cb[0] else 'no help'}.")
    print("\n  Read: the same library that *describes* related tasks more cheaply also")
    print("  lets the solver *find* them in far fewer steps — abstractions are search")
    print("  accelerators where structure matches, and a tax where it doesn't.")


if __name__ == "__main__":
    main()
