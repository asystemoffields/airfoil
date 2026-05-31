#!/usr/bin/env python3
"""
v18: the learned policy in the loop — attacking BRANCHING, co-evolving with the
library that attacks DEPTH.

Search cost is ~|V|^depth. v13-v17 attacked depth (the library turns a deep search
into a shallow one). The other factor, the branching |V|, we left to blind
enumeration — and abstraction even inflates it (more symbols to try). v6 showed a
proposer cuts that, but standalone and bigram-dumb. v18 puts a learned policy
INSIDE the governed wake-sleep loop, co-evolving with the library:

  WAKE   solve train via POLICY-GUIDED best-first search over the current library.
  SLEEP  (a) govern the library by MDL (v14);  (b) retrain the policy on every
         solved trace so far (which symbol tends to follow which).

We measure, on held-out deep tasks, the median search NODES under {uniform-order,
policy-order} over the same library, per round. The library makes deep tasks
reachable; the policy should make reaching them cheap — and the gap should widen
as the two bootstrap each other.

Built on v17's mod-P incompressible domain (each idiom has one sqr -> degree x2 ->
no shorter program exists), so node counts mean what they say. Pure stdlib (the
optional GGUF outer-proposer is staged separately in gguf_proposer.py).
"""
import heapq
import itertools
import math
from collections import Counter
from statistics import median

P = 2_147_483_647
import random
random.seed(18)
TESTPTS = [random.randrange(2, P) for _ in range(12)]

OP = {
    "inc": lambda x: (x + 1) % P, "dec": lambda x: (x - 1) % P,
    "dbl": lambda x: (2 * x) % P, "tpl": lambda x: (3 * x) % P,
    "neg": lambda x: (-x) % P,    "sqr": lambda x: (x * x) % P,
}
BASE = list(OP)
IDIOMS = {"A": ("sqr", "inc"), "B": ("sqr", "dbl"), "C": ("dbl", "sqr"), "D": ("sqr", "neg")}
PHRASES = {"P1": ["A", "B"], "P2": ["C", "D"], "P3": ["B", "C"]}
WAKE_BUDGET = 15000
MEAS_BUDGET = 50000
MAXD = 12


def run(seq, x):
    for op in seq:
        x = OP[op](x)
    return x


def comp(names):
    out = []
    for nm in names:
        out += list(IDIOMS[nm])
    return tuple(out)


def io(prog):
    return tuple(run(prog, x) for x in TESTPTS)


def expand(combo):
    return tuple(op for s in combo for op in s)


def seg(seq, vocab):
    n = len(seq)
    f = [0] + [10**9] * n
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


# ---- search: uniform vs policy-guided (both count programs tested) ----
def uniform_solve(target, vocab, budget):
    n = 0
    for d in range(1, MAXD + 1):
        for combo in itertools.product(vocab, repeat=d):
            n += 1
            if n > budget:
                return None, n
            prog = expand(combo)
            if len(prog) <= MAXD and io(prog) == target:
                return prog, n
    return None, n


def policy_solve(target, vocab, logp, budget):
    """best-first over the policy's program distribution; counts programs tested."""
    n, ctr = 0, 0
    heap = []
    for s in vocab:
        ctr += 1
        heapq.heappush(heap, (-logp(None, s), ctr, (s,), s))
    while heap:
        nlp, _, syms, prog = heapq.heappop(heap)
        n += 1
        if n > budget:
            return None, n
        if io(prog) == target:
            return prog, n
        if len(prog) >= MAXD:
            continue
        for s in vocab:
            if len(prog) + len(s) <= MAXD:
                ctr += 1
                heapq.heappush(heap, (nlp - logp(syms[-1], s), ctr, syms + (s,), prog + s))
    return None, n


# ---- governed sleep (MDL razor, v14) ----
def total_mdl(corpus, vocab):
    lib = sum(len(s) * math.log2(len(BASE)) for s in vocab if len(s) > 1)
    data = sum(len(seg(q, vocab)) for q in corpus) * math.log2(len(vocab))
    return lib + data


def sleep_governed(corpus):
    vocab = [(b,) for b in BASE]
    while True:
        pairs = Counter()
        for q in corpus:
            s = seg(q, vocab)
            for a, b in zip(s, s[1:]):
                pairs[(a, b)] += 1
        cands = [a + b for (a, b), c in pairs.items() if c >= 2]
        best, best_mdl = None, total_mdl(corpus, vocab)
        for m in cands:
            if m not in vocab:
                mdl = total_mdl(corpus, vocab + [m])
                if mdl < best_mdl:
                    best, best_mdl = m, mdl
        if best is None:
            break
        vocab.append(best)
    return vocab


# ---- the policy: bigram over library symbols, fit on solved traces ----
def fit_policy(corpus, vocab):
    trans, ctx = Counter(), Counter()
    for q in corpus:
        prev = None
        for s in seg(q, vocab):
            trans[(prev, s)] += 1
            ctx[prev] += 1
            prev = s
    V = len(vocab)

    def logp(prev, s):
        return math.log((trans[(prev, s)] + 0.3) / (ctx[prev] + 0.3 * V))
    return logp


def main():
    ids = list(IDIOMS)
    train = [comp([i]) for i in ids]
    train += [comp(PHRASES[p]) for p in PHRASES]
    train += [comp(PHRASES[p] + [x]) for p in PHRASES for x in ids]
    train += [comp(PHRASES[p] + PHRASES[q]) for p in PHRASES for q in PHRASES]
    held = [comp(PHRASES[p] + PHRASES[q]) for p, q in (("P1", "P3"), ("P3", "P2"), ("P2", "P1"))]
    held += [comp(PHRASES[p] + PHRASES[q] + PHRASES[r])
             for p, q, r in (("P1", "P3", "P2"), ("P2", "P1", "P3"))]

    print("=" * 80)
    print("v18  LEARNED POLICY IN THE LOOP — branching (policy) x depth (library)")
    print("=" * 80)
    print(f"  wake budget {WAKE_BUDGET:,}; held-out search measured to {MEAS_BUDGET:,} nodes\n")
    print(f"  {'round':<7}{'library':>14}{'held solved':>13}{'uniform nodes':>15}{'policy nodes':>14}{'speedup':>9}")
    print("  " + "-" * 70)

    vocab = [(b,) for b in BASE]
    logp = fit_policy([], vocab)            # uniform to start
    solved = []
    for r in range(5):
        # measure held-out under current library, uniform vs policy order
        u_nodes, p_nodes, solv = [], [], 0
        for t in held:
            tgt = io(t)
            up, un = uniform_solve(tgt, vocab, MEAS_BUDGET)
            pp, pn = policy_solve(tgt, vocab, logp, MEAS_BUDGET)
            solv += (up is not None)
            if up is not None:
                u_nodes.append(un)
            if pp is not None:
                p_nodes.append(pn)
        um = median(u_nodes) if u_nodes else None
        pm = median(p_nodes) if p_nodes else None
        spd = f"{um/pm:.1f}x" if (um and pm) else "-"
        nmac = len([s for s in vocab if len(s) > 1])
        print(f"  {r:<7}{f'{nmac} macros':>14}{f'{solv}/{len(held)}':>13}"
              f"{(f'{um:,}' if um else '-'):>15}{(f'{pm:,}' if pm else '-'):>14}{spd:>9}")
        # WAKE: solve train with policy-guided search, accumulate
        for t in train:
            prog, _ = policy_solve(io(t), vocab, logp, WAKE_BUDGET)
            if prog is None:
                prog, _ = uniform_solve(io(t), vocab, WAKE_BUDGET)
            if prog is not None:
                solved.append(prog)
        # SLEEP: govern library, retrain policy
        new = sleep_governed(solved)
        vocab = new
        logp = fit_policy(solved, vocab)
        if r > 0 and len(new) == nmac + len(BASE) and pm is not None and um is not None and um / pm < 1.05:
            pass  # (no early stop; let it run the fixed rounds)

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    print(f"  library (final): {[' '.join(s) for s in vocab if len(s) > 1]}")
    print("  The library makes deep held-out tasks REACHABLE (depth collapses to a few")
    print("  big symbols); the learned policy makes reaching them CHEAP by trying the")
    print("  high-value macros first instead of blind enumeration over the whole")
    print("  vocabulary — and the two co-evolve round over round (library grows, policy")
    print("  learns to wield it). Depth x branching, both attacked, on a frozen base.")


if __name__ == "__main__":
    main()
