#!/usr/bin/env python3
"""
Incubation STAGE (b): compositional depth. Now the goal needs a COMBINATION of ops — the
repurposing op (to break parity on register r) PLUS home ops to reach the exact value — so the
controller searches over op-combinations (subset-sum over the Delta vectors; order-independent
for these additive ops). Budget = number of combinations simulated. Same directed-vs-non-directed
search; prediction: the crossover SCALES and the gap WIDENS (the combinatorial space gives a
prior-following search exponentially more obvious-but-useless combinations to waste budget on,
while outcome-diversity exploration still finds the repurposing-containing combination cheaply).
This is the size-for-time / "incubation buys creative solutions with compute" claim, quantified.

Reuses affordance_v1's world (3 regs mod 8, 9 ops; only the repurposing op P_r has an ODD effect
on register r, so any odd-offset goal on r REQUIRES P_r in the combination). Run with the venv.
"""
import itertools
import torch

import affordance_v1 as W

torch.manual_seed(0)
K, M, NOP = W.K, W.M, W.NOP
DMAX = 4   # max combination size

# all op-combinations (multisets) up to size DMAX
COMBOS = []
for size in range(1, DMAX + 1):
    COMBOS += list(itertools.combinations_with_replacement(range(NOP), size))
SUMDELTA = torch.stack([W.DELTAS[list(c)].sum(0) % M for c in COMBOS])   # (nCombos, K) net effect mod M
NC = len(COMBOS)


def train_policy(steps=6000):
    m = W.DualMLP()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3); ce = torch.nn.CrossEntropyLoss()
    for _ in range(steps):
        s, o, sp = W.iv_batch(256)
        lw = sum(ce(m.fwd_iv(s, o)[:, k], sp[:, k]) for k in range(K))
        gs, gr, gt, go = W.goal_batch(256)
        lp = ce(m.fwd_goal(gs, gr, gt), go)
        opt.zero_grad(); (lw + lp).backward(); opt.step()
    m.eval(); return m


def make_goals(bs, repurpose):
    """target reachable by a combination; repurpose => odd offset on r (requires P_r)."""
    s = torch.randint(0, M, (bs, K)); r = torch.randint(0, K, (bs,))
    tgt = torch.zeros(bs, dtype=torch.long)
    for i in range(bs):
        ri = r[i].item()
        if repurpose:
            combo = [W.REPURP[ri]] + [W.TYPICAL[ri][torch.randint(0, len(W.TYPICAL[ri]), (1,)).item()]
                                      for _ in range(torch.randint(0, 2, (1,)).item())]
        else:
            combo = [W.TYPICAL[ri][torch.randint(0, len(W.TYPICAL[ri]), (1,)).item()]
                     for _ in range(torch.randint(1, 3, (1,)).item())]
        tgt[i] = (s[i, ri] + W.DELTAS[combo].sum(0)[ri]) % M
    return s, r, tgt


def op_prior(m, s, r):
    """per-op preference (the usage prior) from the trained policy, for each goal."""
    gs = W.state_oh(s); gr = W.oh(r, K)
    gt = W.oh(torch.zeros(s.shape[0], dtype=torch.long), M)   # value-agnostic prior over ops
    with torch.no_grad():
        return m.fwd_goal(gs, gr, gt)                          # (bs, NOP)


def frontier(m, repurpose, n=400):
    s, r, tgt = make_goals(n, repurpose)
    prior = op_prior(m, s, r)                                  # (n, NOP)
    # net result on register r for every combo, per goal: s[r] + SUMDELTA[:,r]
    res_r = (s[:, None, :] + SUMDELTA[None, :, :]) % M         # (n, NC, K)
    reaches = res_r[torch.arange(n)[:, None], torch.arange(NC)[None, :], r[:, None]] == tgt[:, None]  # (n, NC)

    # combo priority by ordering (per goal):
    combo_prior = torch.stack([prior[:, list(c)].sum(1) for c in COMBOS], 1)   # (n, NC) sum of op-priors
    directed = torch.argsort(-combo_prior, dim=1)
    g = torch.Generator().manual_seed(1)
    rand = torch.stack([torch.randperm(NC, generator=g) for _ in range(n)])
    # coverage: front-load distinct predicted result[r] values (explore diverse outcomes)
    rr = res_r[torch.arange(n)[:, None], torch.arange(NC)[None, :], r[:, None]]   # (n, NC) result on r
    cover = torch.zeros(n, NC, dtype=torch.long)
    for i in range(n):
        seen, order = set(), []
        for c in sorted(range(NC), key=lambda c: rr[i, c].item()):
            v = rr[i, c].item()
            if v not in seen:
                seen.add(v); order.append(c)
        order += [c for c in range(NC) if c not in set(order)]
        cover[i] = torch.tensor(order)

    Bs = [5, 10, 20, 50, 100, 200, NC]
    out = {}
    for name, order in [("directed(prior)", directed), ("non-dir(random)", rand), ("non-dir(coverage)", cover)]:
        out[name] = [reaches.gather(1, order[:, :B]).any(1).float().mean().item() for B in Bs]
    return Bs, out


def main():
    f = lambda *a: print(*a, flush=True)
    f("=" * 86)
    f(f"Incubation stage (b) — COMPOSITIONAL search (combos<= {DMAX} ops, {NC} candidates)")
    f("=" * 86)
    m = train_policy()
    for repurpose, label in [(False, "TYPICAL goals (combine obvious ops)"),
                             (True, "REPURPOSING goals (need the non-obvious op in the combo)")]:
        Bs, out = frontier(m, repurpose)
        f(f"  --- {label} ---")
        f("    budget B:        " + "  ".join(f"{b:>4d}" for b in Bs))
        for name, vals in out.items():
            f(f"    {name:<18}" + "  ".join(f"{v*100:4.0f}" for v in vals))
        f("")
    f("=" * 86)
    f("READ: vs single-op (stage a), the combinatorial space should WIDEN the gap — directed")
    f("wastes budget on the many obvious combinations; coverage finds the repurposing-containing")
    f("combo cheaply. Gap-widens-with-space = the size-for-time/incubation claim, quantified.")


if __name__ == "__main__":
    main()
