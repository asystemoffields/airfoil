#!/usr/bin/env python3
"""
Incubation — does the LEARNED explorer still win when coverage is NOT cheap? Scale the op space
to ~30 with a simulation budget of 8 (<< ops): now you CANNOT try everything, the repurposing op
is 1-in-30, and blind op-coverage should FAIL. Only an explorer that LEARNS to focus its
exploration on goal-relevant effect-diversity (goal-informed coverage) should still find the
non-obvious affordance. This is where learning genuinely earns its keep (a fixed order can't).

World: 3 registers mod 8. 27 "home" ops (all-EVEN effects — distinct combinations on a register
and an even side-effect) + 3 repurposing ops (odd +1 on register r, primary on r+1). So the
repurposing op for register r is the ONLY op with an ODD effect on r — needed for odd-r goals,
rare, and "typically for" register r+1. Three LEARNED explorers (REINFORCE), differ only in reward.
Run with /data/llm/.venv/bin/python.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)
K, M = 3, 8

DELTAS, ROLE = [], []
for r in range(K):                                   # 27 even "home" ops
    for dr in (2, 4, 6):
        for side in (0, 2, 4):
            d = [0, 0, 0]; d[r] = dr; d[(r + 1) % K] = side
            DELTAS.append(d); ROLE.append(r)
REPURP = {}
for r in range(K):                                   # 3 repurposing ops: odd +1 on r, primary on r+1
    d = [0, 0, 0]; d[r] = 1; d[(r + 1) % K] = 4
    REPURP[r] = len(DELTAS); DELTAS.append(d); ROLE.append((r + 1) % K)
DELTAS = torch.tensor(DELTAS); NOP = len(DELTAS)
TYPICAL = {r: [o for o in range(NOP) if ROLE[o] == r] for r in range(K)}
B_TRAIN = 8


def apply_op(s, op):
    return (s + DELTAS[op]) % M


def oh(idx, n):
    v = torch.zeros(idx.shape[0], n); v[torch.arange(idx.shape[0]), idx] = 1.0; return v


def sample_goals(n, repurpose):
    s = torch.randint(0, M, (n, K)); r = torch.randint(0, K, (n,)); tgt = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        ri = r[i].item()
        o = REPURP[ri] if repurpose else TYPICAL[ri][torch.randint(0, len(TYPICAL[ri]), (1,)).item()]
        tgt[i] = apply_op(s[i:i + 1], o)[0, ri]
    return s, r, tgt


def solves(s, r, tgt, op):
    return apply_op(s, op)[torch.arange(s.shape[0]), r] == tgt


class Explorer(nn.Module):
    def __init__(self, use_goal):
        super().__init__()
        self.use_goal = use_goal
        din = NOP + 1 + (K if use_goal else 0)
        self.net = nn.Sequential(nn.Linear(din, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, NOP))

    def forward(self, counts, t, r):
        x = [counts / B_TRAIN, torch.full((counts.shape[0], 1), t / B_TRAIN)]
        if self.use_goal:
            x.append(oh(r, K))
        return self.net(torch.cat(x, -1))


def train(explorer, mode, steps=5000):
    opt = torch.optim.Adam(explorer.parameters(), lr=2e-3)
    for _ in range(steps):
        n = 256
        s, r, tgt = sample_goals(n, repurpose=False)
        counts = torch.zeros(n, NOP); seen_rval = torch.zeros(n, M)
        logps = torch.zeros(n); solved = torch.zeros(n, dtype=torch.bool); cov = torch.zeros(n); rcov = torch.zeros(n)
        for t in range(B_TRAIN):
            d = torch.distributions.Categorical(logits=explorer(counts, t, r))
            op = d.sample(); logps = logps + d.log_prob(op)
            cov += (counts[torch.arange(n), op] == 0).float(); counts[torch.arange(n), op] += 1
            rval = apply_op(s, op)[torch.arange(n), r]
            rcov += (seen_rval[torch.arange(n), rval] == 0).float(); seen_rval[torch.arange(n), rval] = 1
            solved |= solves(s, r, tgt, op)
        reward = {"success": solved.float(), "coverage": cov / B_TRAIN, "rcoverage": rcov / B_TRAIN}[mode]
        loss = -(logps * (reward - reward.mean()).detach()).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return explorer


@torch.no_grad()
def evaluate(explorer, repurpose, Bs):
    n = 3000
    s, r, tgt = sample_goals(n, repurpose)
    counts = torch.zeros(n, NOP); ever = torch.zeros(n, dtype=torch.bool); out = {}
    for t in range(max(Bs)):
        op = explorer(counts, t, r).argmax(-1)
        counts[torch.arange(n), op] += 1
        ever |= solves(s, r, tgt, op)
        if (t + 1) in Bs:
            out[t + 1] = ever.float().mean().item()
    return [out[b] for b in Bs]


def main():
    f = lambda *a: print(*a, flush=True)
    f("=" * 84)
    f(f"Incubation SCALED — {NOP} ops, budget {B_TRAIN} (coverage NOT cheap): is learned necessary?")
    f("=" * 84)
    directed = train(Explorer(True), "success")
    opcov = train(Explorer(False), "coverage")
    rcov = train(Explorer(True), "rcoverage")
    Bs = (2, 4, 6, 8, 10, 12)
    for repurpose, label in [(False, "TYPICAL goals"), (True, "REPURPOSING goals")]:
        f(f"\n  --- {label} ---")
        f("    budget B:               " + "  ".join(f"{b:>4d}" for b in Bs))
        f("    directed(success)      " + "  ".join(f"{v*100:4.0f}" for v in evaluate(directed, repurpose, Bs)))
        f("    non-dir(op-coverage)   " + "  ".join(f"{v*100:4.0f}" for v in evaluate(opcov, repurpose, Bs)))
        f("    non-dir(r-coverage)    " + "  ".join(f"{v*100:4.0f}" for v in evaluate(rcov, repurpose, Bs)))
    f("\n" + "=" * 84)
    f("READ: with 30 ops / budget 8, blind op-coverage should FAIL on repurposing (can't try the")
    f("1-in-30 op); only goal-informed r-coverage — which must LEARN to seek goal-relevant effect")
    f("diversity rather than enumerate — should find it. That's learning earning its keep.")


if __name__ == "__main__":
    main()
