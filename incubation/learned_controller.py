#!/usr/bin/env python3
"""
Incubation — LEARNED controller (no heuristics). The exploration policy that decides which ops to
mentally simulate is TRAINED by REINFORCE; the only non-learned piece is the verifier (a goal-check
on what was gathered — a checker, not a search policy).

Two explorers, identical architecture, differing only in their TRAINING OBJECTIVE:
  - DIRECTED   : goal-conditioned, rewarded for GOAL SUCCESS on usage-skewed (typical) goals
                 => learns to gather the typically-useful ops => should FIXATE (fail repurposing).
  - NON-DIRECTED: goal-AGNOSTIC, rewarded for COVERAGE (gathering distinct ops; repeats allowed so
                 spreading is something it must LEARN) => should generalize (gather the repurposing
                 op too) => the verifier can then deploy it for a novel goal.

So the non-directedness is EMERGENT from the objective, not a coded order. Claim: the directed
(success-trained) explorer fixates; the non-directed (coverage-trained) explorer deploys the
non-obvious affordance — and the verifier turns "gathered the right op" into a solve.

World = affordance_v1 (3 regs mod 8, 9 ops; only the repurposing op P_r has an odd effect on
register r). Simulation uses ground-truth effects (= a perfect world-model; the real one is 100%).
Run with /data/llm/.venv/bin/python.
"""
import torch
import torch.nn as nn

import affordance_v1 as W

torch.manual_seed(0)
K, M, NOP = W.K, W.M, W.NOP
B_TRAIN = 6


def sample_goals(n, repurpose):
    s = torch.randint(0, M, (n, K)); r = torch.randint(0, K, (n,)); tgt = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        ri = r[i].item()
        o = W.REPURP[ri] if repurpose else W.TYPICAL[ri][torch.randint(0, len(W.TYPICAL[ri]), (1,)).item()]
        tgt[i] = W.apply_op(s[i:i+1], o)[0, ri]
    return s, r, tgt


def solves(s, r, tgt, op):
    """ground-truth verifier: does op land tgt on register r?"""
    return W.apply_op(s, op)[torch.arange(s.shape[0]), r] == tgt


class Explorer(nn.Module):
    """policy over which op to simulate next, given (counts of ops tried, step, [goal r])."""
    def __init__(self, use_goal):
        super().__init__()
        self.use_goal = use_goal
        din = NOP + 1 + (K if use_goal else 0)
        self.net = nn.Sequential(nn.Linear(din, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, NOP))

    def feat(self, counts, t, r):
        x = [counts / B_TRAIN, torch.full((counts.shape[0], 1), t / B_TRAIN)]
        if self.use_goal:
            x.append(W.oh(r, K))
        return torch.cat(x, -1)

    def forward(self, counts, t, r):
        return self.net(self.feat(counts, t, r))


def train(explorer, directed, steps=4000):
    opt = torch.optim.Adam(explorer.parameters(), lr=3e-3)
    for _ in range(steps):
        n = 256
        s, r, tgt = sample_goals(n, repurpose=False)        # train on TYPICAL goals only (usage skew)
        counts = torch.zeros(n, NOP)
        logps = torch.zeros(n)
        solved = torch.zeros(n, dtype=torch.bool)
        distinct_reward = torch.zeros(n)
        for t in range(B_TRAIN):
            logits = explorer(counts, t, r)
            d = torch.distributions.Categorical(logits=logits)
            op = d.sample(); logps = logps + d.log_prob(op)
            new = counts[torch.arange(n), op] == 0
            distinct_reward += new.float()
            counts[torch.arange(n), op] += 1
            solved |= solves(s, r, tgt, op)
        reward = solved.float() if directed else distinct_reward / B_TRAIN   # objective differs ONLY here
        adv = reward - reward.mean()
        loss = -(logps * adv.detach()).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return explorer


@torch.no_grad()
def evaluate(explorer, repurpose, Bs=(1, 2, 3, 5, 7, 9)):
    n = 3000
    s, r, tgt = sample_goals(n, repurpose)
    counts = torch.zeros(n, NOP)
    ever_solved = torch.zeros(n, dtype=torch.bool)
    reached = {}
    maxB = max(Bs)
    for t in range(maxB):
        logits = explorer(counts, t, r)
        op = logits.argmax(-1)                               # greedy rollout
        counts[torch.arange(n), op] += 1
        ever_solved |= solves(s, r, tgt, op)                 # verifier: did we gather a solving op?
        if (t + 1) in Bs:
            reached[t + 1] = ever_solved.float().mean().item()
    return [reached[b] for b in Bs]


def main():
    f = lambda *a: print(*a, flush=True)
    f("=" * 80)
    f("Incubation — LEARNED explorer (REINFORCE): directed (goal-success) vs non-directed (coverage)")
    f("=" * 80)
    directed = train(Explorer(use_goal=True), directed=True)
    nondir = train(Explorer(use_goal=False), directed=False)
    Bs = (1, 2, 3, 5, 7, 9)
    for repurpose, label in [(False, "TYPICAL goals"), (True, "REPURPOSING goals")]:
        f(f"\n  --- {label} ---")
        f("    budget B:            " + "  ".join(f"{b:>4d}" for b in Bs))
        f("    directed(success)   " + "  ".join(f"{v*100:4.0f}" for v in evaluate(directed, repurpose, Bs)))
        f("    non-dir(coverage)   " + "  ".join(f"{v*100:4.0f}" for v in evaluate(nondir, repurpose, Bs)))
    f("\n" + "=" * 80)
    f("READ: both explorers are LEARNED (REINFORCE) — they differ ONLY in the training objective.")
    f("If the success-trained one fixates on REPURPOSING goals while the coverage-trained one")
    f("deploys the affordance, the non-directedness is EMERGENT from the objective, not coded.")


if __name__ == "__main__":
    main()
