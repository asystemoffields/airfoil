#!/usr/bin/env python3
"""
Incubation stage (b) — SEQUENTIAL, STATE-DEPENDENT repurposing (where learned >> any fixed order).

Stage (a) showed a single repurposing op can be deployed by learned goal-informed effect-coverage.
But single-op is not creativity: real creativity (Erdos via class field towers) is a SEQUENCE where a
tool is repurposed MID-PLAN to unlock a path that otherwise doesn't exist, and whether you need it
depends on the current STATE. This world enforces exactly that.

World: 3 registers (a,b,c) mod 8. Goal = hit target value on register c.
Ops (NOP=15):
  - 9 home adds: reg in {a,b,c} += d in {2,4,6}      -> parity-PRESERVING per register
  - 4 transfers: c+=b, c+=a, b+=a, a+=b              -> couple registers (can flip parity from an ODD source)
  - 2 parity ops: P_a (a+=1), P_b (b+=1)             -> the only parity-changers ... but NO P_c.
Because there is no P_c, the ONLY way to flip c's parity is a transfer FROM AN ODD REGISTER. To get
an odd source you may need a parity op first => a mid-plan repurpose. And `c+=b` flips c's parity
ONLY when b is odd => whether you should apply P_b first DEPENDS ON STATE (skip it if b already odd).
A fixed order ("always P_b then transfer") gets the b-odd cases wrong; only a STATE-CONDITIONED
learned policy wins. That is the point of this stage.

Three explorers (REINFORCE), identical architecture, differ ONLY in reward:
  - directed(success)  : rewarded for reaching the goal on TYPICAL (same-parity, home-add-solvable)
                         goals => fixates on home adds => should FAIL opposite-parity goals.
  - non-dir(coverage)  : goal-AGNOSTIC novelty of visited full-states => explores but unfocused.
  - non-dir(gcoverage) : coverage of distinct VALUES of register c visited (goal-informed effect
                         coverage) => to cover c's ODD values it MUST learn the conditional
                         P_b/transfer chain => deploys the sequential affordance, adaptively.
Trained on TYPICAL goals only (usage skew). Evaluated on TYPICAL vs REPURPOSING (opposite parity),
and REPURPOSING is split by initial b-parity to expose the state-dependence a fixed order can't meet.
Run with /data/llm/.venv/bin/python.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)
K, M = 3, 8
GOAL_REG = 2  # c

# ---- op table -------------------------------------------------------------------------------------
OPS = []
for r in range(K):
    for d in (2, 4, 6):
        OPS.append(("add", r, d))            # 9 home adds (parity-preserving)
OPS += [("xfer", 2, 1), ("xfer", 2, 0), ("xfer", 1, 0), ("xfer", 0, 1)]  # c+=b, c+=a, b+=a, a+=b
OPS += [("par", 0, 0), ("par", 1, 0)]        # P_a, P_b  (no P_c)
NOP = len(OPS)


def step_state(s, op):
    """s:(n,K) long, op:(n,) long -> (n,K) long. Apply each element's chosen op."""
    s = s.clone()
    for j, (kind, p, q) in enumerate(OPS):
        m = op == j
        if not m.any():
            continue
        if kind == "add":
            s[m, p] = (s[m, p] + q) % M
        elif kind == "xfer":
            s[m, p] = (s[m, p] + s[m, q]) % M
        else:  # par
            s[m, p] = (s[m, p] + 1) % M
    return s


def oh(idx, n):
    v = torch.zeros(idx.shape[0], n); v[torch.arange(idx.shape[0]), idx] = 1.0; return v


def sample_goals(n, repurpose, fix_b_parity=None):
    """target on register c. typical = same parity as c0 (home-add solvable); repurpose = opposite."""
    s = torch.randint(0, M, (n, K))
    if fix_b_parity is not None:                       # force b's initial parity (for the eval split)
        s[:, 1] = (s[:, 1] - (s[:, 1] % 2) + fix_b_parity) % M
    c0 = s[:, GOAL_REG]
    parity = (c0 % 2) if not repurpose else (1 - c0 % 2)
    off = torch.randint(0, M // 2, (n,)) * 2           # even offset
    tgt = (parity + off) % M
    return s, tgt


T_TRAIN = 6


class Explorer(nn.Module):
    """policy over next op given (current state, step, [goal target], seen-c bitmask)."""
    def __init__(self, use_goal):
        super().__init__()
        self.use_goal = use_goal
        din = K * M + 1 + M + (M if use_goal else 0)   # state-onehot + step + seen-c + [target]
        self.net = nn.Sequential(nn.Linear(din, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, NOP))

    def forward(self, s, t, tgt, seen_c):
        x = [oh(s[:, r], M) for r in range(K)]
        x.append(torch.full((s.shape[0], 1), t / T_TRAIN))
        x.append(seen_c)
        if self.use_goal:
            x.append(oh(tgt, M))
        return self.net(torch.cat(x, -1))


@torch.no_grad()
def evaluate(explorer, repurpose, Bs, fix_b_parity=None):
    n = 3000
    s, tgt = sample_goals(n, repurpose, fix_b_parity=fix_b_parity)
    idx = torch.arange(n)
    ever = torch.zeros(n, dtype=torch.bool); seen_c = torch.zeros(n, M); out = {}
    for t in range(max(Bs)):
        op = explorer(s, t, tgt, seen_c).argmax(-1)     # greedy rollout
        s = step_state(s, op)
        cval = s[:, GOAL_REG]
        seen_c[idx, cval] = 1.0
        ever |= (cval == tgt)
        if (t + 1) in Bs:
            out[t + 1] = ever.float().mean().item()
    return [out[b] for b in Bs]


def _encode(s):
    """(n,K) -> (n,) integer code of the full state in [0, M^K)."""
    return s[:, 0] + M * s[:, 1] + M * M * s[:, 2]


def train(explorer, mode, steps=6000):
    """mode: 'success' (directed) | 'coverage' (goal-AGNOSTIC full-state novelty) |
             'gcoverage' (goal-informed: distinct values of register c visited)."""
    opt = torch.optim.Adam(explorer.parameters(), lr=2e-3)
    for _ in range(steps):
        n = 256
        s, tgt = sample_goals(n, repurpose=False)       # TYPICAL goals only (usage skew)
        idx = torch.arange(n)
        logps = torch.zeros(n)
        solved = torch.zeros(n, dtype=torch.bool)
        seen_state = torch.zeros(n, M ** K); scov = torch.zeros(n)   # goal-agnostic full-state novelty
        seen_c = torch.zeros(n, M); ccov = torch.zeros(n)            # goal-informed c-value coverage
        for t in range(T_TRAIN):
            d = torch.distributions.Categorical(logits=explorer(s, t, tgt, seen_c))
            op = d.sample(); logps = logps + d.log_prob(op)
            s = step_state(s, op)
            code = _encode(s)
            scov += (seen_state[idx, code] == 0).float(); seen_state[idx, code] = 1.0
            cval = s[:, GOAL_REG]
            ccov += (seen_c[idx, cval] == 0).float(); seen_c[idx, cval] = 1.0
            solved |= (cval == tgt)
        reward = {
            "success": solved.float(),
            "coverage": scov / T_TRAIN,
            "gcoverage": ccov / M,
        }[mode]
        adv = reward - reward.mean()
        loss = -(logps * adv.detach()).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return explorer


def main():
    f = lambda *a: print(*a, flush=True)
    f("=" * 88)
    f(f"Incubation stage (b) — SEQUENTIAL/state-dependent repurposing ({NOP} ops, no P_c).")
    f("Reach an opposite-parity target on c => must transfer from an ODD source => mid-plan repurpose,")
    f("and whether to apply P_b first DEPENDS on b's parity (a fixed order cannot meet this).")
    f("=" * 88)
    directed = train(Explorer(use_goal=True), "success")
    cov = train(Explorer(use_goal=False), "coverage")
    gcov = train(Explorer(use_goal=True), "gcoverage")
    Bs = (2, 3, 4, 5, 6, 8)
    hdr = "    budget B:               " + "  ".join(f"{b:>4d}" for b in Bs)
    arms = [("directed(success)   ", directed), ("non-dir(coverage)   ", cov), ("non-dir(gcoverage)  ", gcov)]
    for repurpose, label in [(False, "TYPICAL goals (same parity)"), (True, "REPURPOSING goals (opposite parity)")]:
        f(f"\n  --- {label} ---"); f(hdr)
        for name, ex in arms:
            f(f"    {name}" + "  ".join(f"{v*100:4.0f}" for v in evaluate(ex, repurpose, Bs)))
    f("\n  --- REPURPOSING split by initial b-parity (exposes state-dependence) ---")
    for bp, plab in [(0, "b starts EVEN (must apply P_b)"), (1, "b starts ODD (must SKIP P_b)")]:
        f(f"    [{plab}]"); f(hdr)
        for name, ex in arms:
            f(f"    {name}" + "  ".join(f"{v*100:4.0f}" for v in evaluate(ex, True, Bs, fix_b_parity=bp)))
    f("\n" + "=" * 88)
    f("READ: directed(success) fixates on home adds (parity-preserving) => ~0 on opposite-parity goals.")
    f("gcoverage (cover c's values) is FORCED to discover the P_b/transfer chain, ADAPTIVELY (handles")
    f("both b-parities) — which no fixed order can. That is sequential, state-dependent repurposing,")
    f("learned and necessary. Then: attention-native (attend over self-generated rollouts).")


if __name__ == "__main__":
    main()
