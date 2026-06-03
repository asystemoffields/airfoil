#!/usr/bin/env python3
"""
Incubation line — AFFORDANCE world, prerequisite check: does the usage-prior FIXATION
exist, and is it "knows-but-won't-use"?

Abstract operator world (Alex: "answers are there" — determinate, checkable):
  state = K registers mod m. Operators add a fixed Delta (mod m).
  - 6 HOME ops: change ONE register by an EVEN amount (its typical role).
  - 3 REPURPOSING ops P_r: big PRIMARY effect on register (r+1) (their typical role) PLUS a
    latent +1 (ODD) effect on register r. So P_r is the ONLY op that can change register r
    by an ODD amount — but it is "for" register r+1.

Two training signals, decoupled (the point of interventions):
  - INTERVENTION: random (state, op) -> state'  => learns each op's FULL effect (affordance).
  - GOAL (usage-SKEWED): register-r goals are only ever solved with r's TYPICAL ops, which
    are all EVEN on r => the policy learns the usage PRIOR "r-goals use r-typical ops", and
    NEVER sees the odd-offset case.

Test = REPURPOSING goals: odd-offset on register r. Only P_r solves it (latent odd), but
P_r's trained role is register r+1. A reactive policy CAN'T (never trained that pairing).

knows-vs-uses, on a shared trunk:
  - world-model head should KNOW P_r flips register r (trained on interventions).
  - policy head should FAIL the repurposing goal (defaults to a typical/lure op).
That dissociation = the usage-prior fixation substrate, and shows the failure is NOT
ignorance of the affordance — it's inability to DEPLOY known knowledge for a novel goal.
(Overcoming it later = simulate the world-model = the "incubation" channel.)

Run with /data/llm/.venv/bin/python.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)
K, M, NOP = 3, 8, 9

# operators: index -> Delta vector (mod M)
HOME = {0: [[2, 0, 0], [4, 0, 0]], 1: [[0, 2, 0], [0, 4, 0]], 2: [[0, 0, 2], [0, 0, 4]]}
DELTAS, ROLE = [], []          # ROLE[op] = the register this op is "typically for"
for r in range(K):
    for d in HOME[r]:
        DELTAS.append(d); ROLE.append(r)
# repurposing ops: primary +4 on (r+1), latent +1 on r
REPURP = {}                    # r -> op index whose latent ODD effect is on register r
for r in range(K):
    d = [0, 0, 0]
    d[(r + 1) % K] = 4         # primary (even) on its typical register r+1
    d[r] = 1                   # latent ODD on register r
    REPURP[r] = len(DELTAS)
    DELTAS.append(d); ROLE.append((r + 1) % K)   # typical role = r+1
DELTAS = torch.tensor(DELTAS)  # (NOP, K)

# typical ops for register r = ops with ROLE==r (its home ops + the P whose primary is r)
TYPICAL = {r: [o for o in range(NOP) if ROLE[o] == r] for r in range(K)}


def apply_op(state, op):
    return (state + DELTAS[op]) % M


def oh(idx, n):
    v = torch.zeros(idx.shape[0], n)
    v[torch.arange(idx.shape[0]), idx] = 1.0
    return v


def state_oh(s):
    return torch.cat([oh(s[:, k], M) for k in range(K)], -1)


def iv_batch(bs):
    s = torch.randint(0, M, (bs, K))
    op = torch.randint(0, NOP, (bs,))
    sp = apply_op(s, op)
    return state_oh(s), oh(op, NOP), sp


def goal_batch(bs, repurpose=False):
    s = torch.randint(0, M, (bs, K))
    r = torch.randint(0, K, (bs,))
    op = torch.zeros(bs, dtype=torch.long)
    for i in range(bs):
        ri = r[i].item()
        if repurpose:
            op[i] = REPURP[ri]                                   # the odd-latent op for r
        else:
            op[i] = TYPICAL[ri][torch.randint(0, len(TYPICAL[ri]), (1,)).item()]
    tgt = apply_op(s, op)[torch.arange(bs), r]                   # target value on register r
    return state_oh(s), oh(r, K), oh(tgt, M), op


class DualMLP(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.in_iv = nn.Linear(K * M + NOP, d)
        self.in_goal = nn.Linear(K * M + K + M, d)
        self.trunk = nn.Sequential(nn.ReLU(), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.world = nn.Linear(d, K * M)
        self.policy = nn.Linear(d, NOP)

    def fwd_iv(self, s_oh, op_oh):
        return self.world(self.trunk(self.in_iv(torch.cat([s_oh, op_oh], -1)))).view(-1, K, M)

    def fwd_goal(self, s_oh, r_oh, t_oh):
        return self.policy(self.trunk(self.in_goal(torch.cat([s_oh, r_oh, t_oh], -1))))


def main():
    f = lambda *a: print(*a, flush=True)
    f("=" * 76)
    f("Affordance world — prerequisite: does usage-prior fixation exist (knows vs uses)?")
    f("=" * 76)
    f(f"  K={K} regs mod {M}, {NOP} ops. repurposing op per register: {REPURP}")
    m = DualMLP()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    for step in range(8000):
        s, o, sp = iv_batch(256)
        pred = m.fwd_iv(s, o)
        lw = sum(ce(pred[:, k], sp[:, k]) for k in range(K))
        gs, gr, gt, go = goal_batch(256)
        lp = ce(m.fwd_goal(gs, gr, gt), go)
        opt.zero_grad(); (lw + lp).backward(); opt.step()
        if step % 2000 == 0:
            f(f"  step {step:4d}  world-loss {lw.item():.3f}  policy-loss {lp.item():.3f}")

    m.eval()
    with torch.no_grad():
        # KNOWS: world-model accuracy, overall and specifically the latent-odd register of each P
        s, o, sp = iv_batch(2000)
        wp = m.fwd_iv(s, o).argmax(-1)
        world_acc = (wp == sp).float().mean().item()
        # restrict to the P ops, check the register they latently flip
        opi = o.argmax(-1)
        knows = []
        for r in range(K):
            mask = opi == REPURP[r]
            if mask.sum() > 0:
                knows.append((wp[mask][:, r] == sp[mask][:, r]).float().mean().item())
        # USES (typical goals): policy accuracy on train-like goals
        gs, gr, gt, go = goal_batch(2000, repurpose=False)
        pol = m.fwd_goal(gs, gr, gt).argmax(-1)
        typ_acc = (pol == go).float().mean().item()
        # USES (repurposing goals): does the policy reach the goal at all? does it pick the lure?
        gs, gr, gt, go = goal_batch(2000, repurpose=True)
        pred_op = m.fwd_goal(gs, gr, gt).argmax(-1)
        # success = chosen op actually lands the target value on register r
        ri = gr.argmax(-1)
        reached = (apply_op(_decode(gs), pred_op)[torch.arange(2000), ri] == gt.argmax(-1))
        succ = reached.float().mean().item()
        picked_repurp = (pred_op == go).float().mean().item()
        picked_typical = torch.tensor([pred_op[i].item() in TYPICAL[ri[i].item()] for i in range(2000)]).float().mean().item()

    f("\n  --- KNOWS (affordance via interventions) ---")
    f(f"  world-model acc overall: {world_acc*100:.1f}%   on each P's latent-odd register: "
      + " ".join(f"{x*100:.0f}%" for x in knows))
    f("  --- USES (policy) ---")
    f(f"  typical goals: {typ_acc*100:.1f}%   |   REPURPOSING goals: reached {succ*100:.1f}%  "
      f"(picked the repurpose op {picked_repurp*100:.0f}%, picked a typical/lure op {picked_typical*100:.0f}%)")
    # PLANNER upper bound: deploy the affordance by SIMULATING each op via the model's OWN
    # world-model head and choosing one predicted to reach the target (= incubation in principle).
    with torch.no_grad():
        gs, gr, gt, go = goal_batch(2000, repurpose=True)
        s_int = _decode(gs); ri = gr.argmax(-1); tv = gt.argmax(-1)
        reaches = torch.zeros(2000, NOP)
        for op in range(NOP):
            wp = m.fwd_iv(gs, oh(torch.full((2000,), op), NOP)).argmax(-1)
            reaches[:, op] = (wp[torch.arange(2000), ri] == tv).float()
        plan_op = reaches.argmax(-1)                         # pick an op the world-model says works
        reached_plan = (apply_op(s_int, plan_op)[torch.arange(2000), ri] == tv).float().mean().item()
    f(f"  --- PLANNER (simulate via own world-model, pick op that reaches target) ---")
    f(f"  REPURPOSING goals reached by simulation: {reached_plan*100:.1f}%")

    f("\n  THE TRIPLE: knows (world-model) high, reactive policy ~0 on repurposing, planner ~100")
    f("  => the affordance was KNOWN and DEPLOYABLE; only the deployment MECHANISM (simulation)")
    f("  was missing. That mechanism = the incubation channel. Next: make it INTERNAL/trained.")


def _decode(s_oh):
    # recover integer state from its concatenated one-hot
    s = torch.zeros(s_oh.shape[0], K, dtype=torch.long)
    for k in range(K):
        s[:, k] = s_oh[:, k * M:(k + 1) * M].argmax(-1)
    return s


if __name__ == "__main__":
    main()
