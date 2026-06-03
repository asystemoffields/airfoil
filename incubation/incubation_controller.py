#!/usr/bin/env python3
"""
Incubation line — the real test: an internal controller that SIMULATES ops through a frozen
interventional world-model under a CAPPED budget, and whether DIRECTED vs NON-DIRECTED search
deploys the non-obvious affordance.

Setup (reuses affordance_v1's world + dual-head model):
  - World-model head: trained ONLY on interventions => knows every op's FULL causal effect
    (task-agnostic; we KEEP the currently-useless structure — the deliberate break from
    value-equivalent/reward-centric MBRL).
  - Policy head: trained on usage-SKEWED goals => its preference IS the usage prior (it
    down-ranks the repurposing op for an r-goal, since that op is "for" register r+1).

The controller simulates ops through the frozen world-model in some ORDER, capped at budget B,
and outputs an op it finds to reach the goal. Directedness = the search ORDER:
  - DIRECTED   : the model's own policy-preference order (follow the usage prior / the obvious).
  - NON-DIRECTED (random)   : explore in random order (ignore the prior).
  - NON-DIRECTED (coverage) : explore by OUTCOME DIVERSITY (try ops with distinct predicted
    effects first) — "imagine the weird pieces too".

Claim (no brain words): with a capped simulation budget, a prior-following search spends the
budget on typical ops and fails to reach the non-obvious affordance; an exploratory search
finds it. The simulator is NECESSARY-BUT-NOT-SUFFICIENT; non-directed exploration is what
points it at the repurposing op. Prediction: on TYPICAL goals directed wins (obvious is right);
on REPURPOSING goals non-directed wins (obvious is a trap) — a crossover.

Run with /data/llm/.venv/bin/python.
"""
import torch

import affordance_v1 as W   # K, M, NOP, DELTAS, REPURP, apply_op, oh, goal_batch, iv_batch, DualMLP, _decode

torch.manual_seed(0)
K, M, NOP = W.K, W.M, W.NOP


def train_model(steps=8000):
    m = W.DualMLP()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    ce = torch.nn.CrossEntropyLoss()
    for _ in range(steps):
        s, o, sp = W.iv_batch(256)
        lw = sum(ce(m.fwd_iv(s, o)[:, k], sp[:, k]) for k in range(K))
        gs, gr, gt, go = W.goal_batch(256)
        lp = ce(m.fwd_goal(gs, gr, gt), go)
        opt.zero_grad(); (lw + lp).backward(); opt.step()
    m.eval()
    return m


def reaches_matrix(s_int, r, tgt):
    """(bs, NOP) bool: does op o, applied to state, land tgt on register r? (ground truth)."""
    bs = s_int.shape[0]
    R = torch.zeros(bs, NOP, dtype=torch.bool)
    for o in range(NOP):
        R[:, o] = W.apply_op(s_int, o)[torch.arange(bs), r] == tgt
    return R


def predicted_r(m, gs, r):
    """(bs, NOP): the model's world-model prediction of register-r value after each op."""
    bs = gs.shape[0]
    P = torch.zeros(bs, NOP, dtype=torch.long)
    with torch.no_grad():
        for o in range(NOP):
            P[:, o] = m.fwd_iv(gs, W.oh(torch.full((bs,), o), NOP)).argmax(-1)[torch.arange(bs), r]
    return P


def coverage_order(pred_r_row):
    """Order ops to front-load DISTINCT predicted outcomes (explore diverse effects first)."""
    order, seen = [], set()
    # first pass: one op per novel predicted-outcome value
    for o in sorted(range(NOP), key=lambda o: pred_r_row[o].item()):
        v = pred_r_row[o].item()
        if v not in seen:
            seen.add(v); order.append(o)
    order += [o for o in range(NOP) if o not in order]   # then the rest
    return order


def frontier(m, repurpose):
    bs = 3000
    gs, gr, gt, go = W.goal_batch(bs, repurpose=repurpose)
    s_int = W._decode(gs); r = gr.argmax(-1); tgt = gt.argmax(-1)
    R = reaches_matrix(s_int, r, tgt)                       # (bs, NOP) which ops actually solve
    with torch.no_grad():
        logits = m.fwd_goal(gs, gr, gt)                    # policy preference (the usage prior)
    directed = torch.argsort(-logits, dim=1)               # high-preference first
    g = torch.Generator().manual_seed(1)
    rand = torch.stack([torch.randperm(NOP, generator=g) for _ in range(bs)])
    pr = predicted_r(m, gs, r)
    cover = torch.tensor([coverage_order(pr[i]) for i in range(bs)])

    def reached_at(order, B):
        idx = order[:, :B]                                  # (bs, B)
        hit = R.gather(1, idx).any(1)                       # any simulated op solves
        return hit.float().mean().item()

    Bs = [1, 2, 3, 5, 7, 9]
    rows = {}
    for name, order in [("directed(prior)", directed), ("non-dir(random)", rand), ("non-dir(coverage)", cover)]:
        rows[name] = [reached_at(order, B) for B in Bs]
    return Bs, rows


def main():
    f = lambda *a: print(*a, flush=True)
    f("=" * 84)
    f("Incubation controller — capped-budget simulation search: directed vs non-directed")
    f("=" * 84)
    m = train_model()
    # sanity: world-model accuracy (the simulator must be good)
    with torch.no_grad():
        s, o, sp = W.iv_batch(2000)
        wm_acc = (m.fwd_iv(s, o).argmax(-1) == sp).float().mean().item()
    f(f"  frozen world-model accuracy (the simulator): {wm_acc*100:.1f}%\n")

    for repurpose, label in [(False, "TYPICAL goals (obvious is right)"),
                             (True, "REPURPOSING goals (obvious is a trap)")]:
        Bs, rows = frontier(m, repurpose)
        f(f"  --- {label} ---")
        f("    budget B:        " + "  ".join(f"{b:>4d}" for b in Bs))
        for name, vals in rows.items():
            f(f"    {name:<18}" + "  ".join(f"{v*100:4.0f}" for v in vals))
        f("")

    f("=" * 84)
    f("READ: reached% vs simulation budget B. Predicted crossover — directed(prior) wins on")
    f("TYPICAL goals (finds the obvious op at small B), but on REPURPOSING goals it wastes the")
    f("budget on typical ops (the repurposing op is ranked last by the prior) while non-directed")
    f("exploration finds it cheaply. Simulator necessary-but-not-sufficient; non-direction = the")
    f("ingredient that points it at the non-obvious affordance.")


if __name__ == "__main__":
    main()
