#!/usr/bin/env python3
"""
Incubation step 7 — value-guided search over the LEARNED world-model (the realism gate).

Step 6 searched the PERFECT simulator (apply_op). Real grounding has only a LEARNED, imperfect model, and
errors COMPOUND across an imagined rollout. Here the controller PLANS over the learned forward model
(s' = s + wl.fwd(E(s), op)) but ACTS in the true world (apply_op), re-planning each step (MPC) — so drift
within a plan is the test, and MPC resets to truth each real step. Question: does value-guided search
survive an imperfect simulator? Compare perfect-model beam (step-6 reference) vs learned-model beam vs
learned-model greedy, on the held-out depth-3 axis (+ refs). Forward-model accuracy diagnostic included
(overall + the rare chain ops P4/T4/C4) to explain any degradation.  Run with /data/llm/.venv/bin/python.
"""
import torch

import multiaxis_struct as MS
from value_search import ValueNet, train_value_togo, oracle   # reuse world-coupled pieces

torch.manual_seed(3)
apply_op, init_states, target_obs, reached, oh = MS.apply_op, MS.init_states, MS.target_obs, MS.reached, MS.oh
NOP, D, H, HELD_OUT, SUBSET = MS.NOP, MS.D, MS.H, MS.HELD_OUT, MS.SUBSET


def make_perfect():
    def trans(states, op):                               # states:(m,D), op:(m,) -> (m,D)
        return apply_op(states, op)
    return trans


def make_learned(wl):
    def trans(states, op):
        return states + wl.fwd(torch.cat([wl.E(states), oh(op, NOP)], -1))   # s' = s + predicted Δs
    return trans


@torch.no_grad()
def plan_first_op(wl, V, s, axis, W, L, trans):
    """beam search guided by V_togo, using `trans` to imagine transitions (perfect or learned)."""
    n = s.shape[0]; ar = torch.arange(n); t = target_obs(s, axis)

    def expand(states):
        m = states.shape[1]
        kids = torch.stack([trans(states.reshape(n * m, D), torch.full((n * m,), j)) for j in range(NOP)], 1)
        kids = kids.view(n, m * NOP, D)
        emb = wl.E(kids.reshape(n * m * NOP, D)).view(n, m * NOP, H)
        reach = (torch.cos(kids[:, :, axis]) < 0.0).float()
        return kids, V(emb, t) + 6.0 * reach

    kids, sc = expand(s.unsqueeze(1))
    topv, topi = sc.topk(min(W, NOP), 1)
    beam = kids.gather(1, topi.unsqueeze(-1).expand(-1, -1, D)); first = topi.clone()
    best_first = first[ar, topv.argmax(1)]
    for _ in range(L - 1):
        Wc = beam.shape[1]
        kids, sc = expand(beam)
        first_rep = first.repeat_interleave(NOP, 1)
        topv, topi = sc.topk(min(W, Wc * NOP), 1)
        beam = kids.gather(1, topi.unsqueeze(-1).expand(-1, -1, D))
        first = first_rep.gather(1, topi)
        best_first = first[ar, topv.argmax(1)]
    return best_first


@torch.no_grad()
def run_search(wl, V, axis, Bs, W, trans, L=5, n=600):
    """PLAN with `trans` (perfect or learned); ACT in the true world (apply_op); MPC re-plan each step."""
    s = init_states(n); ever = torch.zeros(n, dtype=torch.bool); out = {}
    for t in range(max(Bs)):
        op = plan_first_op(wl, V, s, axis, W, L, trans)
        s = apply_op(s, op); ever |= reached(s, axis)             # real transition
        if (t + 1) in Bs: out[t + 1] = ever.float().mean().item()
    return [out[b] for b in Bs]


@torch.no_grad()
def fwd_diagnostic(wl, n=4000):
    s = init_states(n)
    for _ in range(torch.randint(0, 4, (1,)).item()):
        s = apply_op(s, torch.randint(0, NOP, (n,)))
    rows = []
    for j in range(NOP):
        true = apply_op(s, torch.full((n,), j)) - s
        pred = wl.fwd(torch.cat([wl.E(s), oh(torch.full((n,), j), NOP)], -1))
        rows.append(((pred - true) ** 2).mean().item())
    overall = sum(rows) / len(rows)
    return overall, rows[10], rows[11], rows[12]                  # P4, T4, C4


def main():
    f = lambda *a: print(*a, flush=True)
    Bs = (2, 4, 6, 8, 10)
    hdr = "    budget B:                  " + "  ".join(f"{b:>4d}" for b in Bs)
    f("=" * 94)
    f(f"Incubation step 7 — VALUE-GUIDED SEARCH over the LEARNED world-model (realism gate; axis {HELD_OUT}=depth-3)")
    f("=" * 94)
    f("pretraining frozen latent+forward model..."); wl = MS.pretrain_latent(steps=6000)
    ov, p4, t4, c4 = fwd_diagnostic(wl)
    f(f"forward-model Δs MSE: overall {ov:.3f} | chain ops P4 {p4:.3f}  T4 {t4:.3f}  C4 {c4:.3f}")
    f("training V_togo..."); V = train_value_togo(wl)
    perfect, learned = make_perfect(), make_learned(wl)
    for axis, lab in [(HELD_OUT, f"HELD-OUT axis {HELD_OUT} = depth-3 (STRUCTURALLY DIFFERENT) ZERO-SHOT"),
                      (1, "trained-ref axis 1 (depth-2)"), (0, "free axis 0")]:
        f(f"\n  --- {lab} ---"); f(hdr)
        f("    beam PERFECT-model (W=10)" + "  ".join(f"{v*100:4.0f}" for v in run_search(wl, V, axis, Bs, 10, perfect)))
        f("    beam LEARNED-model (W=10)" + "  ".join(f"{v*100:4.0f}" for v in run_search(wl, V, axis, Bs, 10, learned)))
        f("    greedy LEARNED-model(W=1)" + "  ".join(f"{v*100:4.0f}" for v in run_search(wl, V, axis, Bs, 1, learned)))
    f("\n" + "=" * 94)
    f("READ: if LEARNED-model beam stays near PERFECT-model beam on the depth-3 axis, value-guided search")
    f("survives an imperfect simulator (MPC re-planning bounds drift) — the realism gate for grounding.")
    f("If it collapses, the forward-model MSE on the chain ops (esp. C4) explains the error compounding.")


if __name__ == "__main__":
    main()
