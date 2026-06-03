#!/usr/bin/env python3
"""Incubation step 14 — OUR OWN VALUE: fuse the Bellman reach-value with AIMED COVERAGE (Alex: "make our own
kind of thingy"). Standard values (MC, Bellman) are reachability estimators -> they under-credit a depth-3
SETUP state (post-P4: goal coord unchanged, but a locked dimension is now OPENED). The incubation mechanism
(steps 1-5; the step-8 'value+novelty' winner) credits exactly that: opening goal-relevant locked dimensions.
Here the beam score = V_bellman(s,t) + 6*reach + BETA * coverage, where coverage = latent novelty of a child
vs the running mean of the real trajectory's latents (rewards moving into NEW latent directions = unlocking).
Sweep BETA (0 = pure Bellman baseline ~75) on the held-out depth-3 axis. Run with /data/llm/.venv/bin/python."""
import torch

import multiaxis_struct as MS
import value_search as VS
from learned_value import train_value_bellman

torch.manual_seed(0)
apply_op, init_states, target_obs, reached = MS.apply_op, MS.init_states, MS.target_obs, MS.reached
NOP, D, H, HELD_OUT, SUBSET = MS.NOP, MS.D, MS.H, MS.HELD_OUT, MS.SUBSET
Bs = (2, 4, 6, 8, 10)


@torch.no_grad()
def plan_cov(wl, V, s, axis, W, L, mem, beta):
    n = s.shape[0]; ar = torch.arange(n); t = target_obs(s, axis)

    def expand(states):
        m = states.shape[1]
        kids = torch.stack([apply_op(states.reshape(n * m, D), torch.full((n * m,), j)) for j in range(NOP)], 1).view(n, m * NOP, D)
        emb = wl.E(kids.reshape(n * m * NOP, D)).view(n, m * NOP, H)
        reach = (torch.cos(kids[:, :, axis]) < 0.0).float()
        nov = ((emb - mem.unsqueeze(1)) ** 2).mean(-1)          # aimed coverage: new latent directions
        return kids, V(emb, t) + 6.0 * reach + beta * nov

    kids, sc = expand(s.unsqueeze(1)); topv, topi = sc.topk(min(W, NOP), 1)
    beam = kids.gather(1, topi.unsqueeze(-1).expand(-1, -1, D)); first = topi.clone(); bf = first[ar, topv.argmax(1)]
    for _ in range(L - 1):
        Wc = beam.shape[1]; kids, sc = expand(beam); fr = first.repeat_interleave(NOP, 1)
        topv, topi = sc.topk(min(W, Wc * NOP), 1)
        beam = kids.gather(1, topi.unsqueeze(-1).expand(-1, -1, D)); first = fr.gather(1, topi); bf = first[ar, topv.argmax(1)]
    return bf


@torch.no_grad()
def run_cov(wl, V, axis, beta, W=10, L=5, n=1500):
    s = init_states(n); ever = torch.zeros(n, dtype=torch.bool); out = {}; mem = wl.E(s)
    for tt in range(max(Bs)):
        op = plan_cov(wl, V, s, axis, W, L, mem, beta)
        s = apply_op(s, op); mem = 0.9 * mem + 0.1 * wl.E(s); ever |= reached(s, axis)
        if (tt + 1) in Bs: out[tt + 1] = ever.float().mean().item()
    return [out[b] for b in Bs]


def main():
    f = lambda *a: print(*a, flush=True)
    hdr = "    budget B:                  " + "  ".join(f"{b:>4d}" for b in Bs)
    f("=" * 92)
    f(f"Incubation step 14 — OUR-OWN-VALUE: Bellman + AIMED COVERAGE (beta sweep), held-out axis {HELD_OUT}")
    f("=" * 92)
    f("pretrain latent + Bellman value..."); wl = MS.pretrain_latent(); V = train_value_bellman(wl)
    f("")
    for axis, lab in [(HELD_OUT, f"HELD-OUT axis {HELD_OUT} = depth-3 (NOVEL) ZERO-SHOT"),
                      (1, "subset-ref axis 1 (depth-2)")]:
        f(f"  --- {lab} ---"); f(hdr)
        f("    random oracle (floor)     " + "  ".join(f"{v*100:4.0f}" for v in VS.oracle(axis, Bs)))
        for beta in (0.0, 1.0, 3.0, 8.0):
            tag = "Bellman (beta=0)" if beta == 0 else f"Bellman+cov b={beta:.0f}"
            f(f"    {tag:<22s}" + "  ".join(f"{v*100:4.0f}" for v in run_cov(wl, V, axis, beta)))
    f("\n" + "=" * 92)
    f("READ: if Bellman+coverage (beta>0) > pure Bellman (beta=0) on the held-out depth-3 axis, OUR aimed-coverage")
    f("value — crediting the OPENING of goal-relevant locked dimensions, not just goal-proximity — pushes the")
    f("creative ceiling past what a reachability value alone reaches. (beta too large should hurt: pure novelty wanders.)")


if __name__ == "__main__":
    main()
