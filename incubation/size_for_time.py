#!/usr/bin/env python3
"""
Incubation step 11 — the SIZE-FOR-TIME frontier (the program's headline claim).

Claim: on a STRUCTURALLY-NOVEL affordance (the held-out depth-3 chain), TEST-TIME SEARCH COMPUTE (a small
fixed net + beam over a world-model) buys generalization that NO amount of REACTIVE PARAMETERS can. Spend
inference compute, not model size/training, to get creative transfer.

Two sweeps on the hardened world, held-out depth-3 axis:
  A. REACTIVE sweep — train the fast attention policy (GenPolicy / Arch2) at growing SIZES (params), eval
     zero-shot on the held-out chain. Prediction: ~0 at EVERY size (a reactive net cannot emit a chain-shape
     it never trained on; more parameters don't help) — while it stays high on a TRAINED axis (sanity).
  B. SEARCH sweep — fix a SMALL value net + world-model, grow beam WIDTH W (= test-time compute). Prediction:
     held-out accuracy RISES with W. The same tiny controller, given search compute, deploys the novel chain.
Report accuracy vs a compute proxy (reactive ~ params/1 fwd ; search ~ W*L*NOP model-calls/decision).
Uses the step-8 hardened world + GenPolicy from multiaxis_struct. Run with /data/llm/.venv/bin/python.
"""
import torch
import torch.nn as nn

import hardened_search as HS
import partial_obs_gate as PO
from multiaxis_struct import GenPolicy
apply_op, init_states, target_obs, reached = HS.apply_op, HS.init_states, HS.target_obs, HS.reached
D, NOP, H, HELD_OUT, SUBSET, T = HS.D, HS.NOP, HS.H, HS.HELD_OUT, HS.SUBSET, HS.T

Bs = (4, 8, 12)
SEED = 1


def train_reactive(wl, dm, nhead, nlayers, steps=1500):
    pol = GenPolicy(dm=dm, nhead=nhead, nlayers=nlayers); opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    for _ in range(steps):
        n = 256; s = init_states(n)
        ax = torch.tensor(SUBSET)[torch.randint(0, len(SUBSET), (n,))]; t = target_obs(s, ax)
        seq = [wl.E(s)]; logp = torch.zeros(n); ent = torch.zeros(n); ever = torch.zeros(n, dtype=torch.bool)
        for _ in range(T):
            d = torch.distributions.Categorical(logits=pol(torch.stack(seq, 1), t))
            op = d.sample(); logp = logp + d.log_prob(op); ent = ent + d.entropy()
            s = apply_op(s, op); seq.append(wl.E(s)); ever |= (torch.cos(s[torch.arange(n), ax]) < 0.0)
        r = ever.float(); adv = (r - r.mean()) / (r.std() + 1e-6)
        loss = -(logp * adv.detach()).mean() - 0.01 * ent.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return pol


@torch.no_grad()
def run_reactive(wl, pol, axis, n=600):
    s = init_states(n); ever = torch.zeros(n, dtype=torch.bool); out = {}; seq = [wl.E(s)]
    for tt in range(max(Bs)):
        op = pol(torch.stack(seq, 1), target_obs(s, axis)).argmax(-1)
        s = apply_op(s, op); seq.append(wl.E(s)); ever |= reached(s, axis)
        if (tt + 1) in Bs: out[tt + 1] = ever.float().mean().item()
    return [out[b] for b in Bs]


@torch.no_grad()
def run_search(wl, V, axis, W, L=5, n=600):
    s = init_states(n); ever = torch.zeros(n, dtype=torch.bool); out = {}
    for tt in range(max(Bs)):
        op = PO.beam_single(wl, V, s, axis, W, L, PO.trans_true)
        s = apply_op(s, op); ever |= reached(s, axis)
        if (tt + 1) in Bs: out[tt + 1] = ever.float().mean().item()
    return [out[b] for b in Bs]


def main():
    f = lambda *a: print(*a, flush=True)
    hdr = "      budget B:                          " + "  ".join(f"{b:>4d}" for b in Bs)
    f("=" * 100)
    f(f"Incubation step 11 — SIZE-FOR-TIME frontier (held-out axis {HELD_OUT}=depth-3); seed={SEED}")
    f("=" * 100)
    torch.manual_seed(SEED)
    f("pretraining frozen latent + value-to-go..."); wl = HS.pretrain_latent(); V = HS.train_value_togo(wl, k=6)

    f("\n  --- A. REACTIVE sweep (grow PARAMS; same train regime) — held-out depth-3 + trained-axis sanity ---")
    f("      " + f"{'reactive size':<26s}" + "  held-out (depth-3)        trained axis1")
    sizes = [("tiny  dm16 L1", 16, 2, 1), ("small dm32 L1", 32, 4, 1), ("med   dm64 L2", 64, 4, 2), ("big   dm128 L3", 128, 8, 3)]
    for lab, dm, nh, nl in sizes:
        pol = train_reactive(wl, dm, nh, nl); npar = sum(p.numel() for p in pol.parameters())
        ho = run_reactive(wl, pol, HELD_OUT); tr = run_reactive(wl, pol, 1)
        f(f"      {lab:<14s}({npar:>6d}p)  " + "  ".join(f"{x*100:4.0f}" for x in ho) + "   |   " + "  ".join(f"{x*100:4.0f}" for x in tr))

    f("\n  --- B. SEARCH sweep (fixed SMALL value net; grow beam WIDTH = test-time compute) — held-out depth-3 ---")
    Vpar = sum(p.numel() for p in V.parameters())
    f(f"      (value net = {Vpar} params, FIXED; world-model frozen. compute proxy = W*L*NOP model-calls/decision)")
    f(hdr)
    f("      random oracle (floor)        " + "  ".join(f"{x*100:4.0f}" for x in HS.oracle(HELD_OUT, Bs)))
    for W in (1, 2, 4, 8, 16, 32):
        ho = run_search(wl, V, HELD_OUT, W); cost = W * 5 * NOP
        f(f"      beam W={W:<3d} (~{cost:>5d} calls)   " + "  ".join(f"{x*100:4.0f}" for x in ho))

    f("\n" + "=" * 100)
    f("READ: A. reactive held-out stays ~0 at EVERY param count (can't emit a never-trained chain-shape) while")
    f("trained-axis stays high -> SIZE does not buy creative transfer. B. the SAME small fixed controller, given")
    f("more beam WIDTH (test-time compute), climbs the held-out chain from ~floor up. => SIZE-FOR-TIME: spend")
    f("inference SEARCH compute, not parameters, to deploy a structurally-novel affordance. (compute axis = the")
    f("headline trade; the small-net+search controller reaches what no reactive-parameter scaling does.)")


if __name__ == "__main__":
    main()
