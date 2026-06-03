#!/usr/bin/env python3
"""Incubation step 17 — HYBRID-C END-TO-END: the deployable artifact for the 7GB target.
Operationalizes the s12 deployment law (amortize the routine, SEARCH the novel) in ONE progress-gated controller:
run the DISTILLED FAST reactive net R by default; fall back to value-guided SEARCH only when the target coordinate
STOPS IMPROVING (the env has stalled -> likely a novel chain the fast net can't emit). Measures both reached% AND
the SEARCH-INVOCATION RATE (fraction of steps that needed the expensive slow path) — the deployment win is being
reactive-cheap on the KNOWN (rarely searches) and creative on the NOVEL (searches only there).
Compare: R-only (fast, fails novel) ; search-only (solves novel, always expensive) ; HYBRID (gated).
Run with /data/llm/.venv/bin/python."""
import torch
import torch.nn as nn

import multiaxis_struct as MS
import value_search as VS
from learned_value import train_value_bellman
from distill_proposer import Recognizer, collect

torch.manual_seed(0)
apply_op, init_states, target_obs, reached = MS.apply_op, MS.init_states, MS.target_obs, MS.reached
NOP, D, H, HELD_OUT, SUBSET = MS.NOP, MS.D, MS.H, MS.HELD_OUT, MS.SUBSET
Bs = (2, 4, 6, 8, 10)


def train_R(E, G, Y, steps=4000):
    R = Recognizer(); opt = torch.optim.Adam(R.parameters(), lr=1e-3); ce = nn.CrossEntropyLoss(); N = E.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, N, (256,)); loss = ce(R(E[idx], G[idx]), Y[idx]); opt.zero_grad(); loss.backward(); opt.step()
    return R


@torch.no_grad()
def run_R(wl, R, axis, n=1200):
    s = init_states(n); ever = torch.zeros(n, dtype=torch.bool); out = {}
    for tt in range(max(Bs)):
        op = R(wl.E(s), target_obs(s, axis)).argmax(-1); s = apply_op(s, op); ever |= reached(s, axis)
        if (tt + 1) in Bs: out[tt + 1] = ever.float().mean().item()
    return [out[b] for b in Bs], 0.0


@torch.no_grad()
def run_search(wl, V, axis, n=1200):
    s = init_states(n); ever = torch.zeros(n, dtype=torch.bool); out = {}
    for tt in range(max(Bs)):
        op = VS.plan_first_op(wl, V, s, axis, W=10, L=5); s = apply_op(s, op); ever |= reached(s, axis)
        if (tt + 1) in Bs: out[tt + 1] = ever.float().mean().item()
    return [out[b] for b in Bs], 1.0


@torch.no_grad()
def run_hybrid(wl, R, V, axis, n=1200, stuck_max=2):
    s = init_states(n); ar = torch.arange(n); ever = torch.zeros(n, dtype=torch.bool); out = {}
    mode_fast = torch.ones(n, dtype=torch.bool); stuck = torch.zeros(n); prev = torch.cos(s[:, axis]); searches = 0; steps = 0
    for tt in range(max(Bs)):
        fop = R(wl.E(s), target_obs(s, axis)).argmax(-1)
        sop = VS.plan_first_op(wl, V, s, axis, W=10, L=5)      # slow path (computed for all; only stalled envs USE it)
        use_search = ~mode_fast
        op = torch.where(mode_fast, fop, sop)
        searches += int(use_search.sum()); steps += n
        s = apply_op(s, op); cur = torch.cos(s[:, axis]); prog = cur < prev - 1e-3
        stuck = torch.where(mode_fast & ~prog, stuck + 1, torch.where(mode_fast, torch.zeros(n), stuck))
        mode_fast = mode_fast & (stuck < stuck_max); prev = cur; ever |= reached(s, axis)
        if (tt + 1) in Bs: out[tt + 1] = ever.float().mean().item()
    return [out[b] for b in Bs], searches / steps


def main():
    f = lambda *a: print(*a, flush=True)
    hdr = "    budget B:                " + "  ".join(f"{b:>4d}" for b in Bs) + "   search-rate"
    f("=" * 96); f(f"Incubation step 17 — HYBRID-C END-TO-END (distilled fast + gated search slow); held-out axis {HELD_OUT}")
    f("=" * 96)
    f("pretrain latent + V_togo + Bellman value..."); wl = MS.pretrain_latent(); Vmc = VS.train_value_togo(wl); Vbe = train_value_bellman(wl)
    f("collect search demos + train distilled R..."); E, G, Y = collect(wl, Vmc, episodes=30); R = train_R(E, G, Y)
    f("")
    for axis, lab in [(HELD_OUT, f"HELD-OUT axis {HELD_OUT} = depth-3 (NOVEL)"),
                      (1, "subset-ref axis 1 (depth-2, KNOWN)"), (0, "free axis 0 (KNOWN)")]:
        f(f"  --- {lab} ---"); f(hdr)
        for name, fn in (("R-only (fast)", lambda a: run_R(wl, R, a)),
                         ("search-only (slow)", lambda a: run_search(wl, Vbe, a)),
                         ("HYBRID-C (gated)", lambda a: run_hybrid(wl, R, Vbe, a))):
            vals, rate = fn(axis)
            f(f"    {name:<20s}" + "  ".join(f"{v*100:4.0f}" for v in vals) + f"     {rate*100:4.0f}%")
    f("\n" + "=" * 96)
    f("READ: HYBRID-C should match R-only on the KNOWN axes (high, ~0% search-rate = cheap) AND approach search-only")
    f("on the NOVEL axis (high, search-rate climbs only there). = the deployment law realized: reactive-cheap on the")
    f("routine, search the novel, automatically gated by progress. The single controller for the 7GB target.")


if __name__ == "__main__":
    main()
