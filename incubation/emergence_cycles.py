#!/usr/bin/env python3
"""Incubation step 15 — EMERGENCE-VIA-CYCLES (Alex's hypothesis): does held-out (never-trained depth-3) transfer
EMERGE after enough training cycles on the same weights — or is it flat (reactive can't) while only SEARCH transfers?

Track held-out axis-4 reached% as a function of TRAINING CYCLES for two substrates, both trained ONLY on subset
axes {0,1,2,3}:
  (a) REACTIVE net R (distilled from the search's subset solutions): more cycles -> does depth-3 transfer appear?
  (b) SEARCH + a bootstrapped VALUE trained for more cycles: does the value's structure-generality (and thus the
      search's novel-chain ceiling) rise with value cycles?
If (a) stays flat/low while (b) rises -> creativity needs SEARCH, not more reactive cycles (confirms s11/12). If (a)
suddenly rises at some cycle count -> emergence-via-cycles is real for reactive nets too (would be a big surprise).
Run with /data/llm/.venv/bin/python."""
import torch
import torch.nn as nn

import multiaxis_struct as MS
import value_search as VS
from learned_value import reach_ax
from distill_proposer import Recognizer, collect, run_R

torch.manual_seed(0)
apply_op, init_states, target_obs, reached = MS.apply_op, MS.init_states, MS.target_obs, MS.reached
NOP, D, H, HELD_OUT, SUBSET = MS.NOP, MS.D, MS.H, MS.HELD_OUT, MS.SUBSET
CYCLES = [250, 500, 1000, 2000, 4000, 8000]
EVB = (4, 8)


def reach_at(curve, B):
    return curve[list(EVB).index(B)]


@torch.no_grad()
def eval_R(wl, R, axis, n=1200):
    s = init_states(n); ever = torch.zeros(n, dtype=torch.bool); out = {}
    for tt in range(max(EVB)):
        op = R(wl.E(s), target_obs(s, axis)).argmax(-1); s = apply_op(s, op); ever |= reached(s, axis)
        if (tt + 1) in EVB: out[tt + 1] = ever.float().mean().item()
    return [out[b] for b in EVB]


@torch.no_grad()
def eval_search(wl, V, axis, n=800):
    s = init_states(n); ever = torch.zeros(n, dtype=torch.bool); out = {}
    for tt in range(max(EVB)):
        op = VS.plan_first_op(wl, V, s, axis, W=10, L=5); s = apply_op(s, op); ever |= reached(s, axis)
        if (tt + 1) in EVB: out[tt + 1] = ever.float().mean().item()
    return [out[b] for b in EVB]


def bellman_step(wl, V, opt, bce, gamma=0.9):
    n = 256; s = init_states(n)
    axis = torch.tensor(SUBSET)[torch.randint(0, len(SUBSET), (n,))]; t = target_obs(s, axis)
    with torch.no_grad():
        now = reach_ax(s, axis); best = torch.zeros(n)
        for op in range(NOP):
            child = apply_op(s, torch.full((n,), op))
            best = torch.maximum(best, torch.maximum(reach_ax(child, axis).float(),
                                                     torch.sigmoid(V(wl.E(child).unsqueeze(1), t).squeeze(1))))
        target = torch.where(now, torch.ones(n), gamma * best)
    loss = bce(V(wl.E(s).unsqueeze(1), t).squeeze(1), target)
    opt.zero_grad(); loss.backward(); opt.step()


def main():
    f = lambda *a: print(*a, flush=True)
    f("=" * 96); f(f"Incubation step 15 — EMERGENCE-VIA-CYCLES: held-out axis {HELD_OUT} transfer vs training cycles")
    f("=" * 96)
    f("pretrain latent..."); wl = MS.pretrain_latent()
    f("collecting search demos for the reactive net (subset axes)..."); E, G, Y = collect(wl, VS.train_value_togo(wl), episodes=30)
    f(f"  {E.shape[0]} demos\n")

    R = Recognizer(); optR = torch.optim.Adam(R.parameters(), lr=1e-3); ceR = nn.CrossEntropyLoss(); N = E.shape[0]
    V = VS.ValueNet(); optV = torch.optim.Adam(V.parameters(), lr=2e-3); bce = nn.BCEWithLogitsLoss()

    f(f"{'cycles':>7s} | REACTIVE-R  held-out(B4/B8)  subset(B4/B8) | SEARCH+VALUE held-out(B4/B8)  subset(B4/B8)")
    prev = 0
    for c in CYCLES:
        for _ in range(c - prev):
            idx = torch.randint(0, N, (256,)); l = ceR(R(E[idx], G[idx]), Y[idx]); optR.zero_grad(); l.backward(); optR.step()
            bellman_step(wl, V, optV, bce)
        prev = c
        rh = eval_R(wl, R, HELD_OUT); rs = eval_R(wl, R, 1)
        vh = eval_search(wl, V, HELD_OUT); vs_ = eval_search(wl, V, 1)
        g = lambda a: f"{a[0]*100:3.0f}/{a[1]*100:3.0f}"
        f(f"{c:>7d} |   reactive  {g(rh):>9s}     {g(rs):>9s}   |    search    {g(vh):>9s}     {g(vs_):>9s}")
    f("\n" + "=" * 96)
    f("READ: held-out columns over CYCLES. If REACTIVE-R held-out stays flat/low at every cycle count while")
    f("SEARCH+VALUE held-out RISES -> creativity needs SEARCH, not more reactive training cycles (confirms s11/12;")
    f("emergence-via-cycles does NOT rescue a reactive net). If reactive held-out suddenly climbs -> emergence is real.")


if __name__ == "__main__":
    main()
