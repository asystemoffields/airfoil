#!/usr/bin/env python3
"""Incubation step 13 — LEARN A BETTER VALUE (the one lever on the irreducible creative engine).
Steps 11-12: novelty needs SEARCH at inference (can't be amortized into params). So to lift the novel-chain
ceiling we improve the SEARCH's VALUE. Current V_togo is a 1-shot Monte-Carlo estimate (P(random k-rollout
reaches target)) — noisy, and a random rollout rarely executes a depth-3 chain so it under-credits deep setups.
A BOOTSTRAPPED Bellman value does value-iteration over the frozen model:
    V(s,t) <- reached(s) ? 1 : gamma * max_op [ reached(model(s,op)) ? 1 : V(model(s,op), t) ]
This propagates reachability BACKWARD through the chain (credits P4->T4->C4 setups sharply), trained on SUBSET
axes only (structure-general). Question: does beam search guided by V_bellman beat search guided by V_togo on
the held-out DEPTH-3 axis? (= can a better LEARNED value raise the creative ceiling, no extra inference cost?)
Run with /data/llm/.venv/bin/python."""
import torch
import torch.nn as nn

import multiaxis_struct as MS
import value_search as VS

torch.manual_seed(0)
apply_op, init_states, target_obs, reached = MS.apply_op, MS.init_states, MS.target_obs, MS.reached
NOP, D, H, HELD_OUT, SUBSET = MS.NOP, MS.D, MS.H, MS.HELD_OUT, MS.SUBSET
Bs = (2, 4, 6, 8, 10)


def reach_ax(s, axis_vec):
    n = s.shape[0]; return torch.cos(s[torch.arange(n), axis_vec]) < 0.0


def train_value_bellman(wl, gamma=0.9, steps=4000):
    """value iteration over the frozen PERFECT model, SUBSET axes, per-example axis. BCE to a bootstrapped target."""
    V = ValueNet = VS.ValueNet(); opt = torch.optim.Adam(V.parameters(), lr=2e-3); bce = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        n = 256; s = init_states(n)
        axis = torch.tensor(SUBSET)[torch.randint(0, len(SUBSET), (n,))]; t = target_obs(s, axis)
        with torch.no_grad():
            now = reach_ax(s, axis)
            best_next = torch.zeros(n)
            for op in range(NOP):                          # 1-step lookahead over all ops
                child = apply_op(s, torch.full((n,), op))
                rc = reach_ax(child, axis).float()
                vc = torch.sigmoid(V(wl.E(child).unsqueeze(1), t).squeeze(1))
                cand = torch.maximum(rc, vc)               # reached child = 1 else bootstrapped value
                best_next = torch.maximum(best_next, cand)
            target = torch.where(now, torch.ones(n), gamma * best_next)
        loss = bce(V(wl.E(s).unsqueeze(1), t).squeeze(1), target)
        opt.zero_grad(); loss.backward(); opt.step()
    return V


def main():
    f = lambda *a: print(*a, flush=True)
    hdr = "    budget B:                    " + "  ".join(f"{b:>4d}" for b in Bs)
    f("=" * 92)
    f(f"Incubation step 13 — LEARNED Bellman value vs Monte-Carlo V_togo, search on held-out axis {HELD_OUT}")
    f("=" * 92)
    f("pretrain latent..."); wl = MS.pretrain_latent()
    f("train V_togo (Monte-Carlo, k-rollout)..."); Vmc = VS.train_value_togo(wl)
    f("train V_bellman (bootstrapped value iteration)..."); Vbe = train_value_bellman(wl)
    f("")
    for axis, lab in [(HELD_OUT, f"HELD-OUT axis {HELD_OUT} = depth-3 (NOVEL) ZERO-SHOT"),
                      (1, "subset-ref axis 1 (depth-2)")]:
        f(f"  --- {lab} ---"); f(hdr)
        f("    random oracle (floor)       " + "  ".join(f"{v*100:4.0f}" for v in VS.oracle(axis, Bs)))
        f("    search + V_togo  (Monte-Carlo)" + "  ".join(f"{v*100:4.0f}" for v in VS.run_search(wl, Vmc, axis, Bs, W=10)))
        f("    search + V_bellman (bootstrap)" + "  ".join(f"{v*100:4.0f}" for v in VS.run_search(wl, Vbe, axis, Bs, W=10)))
    f("\n" + "=" * 92)
    f("READ: if search+V_bellman > search+V_togo on the held-out depth-3 axis, a better LEARNED value raises the")
    f("creative ceiling at NO extra inference cost — the value is the lever on the irreducible search engine.")
    f("If equal, the Monte-Carlo value already saturates what the beam can use here.")


if __name__ == "__main__":
    main()
