#!/usr/bin/env python3
"""
Incubation step 10b — multi-seed confirmation of the ONE unconfirmed step-10 claim: the NON-MONOTONIC
frontier (rho=0.66 beat rho=1.0 on 1 seed; conjecture = a little observation noise REGULARIZES the step-8
model-exploitation that a full-fidelity learned model suffers). Pessimism-always-hurts and the realism
threshold are already robust by mechanism + consistent across the rho sweep — not re-litigated here.

Isolate rho in {0.66, 1.0} + the perfect/full reference, over seeds {1,2,3}. If ens-mean@rho0.66 > {single
@rho1.0, perfect} holds across seeds, the non-monotonicity (intermediate fidelity > full) is real. Reuses
the step-10 graded-obs machinery. Run with /data/llm/.venv/bin/python.
"""
import torch

import hardened_search as HS
import partial_obs_gate as PO
import realism_frontier as RF

HELD_OUT, NOP = HS.HELD_OUT, HS.NOP
SEEDS = (1, 2, 3)
RHOS = (0.66, 1.0)
Bs = RF.Bs
M_ENS = RF.M_ENS


def main():
    f = lambda *a: print(*a, flush=True)
    hdr = "      budget B:                       " + "  ".join(f"{b:>4d}" for b in Bs)
    f("=" * 98)
    f(f"Incubation step 10b — multi-seed confirmation of NON-MONOTONIC frontier (rho 0.66 vs 1.0); axis {HELD_OUT}")
    f(f"   seeds={SEEDS}, ensemble M={M_ENS}")
    f("=" * 98)

    rows = {}  # name -> list of per-seed [B...]
    def rec(name, v):
        rows.setdefault(name, []).append(v)

    for sd in SEEDS:
        torch.manual_seed(sd); f(f"\n----- seed {sd} -----"); f(hdr)
        wl_full = HS.pretrain_latent(); V_full = HS.train_value_togo(wl_full, k=6)
        per = RF.run(lambda s, ax: PO.beam_single(wl_full, V_full, s, ax, 10, 5, PO.trans_true), HELD_OUT)
        rec("perfect/full beam", per); f("      perfect/full beam            " + "  ".join(f"{x*100:4.0f}" for x in per))
        for rho in RHOS:
            ens = [RF.pretrain_rho(rho) for _ in range(M_ENS)]
            for wl in ens:
                wl.fwd_step = (lambda WL: (lambda flat, j: WL.fwd(torch.cat([WL.E(flat), HS.oh(torch.full((flat.shape[0],), j), NOP)], -1)) + flat))(wl)
            V_po = PO.train_value_po(ens[0], k=6); tr = PO.make_trans_po(ens[0])
            sg = RF.run(lambda s, ax: PO.beam_single(ens[0], V_po, s, ax, 10, 5, tr), HELD_OUT)
            em = RF.run(lambda s, ax: PO.beam_ensemble(ens, V_po, s, ax, 10, 5, 0.0), HELD_OUT)
            rec(f"rho{rho:.2f} single", sg); rec(f"rho{rho:.2f} ens-mean", em)
            f(f"      rho={rho:.2f} single            " + "  ".join(f"{x*100:4.0f}" for x in sg))
            f(f"      rho={rho:.2f} ens-mean          " + "  ".join(f"{x*100:4.0f}" for x in em))

    f("\n" + "=" * 98)
    f(f"SUMMARY — held-out depth-3, mean[min..max] over {len(SEEDS)} seeds (% reached)"); f(hdr)
    for name in ["perfect/full beam", "rho0.66 single", "rho0.66 ens-mean", "rho1.00 single", "rho1.00 ens-mean"]:
        r = rows[name]
        cells = [f"{sum(x[j]*100 for x in r)/len(r):3.0f}[{min(x[j]*100 for x in r):3.0f}..{max(x[j]*100 for x in r):3.0f}]" for j in range(len(Bs))]
        f(f"      {name:<26s}" + "  ".join(cells))
    f("\n  CONFIRMED if rho0.66 (esp ens-mean) >= rho1.00 AND >= perfect across seeds -> intermediate fidelity")
    f("  beats full fidelity: a little obs noise regularizes model-exploitation. REFUTED if they overlap/flip.")


if __name__ == "__main__":
    main()
