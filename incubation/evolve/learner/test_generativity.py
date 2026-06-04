#!/usr/bin/env python3
"""GENERATIVITY GUARD — the runnable enforcement of SELF_EVOLVING_CHARTER.md (protects the concept of the cell fix).

POSITIVE (generativity): the SAME earn_cell_effect must earn HELD-OUT effects -- ones the box demo never used (the
full isometry group + a glide-reflection from the closure) -- with ZERO new code. If this FAILS, the basis lacks a
GENERATOR (add one, principled) or someone is about to hand-code (forbidden). NEGATIVE (treadmill detector): the
self-evolving module has ONE earn entry, searches the generator CLOSURE (not a per-family menu), and the production
solver routes through it. GREEN = the self-evolving substrate is protected. Run: /data/llm/.venv/bin/python test_generativity.py"""
import inspect
import numpy as np
import cell_evolve
from cell_evolve import earn_cell_effect

rng = np.random.RandomState(3)


def make_inv_task(n, mapfn, occ=9):
    """a grid INVARIANT under mapfn (any index-map), sparsely occluded on recoverable cells -> output = restored."""
    demos = []
    for _ in range(n):
        H = W = 10; base = np.zeros((H, W), int)
        for _k in range(20):
            r, c = rng.randint(0, H), rng.randint(0, W); col = int(rng.randint(1, 7))
            cur = (r, c); orbit = []
            for _ in range(6):
                if not (0 <= cur[0] < H and 0 <= cur[1] < W) or cur in orbit:
                    break
                orbit.append(cur); cur = mapfn(*cur, H, W)
            for (rr, cc) in orbit:
                base[rr, cc] = col
        corrupt = base.copy()
        cand = [(r, c) for (r, c) in zip(*np.nonzero(base))
                if mapfn(r, c, H, W) != (r, c)            # skip fixed points (map to themselves -> unrecoverable)
                and (lambda nr, nc: 0 <= nr < H and 0 <= nc < W and base[nr, nc] != 0)(*mapfn(r, c, H, W))]
        occluded = set()                                  # never occlude two cells of the SAME orbit (else unrecoverable)
        for i in rng.permutation(len(cand)):
            r, c = cand[i]; img = mapfn(r, c, H, W)
            if (r, c) in occluded or img in occluded:
                continue
            occluded.add((r, c)); corrupt[r, c] = occ
            if len(occluded) >= 3:
                break
        demos.append((corrupt, base))
    return demos


HELD_OUT = {   # effects the box demo NEVER used (it used mirror_h/v + periodic + fill); must earn with ZERO new code
    "rot180-symmetry":          lambda r, c, H, W: (H-1-r, W-1-c),
    "diagonal-symmetry":        lambda r, c, H, W: (c, r),
    "antidiagonal-symmetry":    lambda r, c, H, W: (W-1-c, H-1-r),
    "rot90-symmetry":           lambda r, c, H, W: (c, H-1-r),
    "glide-reflection(closure)": lambda r, c, H, W: (r, (W-1-c) + 2),
}


def positive():
    print("POSITIVE (generativity) — the SAME earn_cell_effect earns held-out effects with ZERO new code:")
    allok = True
    for name, mapfn in HELD_OUT.items():
        ok = 0
        for _ in range(8):
            if earn_cell_effect(make_inv_task(4, mapfn), make_inv_task(2, mapfn)) is not None:
                ok += 1
        passed = ok >= 6; allok = allok and passed
        print(f"  {name:<28}: {ok}/8 earned -> {'PASS' if passed else 'FAIL'}")
    return allok


def negative():
    print("\nNEGATIVE (treadmill detector):")
    earn_fns = [n for n, _ in inspect.getmembers(cell_evolve, inspect.isfunction) if n.startswith("earn_")]
    forbidden = [n for n in earn_fns if n != "earn_cell_effect"]
    src = inspect.getsource(cell_evolve)
    closure = "INV = generative_maps()" in src
    try:
        solver_ok = "earn_cell_effect" in inspect.getsource(__import__("ground_vine_evolve"))
    except Exception:
        solver_ok = False
    p1 = not forbidden; p2 = closure
    print(f"  one earn entry (no per-family earners): {earn_fns} -> {'PASS' if p1 else f'FAIL {forbidden}'}")
    print(f"  earn searches the generator CLOSURE (not a hand-listed menu): {'PASS' if p2 else 'FAIL'}")
    print(f"  production solver routes through earn_cell_effect: {'PASS' if solver_ok else 'FAIL'}")
    return p1 and p2 and solver_ok


if __name__ == "__main__":
    p = positive(); n = negative()
    print(f"\nGENERATIVITY GUARD: {'GREEN — the self-evolving substrate is protected' if (p and n) else 'RED — treadmill risk; see SELF_EVOLVING_CHARTER.md'}")
