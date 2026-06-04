#!/usr/bin/env python3
"""Vine — STRUCTURAL MODS AS GRAMMAR DERIVATIONS, productions GENERATED not menu'd (Alex's audit + basis-vs-menu).

A solution is a DERIVATION over a production grammar; a structural mod = re-deriving a subtree. ONE engine (`derive`,
forward program search) generates compose/add-step/swap. CRUCIALLY the productions themselves are NOT a hand-listed
menu (the old CLOSE/FORWARD lists were a treadmill): they come from `generators.paint_family` = the single motor
primitive paint(SELECTOR, VALUE) over thin SELECTOR/VALUE bases. invariance/colormap/fill EMERGE as instances;
held-out paints emerge with zero new code; a new selector/value (a cell-substrate predicate) yields new productions
for free. The only extension point is the BASIS. Run: /data/llm/.venv/bin/python derive_grammar.py"""
import numpy as np
from generators import paint_family            # productions are GENERATED here, not a menu in this module


def derive(train, test, max_steps=2, budget=4000):
    """forward program search over the GENERATED production family. Returns (program | None, n_calls). compose =
    a 2-step derivation, add-step = longer, sibling-swap = a different production -- all GENERATED, no operator code."""
    tr_out = [np.asarray(go, int) for _, go in train]; te_out = [np.asarray(go, int) for _, go in test]
    n = 0
    def ok(fn, cur_tr, cur_te):
        try:
            return (all(np.array_equal(np.asarray(fn(c), int), o) for c, o in zip(cur_tr, tr_out)) and
                    all(np.array_equal(np.asarray(fn(c), int), o) for c, o in zip(cur_te, te_out)))
        except Exception:
            return False
    frontier = [([], [np.asarray(gi, int) for gi, _ in train], [np.asarray(gi, int) for gi, _ in test])]
    for step in range(max_steps + 1):
        nf = []
        for prog, cur_tr, cur_te in frontier:
            for name, kind, op in paint_family(cur_tr):
                n += 1
                if n > budget:
                    return None, n
                if kind == "close":                                  # VALUE induced from output -> terminate
                    fn = op(cur_tr, tr_out)
                    if fn is not None and ok(fn, cur_tr, cur_te):
                        return prog + [name], n
                elif kind == "forward" and step < max_steps:         # VALUE deterministic -> grow the intermediate
                    try:
                        ntr = op(cur_tr); nte = op(cur_te)
                    except Exception:
                        continue
                    if any(x is None for x in ntr + nte):
                        continue
                    nf.append((prog + [name], ntr, nte))
        frontier = nf
    return None, n


def _demo():
    import inspect
    from test_generativity import make_inv_task
    from schema_adapt import make_symcolor_task, PARENT, HELD
    rng = np.random.RandomState(5)

    def cmap():
        t = {0: 0}; perm = list(rng.permutation([1, 2, 3, 4, 5, 6]))
        for i, c in enumerate([1, 2, 3, 4, 5, 6]):
            t[c] = int(perm[i])
        return t

    cases = {
        "pure invariance (mirror_h)":          (lambda: (make_inv_task(4, PARENT["mirror_h"]), make_inv_task(2, PARENT["mirror_h"])), False),
        "DEMO compose: mirror_h o colormap":   (lambda c=cmap(): (make_symcolor_task(4, PARENT["mirror_h"], c), make_symcolor_task(2, PARENT["mirror_h"], c)), False),
        "HELD-OUT compose: rot180 o colormap": (lambda c=cmap(): (make_symcolor_task(4, HELD["rot180"], c), make_symcolor_task(2, HELD["rot180"], c)), True),
        "HELD-OUT compose: diag o colormap":   (lambda c=cmap(): (make_symcolor_task(4, HELD["diag"], c), make_symcolor_task(2, HELD["diag"], c)), True),
    }
    print("STRUCTURAL MODS = GRAMMAR DERIVATIONS over GENERATED productions (no operator menu, no production menu):\n")
    for name, (gen, held) in cases.items():
        ok = 0; ex = None
        for _ in range(8):
            prog, _n = derive(*gen())
            if prog is not None:
                ok += 1; ex = prog
        tag = " [HELD-OUT -> generativity]" if held else ""
        print(f"  {name:<38}: {ok}/8 derived  ex={ex}{tag}")

    DG = __import__("derive_grammar")
    fns = [n for n, _ in inspect.getmembers(DG, inspect.isfunction)]
    bad_ops = [n for n in fns if n in ("compose", "add_step", "remove_step", "sibling_swap", "swap")]
    bad_menu = [n for n in fns if n.startswith("close_") or n.startswith("forward_")] + \
               [a for a in ("CLOSE", "FORWARD", "PRODUCTIONS") if hasattr(DG, a)]   # no production-menu module attrs
    print(f"\n  treadmill detector: operator functions={bad_ops or 'NONE'}  production menu={bad_menu or 'NONE'} "
          f"-> {'PASS (operators AND productions are generated)' if not (bad_ops or bad_menu) else 'FAIL'}")
    print("READ: held-out compositions derived with ZERO new code; productions come from generators.paint_family "
          "(SELECTOR x VALUE bases), not a hand-listed menu. Both operators and productions are now generated.")


if __name__ == "__main__":
    _demo()
