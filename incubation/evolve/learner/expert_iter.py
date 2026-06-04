#!/usr/bin/env python3
"""Vine — EXPERIENCE COMPOUNDS (expert-iteration layer 2a): route the accumulated library FIRST.

Once a concept is earned + verified, it joins the persistent library; the next similar task REUSES it (try the
library first) instead of re-searching. So a composed concept that cost a big search the first time is routed in
O(library) the next time -- experience compounds, on CPU, no training. (2b -- fine-tuning V2 on verified solves --
is the deeper layer.) Demo: solve a contained-in-largest task COLD (composition search), let the library keep it,
then a fresh one WARM (library reuse). Run: /data/llm/.venv/bin/python expert_iter.py"""
import os
import rel_dsl as D
from effect_faculty import verify_effect, Gesture
from open_loop import open_solve
from persist_library import Library, _earned
from grow_library import make_contained_in_largest_task, make_containment_task


def solve_with_library(train, test, lib):
    """try the ACCUMULATED library concepts first (cheap reuse), then fall back to the full open search."""
    n = 0
    for c in lib.concepts():
        n += 1
        if isinstance(c, Gesture):
            if verify_effect(c, train, test):
                return c, f"library({n})", n
        else:
            prog = D.induce_recolor(c, train)
            if prog is not None and D.verify(prog, train, test):
                return prog, f"library({n})", n
    prog, kind, cost = open_solve(train, test)
    if prog is not None:
        lib.add(_earned(prog))                       # GROW the library from the new solve
    return prog, kind, n + cost


def main():
    PATH = "/tmp/vine_expiter.json"
    if os.path.exists(PATH): os.remove(PATH)
    lib = Library(PATH)

    # warm the library up on a couple of simpler tasks first (so the composed one is genuinely deeper in the search)
    for _ in range(2):
        solve_with_library(make_containment_task(4), make_containment_task(2), lib)

    # COLD: first contained-in-largest -- composition search
    _, k_cold, cost_cold = solve_with_library(make_contained_in_largest_task(4), make_contained_in_largest_task(2), lib)
    # WARM: a fresh contained-in-largest -- the library now HAS the composition, routed first
    _, k_warm, cost_warm = solve_with_library(make_contained_in_largest_task(4), make_contained_in_largest_task(2), lib)

    print("EXPERIENCE COMPOUNDS (route the accumulated library first):")
    print(f"  contained-in-largest COLD (search):     cost {cost_cold:>3}  via {k_cold}")
    print(f"  contained-in-largest WARM (library):    cost {cost_warm:>3}  via {k_warm}")
    print(f"  library now holds {len(lib)} concepts: {[str(c) for c in lib.concepts()]}")
    print(f"  speedup from accumulated experience: {cost_cold/max(1,cost_warm):.0f}x")
    print("READ: a concept earned ONCE is reused on the next similar task at O(library) instead of re-searched = "
          "experience compounds, CPU-only, no training. Threading the persistent library as features (V2 routes it) "
          "+ fine-tuning V2 on verified solves (2b) are the deeper openness-to-experience layers.")


if __name__ == "__main__":
    main()
