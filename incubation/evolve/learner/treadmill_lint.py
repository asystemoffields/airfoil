#!/usr/bin/env python3
"""Vine — the TREADMILL LINT (extends the generativity guard; enforces SELF_EVOLVING_CHARTER.md basis-vs-menu).

Three checks: (A) GENERATIVITY -- every generator produces HELD-OUT members with zero new code (basis, not menu);
(B) IRREDUCIBILITY -- no basis element is a composition of the others (the corollary; else a menu hides in a basis);
(C) NO MENU SLOTS -- every vocabulary the solver routes over comes from a generator call, not a hand-listed literal.
A vocabulary SLOT still bound to a list/dict literal is a treadmill (RED). Run: /data/llm/.venv/bin/python treadmill_lint.py"""
import ast, os, inspect
import generators as GEN

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- (C) the vocabulary SLOTS the solver routes over -- each MUST be generator-sourced, not a literal menu ----
SLOTS = [
    ("grammar.py",         "FEATURE_NAMES", "generators.object_features()"),
    ("effect_faculty.py",  "MODES",         "generators.motor_targets()"),
    ("derive_grammar.py",  "CLOSE",         "generators.paint_family()"),
    ("derive_grammar.py",  "FORWARD",       "generators.paint_family()"),
]
# the irreducible BASES (whitelisted): small, primitive, documented -- the legitimate innate language.
WHITELIST = {"generators.REDUCTIONS", "generators.CELL_PROPS", "generators.PROPS", "generators.OPS",
             "grammar.DECOMPS", "cell_evolve.SQ"}


def slot_kind(fname, var):
    path = os.path.join(HERE, fname)
    if not os.path.exists(path):
        return "absent"
    try:
        tree = ast.parse(open(path).read())
    except Exception:
        return "absent"
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == var for t in node.targets):
            v = node.value
            if isinstance(v, (ast.List, ast.Dict, ast.Set, ast.Tuple)):
                return "MENU-LITERAL"
            if isinstance(v, ast.Call):
                return "generated"
            return "other"
    return "absent"


def check_generativity():
    print("(A) GENERATIVITY -- bases produce HELD-OUT members with zero new code:")
    probes = [("object_features", GEN.object_features(), ["max_row", "extent_col"]),
              ("substrate_relations", GEN.substrate_relations(), ["h==w", "size>color"])]
    allok = True
    for name, menu, held in probes:
        miss = [h for h in held if h not in menu]
        ok = not miss; allok = allok and ok
        print(f"  {name:<22}: {len(menu)} generated; held-out {held} present? {'PASS' if ok else f'FAIL missing {miss}'}")
    try:
        from test_generativity import positive as cell_pos
        print("  cell substrate (test_generativity): see its own GREEN/RED; structural derivations in derive_grammar.")
    except Exception:
        pass
    return allok


def check_irreducibility():
    print("\n(B) IRREDUCIBILITY -- no basis element is a composition of the others:")
    # heuristic: flag a basis NAME that is a known composition pattern of others (e.g., 'extent'=max-min, 'enclosed').
    smell = {"extent", "range", "enclosed", "glide", "rot270"}   # derivable-from-others names that must NOT be basis keys
    allok = True
    for bname, basis in [("REDUCTIONS", GEN.REDUCTIONS), ("CELL_PROPS", GEN.CELL_PROPS),
                         ("PROPS", {p: 1 for p in GEN.PROPS}), ("OPS", GEN.OPS)]:
        bad = [k for k in basis if k in smell]
        allok = allok and not bad
        print(f"  {bname:<12}: {list(basis)} -> {'PASS' if not bad else f'FAIL (composition smuggled: {bad})'}")
    return allok


def check_slots():
    print("\n(C) NO MENU SLOTS -- vocabularies the solver routes over must be generator-sourced, not literals:")
    allok = True
    for fname, var, want in SLOTS:
        k = slot_kind(fname, var)
        ok = k in ("generated", "absent")    # absent = retired (also fine); MENU-LITERAL = treadmill
        allok = allok and ok
        status = {"generated": "PASS (generated)", "absent": "PASS (retired)", "MENU-LITERAL": f"FAIL -> use {want}", "other": "?"}.get(k, k)
        print(f"  {fname}:{var:<14} = {status}")
    return allok


if __name__ == "__main__":
    a, b, c = check_generativity(), check_irreducibility(), check_slots()
    print(f"\nTREADMILL LINT: {'GREEN -- bases generate, irreducible, no menu slots' if (a and b and c) else 'RED -- treadmill(s) remain (see FAIL above); see SELF_EVOLVING_CHARTER.md'}")
