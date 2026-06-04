#!/usr/bin/env python3
"""GATE VALIDATION SHIM (not an inventor). Wraps gen1_05_param-struct (whole-template RETRIEVAL) and
adds the documented gate hooks so the invention gate can be validated:
  * solve_ablated == solve  -> ablation is identical to the full solver, so by construction every full
    solve is reproduced under ablation => INVENTED must be ~0 (retrieval is NOT invention).
  * reset_library()         -> clears the wrapped solver's in-session experience library (_LIB).
This file exists only to prove the gate's certification logic on a known-retrieval solver."""
import os
import sys
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
_path = os.path.join(HERE, "gen1_05_param-struct.py")
_spec = importlib.util.spec_from_file_location("gen1_05_param_struct", _path)
_ps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ps)

META = {"name": "param_struct_gate_test",
        "desc": "RETRIEVAL solver + trivial solve_ablated==solve + reset_library (gate validation)"}


def solve(train, test_inputs, budget):
    return _ps.solve(train, test_inputs, budget)


def solve_ablated(train, test_inputs, budget):
    # Invention 'disabled' is trivially the same as enabled here: retrieval has no invention to remove.
    return _ps.solve(train, test_inputs, budget)


def reset_library():
    _ps._LIB["concept_hits"] = {}
