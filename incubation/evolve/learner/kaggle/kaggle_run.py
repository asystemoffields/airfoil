#!/usr/bin/env python3
"""KAGGLE-1 entry — scale the expert-iteration loop over the rich relation distribution + the leave-one-FAMILY-out
GO gate. Runs headless on a T4 session. The box (CPU) validated every organ; this is the first scale spend.

UPLOAD: the learner source must be a Kaggle dataset 'USERNAME/airfoil-learner-src' containing the repo subtree
  incubation/evolve/learner/  + incubation/evolve/{harness.py,...}  + incubation/arc/{dsl.py,...}  + grammar files,
  mounted at /kaggle/input/airfoil-learner-src/. Set USERNAME in kernel-metadata.json + this file's SRC path.

WHAT THIS RUNS (KAGGLE-1, the falsifier -- NOT a full scale-up):
  Phase A (runnable NOW, recognizer-guided SEARCH loop): stream BARC ARC-Heavy + RE-ARC + ConceptARC, run
    kaggle_loop.loop at large N -> invention-certified solves + grown library + per-source coverage. Quantifies
    real-data coverage at scale with the CURRENT (recolor/select) effect set.
  Phase B (KAGGLE-1 build, TODO -- the GO bet): (1) GROW the EFFECT leaves (move/draw/fill/construct) from BARC
    concept-tags + verified solves (effects are EARNED, thin-core); (2) cold-start a LEARNED suffix policy
    (autoregressive over the DSL) from Phase-A's verified solves; (3) verifier-as-reward expert iteration.
  GO GATE: strict leave-one-FAMILY-out; GO to KAGGLE-2 (full loop) iff held-out-family beyond_gen6 >= 3.
"""
import sys, os, time

SRC = "/kaggle/input/airfoil-learner-src/incubation/evolve/learner"
for p in (SRC, SRC + "/..", SRC + "/../../arc"):
    if os.path.isdir(p):
        sys.path.insert(0, os.path.abspath(p))

t0 = time.time()
print("KAGGLE-1: airfoil relation-learner scale loop", flush=True)
try:
    import torch; print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), flush=True)
except Exception as e:
    print("torch import:", e, flush=True)

# ---- Phase A: scale the recognizer-guided expert-iteration loop over the rich distribution ----
# from kaggle_loop import loop
# from ground_barc import stream_barc          # streams BARC ARC-Heavy JSONL over https
# from ground_conceptarc import load_conceptarc
# N = 20000
# loop(stream_barc(N), "BARC ARC-Heavy")       # invention + library growth at scale
# loop(load_conceptarc(), "ConceptARC")
#
# ---- Phase B (KAGGLE-1 build): grow effect leaves + learned policy + expert iteration + the GO gate ----
# TODO(scale): effect-leaf growth, learned suffix policy cold-start, verifier-as-reward rounds,
#              leave-one-FAMILY-out, print GO iff held-out-family beyond_gen6 >= 3.

print(f"[{time.time()-t0:.0f}s] scaffold ready -- uncomment Phase A once the dataset is mounted; build Phase B.", flush=True)
