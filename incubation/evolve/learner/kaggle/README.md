# KAGGLE-1 handoff — airfoil relation-learner scale loop

The box (CPU) validated every organ of the factored proposer (see `../LEARNER.md`). This is the first GPU spend:
one ≤8h T4×2 expert-iteration round, **GO to KAGGLE-2 iff held-out-family `beyond_gen6` ≥ 3**.

**Two lanes — no need to wait for the Kaggle quota:**
- **Colab (now):** open `colab_run.ipynb`, set a GitHub read token, Run all. Phase A is CPU-bound (runs on any
  runtime); use a T4 runtime for Phase B. Final weights (`learner_v2h.pt`, `learner_v3_geo.pt`) are committed so the
  clone is self-contained.
- **Kaggle (quota ~Jun 7-8):** the script kernel below, for longer headless T4×2 sessions.

## One-time setup
1. Set `USERNAME` in `kernel-metadata.json` (id + dataset_sources) and in `kaggle_run.py` (SRC path is fine as-is).
2. Package the source as a Kaggle dataset `USERNAME/airfoil-learner-src` (mounted at `/kaggle/input/...`):
   - `incubation/evolve/learner/` (all `.py` + `learner_v2h.pt`, `learner_v3_geo.pt`)
   - `incubation/evolve/harness.py` + the ARC data refs, `incubation/arc/dsl.py`, and `grammar*.py`
3. Auth: token at `~/.kaggle/kaggle.json`; CLI at `/data/kagglecli-venv/bin/kaggle` (cb-clone mints the Legacy API key if needed). Limits: 1 concurrent GPU, 30h/wk, 12h/session, `/kaggle/working` 19.5GB.

## Push / poll / pull (headless)
```
kaggle datasets create -p <src_dir> -r zip          # first time; later: datasets version
kaggle kernels push -p incubation/evolve/learner/kaggle
kaggle kernels status USERNAME/airfoil-rellearner-kaggle1
kaggle kernels output USERNAME/airfoil-rellearner-kaggle1 -p ./k1-out
```
Verify the OUTPUT artifact (the GO-gate number), not just `COMPLETE`.

## What KAGGLE-1 runs
- **Phase A (runnable now):** scale the recognizer-guided loop (`kaggle_loop.loop`) over BARC ARC-Heavy stream +
  RE-ARC + ConceptARC → invention-certified solves + grown library + per-source coverage at scale.
- **Phase B (the GO bet, build on Kaggle):** grow the EFFECT leaves (move/draw/fill/construct — *earned*, thin-core)
  + cold-start the learned suffix policy + verifier-as-reward expert iteration + the leave-one-FAMILY-out GO gate.

Free-compute only. The box result is the floor; do not regress it.
