# Inventory — everything added to the machine during the autonomous run

Keep this honest so nothing is a mystery later. Format: what / where / size / why.

## System packages (apt) installed earlier this session (with Alex present)
- `git`, `python3-venv`/`python3.12-venv`, `python3-pip`, `xclip` — dev basics.
  (No further apt installs during the autonomous run — needs sudo/password.)

## Python venvs
- `/data/llm/.venv` — `llama-cpp-python` 0.3.23, prebuilt CPU wheel (no cmake / no
  compile). Verified: runs GGUFs on CPU. For LLM-as-proposer / generation work.
  ALSO holds **torch 2.12.0+cpu** — the venv the `incubation/` engine (the
  torch-based world-model / value / controller experiments) and the active solver
  campaign run under (`/data/llm/.venv/bin/python`).
- `/data/Windows-files/Documents/cb/.venv` — playwright 1.60 (the `cb` browser harness).
- (The airfoil v0–v27 INDUCTION experiments themselves are pure stdlib — no venv
  needed; the `incubation/` engine needs the torch venv above.)

## Downloaded models / data
- `/data/llm/models/smollm2-360m-q8.gguf` — SmolLM2-360M-Instruct Q8_0, 384 MB
  (bartowski/SmolLM2-360M-Instruct-GGUF). Verified generating on CPU at ~72 tok/s
  (Zen3). For LLM-as-proposer experiments.
- `/data/llm/models/qwen3-1.7b-pmra.gguf` — **Qwen3-1.7B-PMRA**, ~962 MB (Alex's own
  mixed-rate-quant frankentensor, pulled from HF). The 1.7B proposer/recognizer arm
  used by **v22–v27** (the transfer suite + recognition/selection probes).
- `/data/pmra-runs/smoke-local/ggufs/` — **Qwen2.5-0.5B-Instruct** quants
  (Q3_K_M 355 MB, Q4_K_M 398 MB, Q5_K_M 420 MB). The 0.5B proposer used by the
  ARC grounding smoke + step-1 proposer eval.
- `/data/arc/` — **ARC-AGI-1** dataset (git clone fchollet/ARC; 400 training + 400
  evaluation tasks, JSON grids). The external benchmark for v19+ and the first
  ARC-AGI target of the active solver campaign.
- `/data/arc-agi-2/data/` — **ARC-AGI-2** corpus, newly cloned: `training/` (1000
  tasks) + `evaluation/` (120 tasks), standard `{train,test}` JSON. The harder
  second target for the active solver campaign (smoke ARC-AGI-1 first, then this).

## Active solver campaign (this chapter)
- `incubation/evolve/` — the **DIY-AlphaEvolve** creativity-campaign harness:
  `harness.py` is the fitness function (runs a candidate `solve(train, test_inputs,
  budget)` over a named ARC split — arc1/arc2 train/eval — scores by the ARC
  2-attempt rule + partial credit, persists a json log per run); `seed_solver.py`
  is gen-0 (the current best-first grid-distance DSL search over the 33-op grid DSL,
  MDL-pick of 2 attempts, no LLM/no learned nets). Run under `/data/llm/.venv`.
  (Two workflows actively write here — treat as live.)

## Notes
- Project lives at /data/Windows-files/Documents/airfoil (local git +
  github.com/asystemoffields/airfoil).
- GGUFs DO run locally (corrected 2026-05-31): `llama-cpp-python` CPU wheel, no
  cmake/compile needed. The earlier "blocked" note was a stale Windows-era assumption.
