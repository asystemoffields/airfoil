# Inventory — everything added to the machine during the autonomous run

Keep this honest so nothing is a mystery later. Format: what / where / size / why.

## System packages (apt) installed earlier this session (with Alex present)
- `git`, `python3-venv`/`python3.12-venv`, `python3-pip`, `xclip` — dev basics.
  (No further apt installs during the autonomous run — needs sudo/password.)

## Python venvs
- `/data/llm/.venv` — `llama-cpp-python` 0.3.23, prebuilt CPU wheel (no cmake / no
  compile). Verified: runs GGUFs on CPU. For LLM-as-proposer / generation work.
- `/data/Windows-files/Documents/cb/.venv` — playwright 1.60 (the `cb` browser harness).
- (The airfoil v0–v17 experiments themselves are pure stdlib — no venv needed.)

## Downloaded models / data
- `/data/llm/models/smollm2-360m-q8.gguf` — SmolLM2-360M-Instruct Q8_0, 384 MB
  (bartowski/SmolLM2-360M-Instruct-GGUF). Verified generating on CPU at ~72 tok/s
  (Zen3). For LLM-as-proposer experiments.

## Notes
- Project lives at /data/Windows-files/Documents/airfoil (local git +
  github.com/asystemoffields/airfoil).
- GGUFs DO run locally (corrected 2026-05-31): `llama-cpp-python` CPU wheel, no
  cmake/compile needed. The earlier "blocked" note was a stale Windows-era assumption.
