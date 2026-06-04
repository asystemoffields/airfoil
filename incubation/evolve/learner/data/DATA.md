# SCALE-phase relation corpus — Branch-B

Data pipeline for crossing Branch-B from box-validation to the Kaggle SCALE phase, where
we train a **generative relation proposer** whose bet is to *generate relations beyond a
fixed grammar* (the coverage lever). This dir holds the unified loaders, a verified on-box
sample, and the Kaggle full-pull plan.

All loaders live in `arc_sources.py` and yield the learner's standard contract (identical to
`../ground_conceptarc.py`):

```python
(task_id: str, train: list[(in, out)], test: list[(in, out)])   # grids = numpy int arrays, <=30x30, 0-9
```

Program-bearing sources (BARC, RE-ARC) add a 4th element: the **relation program** (and for
BARC also `# concepts:` / `# description:` NL labels). That program/label text is the
relation-curriculum supervision signal.

---

## Corpus design — which sources, why, and how the PROGRAMS seed the curriculum

The proposer needs `(relation, demonstrations)` supervision at a scale ARC-1/ARC-2 cannot
provide, plus held-out distributions to *test coverage beyond a fixed grammar*. Four tiers:

| tier | source | n | programs? | role in the curriculum |
|---|---|---|---|---|
| **train (scale)** | **BARC ARC-Heavy** `barc0/*` | ~274k tasks | yes — imperative numpy `main()` + NL concept tags | **diversity + relation-mining pool.** Each task carries an explicit executable relation and a dense ARC-flavored concept label (symmetry detection, color mapping, reflection, object extraction, occlusion…). Use the `# concepts:`/`# description:` blocks as a *relation vocabulary to seed/order the curriculum*, the `main()` body as a *corpus to mine/abstract new relation primitives from* (concept tags now; program-AST / LLM-distilled relations later) — exactly the SCALE-phase "widen the grammar" bet. **Caveat:** the programs are imperative numpy, NOT in Branch-B's factored `effect × feature` grammar, so they are *not* drop-in relation labels — value is breadth/self-distillation + primitive mining, not direct labels. |
| **train (curriculum lever)** | **RE-ARC** `michaelhodel/re-arc` (HF mirror) | 400 relations, **unlimited** instances | yes — SSA-style DSL `verify_<id>` per relation | **the strongest CPU lever.** A procedural generator + executable verifier per ARC-1 training relation. Sample unlimited fresh labeled instances on CPU with a difficulty knob → the proposer sees each relation under huge within-relation variation (sizes/colors/object counts), which **forces generalizing relation representations over instance lookup**. The compact DSL (`palette, ofcolor, fill, objects, shift, compose, lbind…`) is itself a *target grammar to MEASURE whether the proposer generates relations outside it*. Difficulty band `diff_lb/diff_ub ∈ [0,1]` schedules easy→hard. |
| **held-out eval** | **ConceptARC** (local `/data/conceptarc`) | 160 | no | different, human, concept-labeled distribution; its 16 named concepts (ExtractObjects, Count, InsideOutside, …) give human-interpretable axes to **bucket proposer coverage**. Fixed/small → eval, not training. |
| **gate target** | **ARC-1** (`/data/arc`) + **ARC-2** (`/data/arc-agi-2`) | 400+400 / 1000+120 | no | the real targets. Held-out gate: `beyond gen2_base` on these. ARC-2 is the harder ceiling. |

**Coverage measurement (why two program sources, not one).** RE-ARC's DSL is a *known, bounded*
grammar (the 400 ARC-1 relations only — it adds breadth-per-relation, not new relation *types*).
BARC is *open* (LLM-authored, only auto-filtered, noisy, novel transforms). Training on RE-ARC +
holding out a RE-ARC relation subset tests in-grammar compositional generalization; evaluating on
BARC concepts / ConceptARC tests whether the proposer reaches relations *outside* the RE-ARC DSL —
the coverage lever's actual claim. The BARC `suggestfunction` variant (programs written against a
*suggested helper library*) is the bridge: it teaches a compositional vocabulary, closest to a
grammar/library-aware proposer.

---

## Exact format

**Task contract** (all sources): `(task_id, train, test)`; `train`/`test` are lists of
`(input_grid, output_grid)`, each grid a numpy int array, dims ≤30, values 0–9.

**BARC record** (`{source, examples, seeds}`, one JSON line):
- `source` — full Python module: `from common import *`, then `# concepts:\n# <tags>`,
  `# description:\n# <NL>`, then `def main(input_grid: np.ndarray) -> np.ndarray:` (often
  also `generate_input()`). **You cannot execute it** without BARC's `common.py` DSL — and you
  don't need to: the grids are precomputed.
- `examples` — flat list of `[input, output]` pairs (6–50 per record, all solved by the one
  program). **No built-in train/test split** — we shuffle (per-record seed) and carve first
  `n_train` (default 4) as train demos, the rest as test.
- `seeds` — origin ARC task filenames (e.g. `00d62c1b.py`).

**RE-ARC** — `generate_<id>(diff_lb, diff_ub) -> {"input","output"}` (DSL tuple-of-tuples;
`np.asarray(...,int)` handles it); `verify_<id>(I)->O` is the matching relation program.

**ConceptARC** — `challenges[tid]{"train":[{input,output}],"test":[{input}]}` +
`solutions[tid] = [output_per_test]`; concept label = alpha prefix of `tid`.

**Saved samples** (`samples/*.jsonl`, written by `pull_sample.py`): one task per line,
`{"task_id", "train":[[in,out]...], "test":[...], "program", "concepts"...}` with grids as
plain int lists. Re-load with `arc_sources.load_sample_file(name)`.

---

## What's pulled on the box (the VERIFIED sample)

RAM-safe: BARC streamed line-by-line (stopped after a few hundred records — **never** the ~3GB
file); RE-ARC generated on CPU from the <1MB code files. Total on-disk = **~17MB**. See
`manifest.json` for exact counts/bytes. Regenerate with `python data/pull_sample.py`.

- `samples/barc_heavy_4omini_sample.jsonl` — 300 canonical ARC-Heavy tasks + programs + concepts.
- `samples/barc_suggestfunction_sample.jsonl` — 100 library-aware-program tasks.
- `samples/rearc_generated_sample.jsonl` — 782 instances across all 400 relations (a few
  upstream-flaky generators auto-skipped) + verifier programs, diff band `[0.0, 0.6]`.

ConceptARC / ARC-1 / ARC-2 are already on disk; no pull needed.

### Verify the loaders

```bash
/data/llm/.venv/bin/python data/arc_sources.py     # runs every source on a small live sample, prints counts/shapes
/data/llm/.venv/bin/python data/pull_sample.py      # re-pulls the on-box sample + manifest
```

---

## KAGGLE DATA PLAN (full corpus, no 7GB box constraint)

The box only ever holds a sample. The full corpus pull happens on Kaggle (free GPU, ~19.5GB
`/kaggle/working`, 1 concurrent GPU, 30h/wk, 12h/session). Two channels, both already coded in
`arc_sources.py`:

**1. BARC ARC-Heavy — full pull, then local line-stream.**
```python
from huggingface_hub import hf_hub_download
# repo -> exact resolve filename (VERIFIED 2026-06-04 via HfApi.list_repo_files):
#   barc0/100k-gpt4omini-...   100k_gpt4o-mini_generated_problems.jsonl              (~3.15GB, ~102k)
#   barc0/100k-gpt4-...        100k-gpt4-description-...-code_generated_problems.jsonl(~3.1GB,  ~103k)
#   barc0/200k_HEAVY_...       data_100k.jsonl (2.8GB) + data_suggestfunction_100k.jsonl (3.4GB)
p = hf_hub_download("barc0/100k-gpt4omini-description-gpt4omini-code_generated_problems",
                    "100k_gpt4o-mini_generated_problems.jsonl", repo_type="dataset")
for tid, train, test, meta in arc_sources.load_barc_local(p):   # constant-RAM line stream
    ...   # meta.program / meta.concepts = relation-curriculum gold
```
~12GB of JSONL across all three repos fits in `/kaggle/working` (19.5GB). `load_barc_local`
streams line-by-line so RAM stays flat even there. Each repo fits in one session's download.
Filter degenerate/oversized tasks via `validate=True` (default) — LLM data is only auto-filtered,
expect noise.

**2. RE-ARC — pull the small code, generate unlimited on GPU node CPU.**
```python
arc_sources.ensure_rearc_code("/kaggle/working/rearc_code")   # <1MB; matplotlib auto-shimmed
for tid, train, test, prog in arc_sources.load_rearc_generated(
        n_train=4, n_test=1, diff_lb=0.0, diff_ub=1.0, instances_per_task=50):
    ...   # 400 relations x 50 = 20k labeled instances, regenerate freshly every epoch
```
Generation is CPU-cheap and unbounded — schedule an easy→hard curriculum by ramping
`diff_ub`. **Do not** pull `re_arc.zip` / the 400 pre-gen JSONs (~700MB) unless offline; the
generators dominate them. For a fixed offline set use `load_rearc_pregen_local(tasks_dir)`.

**3. Packaging for repeatable runs.** To avoid re-downloading BARC each session, snapshot a
filtered+sharded subset (e.g. 50k validated tasks as a compact JSONL) as a **Kaggle Dataset**
once, then mount it read-only across sessions — sidesteps the per-session download and the HF
rate path. RE-ARC is regenerated each run (no dataset needed; pin to GitHub
`michaelhodel/re-arc@master` for provenance — the HF mirror is third-party).

**Splits for the coverage test.** Train the proposer on RE-ARC generated streams + BARC; **hold
out** a RE-ARC relation subset *and* all of ConceptARC for coverage/transfer eval; gate on
ARC-1/ARC-2. Keep held-out relations entirely unseen during training.

---

## Gotchas (carried from the source survey, confirmed here)

- **BARC filenames** differ per repo and were verified live: the gpt4 set's file is
  `100k-gpt4-description-gpt4omini-code_generated_problems.jsonl` (not `100k_gpt4_...`), and the
  200k repo uses `data_100k.jsonl` + `data_suggestfunction_100k.jsonl`. `BARC_REPOS` in
  `arc_sources.py` holds the verified map.
- **Never `load_dataset()` / full-download BARC on the box** — one 3GB JSONL OOMs 7GB RAM. Stream.
- **BARC programs `from common import *`** (BARC's DSL) and define `generate_input()` — not
  importable without that DSL. Not needed: grids are precomputed in `examples`.
- **RE-ARC `utils.py` imports matplotlib at module top** (only for a plotting helper). We shim
  matplotlib + tqdm in `ensure_rearc_code` so generation needs neither — don't install the
  matplotlib stack just to generate.
- **A few RE-ARC generators raise on some diff draws** (e.g. `generate_11852cab`). `load_rearc_generated`
  retries (`max_attempts=8`) and skips a still-failing relation rather than crashing the stream.
- **RE-ARC covers only the 400 ARC-1 *training* relations** — breadth-per-relation, not new
  relation *types*. Keep that in mind when claiming "coverage beyond a fixed grammar."
- **RE-ARC mirror path has a literal space + parens**: `re-arc/re_arc (1)/re_arc/tasks/<id>.json` —
  quote it exactly (`REARC_TASK_TMPL`).
- **200k_HEAVY `/info` reports `examples` as float64** (datasets-server type-inference quirk);
  values are integer grids — we cast to int (`validate=True` also bounds-checks 0–9).
- **BARC is LLM-generated, auto-filtered only** (program reproduces its examples), NOT
  human-verified — expect noise/degenerate tasks. `barc_record_to_task(validate=True)` drops
  malformed/oversized/all-identity tasks; filter further before curriculum use.
- **Avoid the `induction_*/transduction_*_messages_format` parquet variants** — chat-formatted SFT
  data with grids+code buried in prompt strings. Parse the `*_generated_problems` JSON sets.
