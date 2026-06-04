# Branch B — the learner (amortized relation-induction)

**Goal.** Learn to *propose* cause→effect relations (vs blind-fit the hand-authored menu), so a relation space
*too big to enumerate* becomes navigable — the only regime where a learner beats the families' enumerate+verify.
Box-first (CPU, small/deep models); Kaggle later (scale + the RL/verifier-as-reward loop). Standardized gate
unchanged: `beyond gen2_base` on held-out ARC. Data: the self-gen curriculum (`grammar.py`, the only source
with clean *relation labels*) now; the HF ARC-ish sets (ConceptARC, BARC ARC-Heavy) for diversity + self-
distillation later.

**The decisive unit test (pure synthetic, CPU):** train a *factored* proposer on a subset of (effect × feature)
combos, hold some out entirely; **held-out feature-acc ≫ 0.10** = the model identifies a causal feature it never
saw under that effect = compositional generalization over the grammar = the prerequisite for a learner that
*invents* rather than fits.

## Results

- **v0 — pixel CNN (`train.py`).** Encode demos as one-hot grids → effect + feature heads.
  `EFFECT 0.96 trained / 0.96 held-out` (generalizes); `FEATURE 0.27 trained / 0.00 held-out` (fails).
  *Lesson:* a pixel-CNN learns the transform TYPE but can't extract the causal FEATURE (counting holes/sizes +
  per-object property→outcome correlation is exactly what CNNs are worst at) — **and we shouldn't make it**:
  `grammar.py` segments objects and computes every feature deterministically. Don't re-learn perception.

- **v1 — object-centric relational transformer (`train_v1.py`, 0.54M).** FEED the deterministic object
  feature-tables + each object's outcome to a 4-layer transformer (objects attend to each other).
  `EFFECT 0.88 / 1.00`; `FEATURE 0.21 trained (≈random) / 0.00 held-out`; **loss FLAT (1.89→1.86)**.
  *Lesson:* not under-training — a generic relational model does **not discover** the consistency-detection
  ("which feature's value determines the outcome across objects/demos"). Relevance is the wall, not perception.

- **Meta-finding.** On a clean, small, exactly-verifiable feature space, learned relevance is (a) hard to train
  end-to-end and (b) **unnecessary** — deterministic enumerate+verify already solves it (the families' 14
  beyond-base). The learner's value is only at SCALE/COMPOSITION (space too big to enumerate) or under NOISE
  (no exact verifier). So this toy is a unit test of "can the model learn relevance *at all*"; only after it
  passes do we move to the regime where the learner actually pays.

- **v2 — explicit pairwise CONSISTENCY (`train_v2.py`, 2.6K params): POSITIVE.** Bake the bias in: for each
  feature *j*, a *shared, feature-index-agnostic* scorer maps the per-feature pairwise stats
  [same-value_j, same-outcome, and cross-terms] → score_j; feature = argmax_j score_j.
  `EFFECT 0.87 / 0.99`; `FEATURE 0.64 trained / **0.50 held-out**` (5× random) — and it **transfers** to combos
  it never trained. **The Branch-B prerequisite is MET**: with the right inductive bias, causal-feature
  relevance is learnable AND compositionally generalizable. The 2.6K param count is the point — the lever is
  relational STRUCTURE, not scale ("keep perception + structure as architecture, learn only the relevance").
  The recognition/relevance wall (recurring since the v22–v25 LLM-selector) yields to the right relational bias.

  *The v0→v1→v2 arc:* pixel-CNN can't (0.00) → generic relational transformer can't (0.00) → consistency-
  structured cracks it (0.50 held-out), at 2.6K params.

## What v2 establishes, and the two value-tests next

v2 proves the architecture *can represent + transfer* relevance. It does NOT yet prove the learner is *worth
it* — on the clean toy this is redundant with `induce()`'s deterministic per-feature check (enumerate+verify
already solves it). The learner only pays in two regimes, which are the next experiments:
1. **SIM-TO-REAL (the immediate next step):** does the synthetic-trained recognizer transfer to *real ARC*
   tasks? Pipeline: recognizer proposes top-K (effect, feature) → `induce()` over decomps → exact-verify →
   measure solve-rate + `beyond_base`. If it transfers, the recognizer is real; if not, close the gap with the
   HF ARC-ish sets (ConceptARC held-out, BARC ARC-Heavy self-distillation) + verified-ARC-solution distillation.
2. **SCALE (does it pay):** widen the grammar to composed/multi-step relations (a space too big to blind-
   enumerate under a budget) and check learned-proposal + verify > blind-enumeration + the same verify budget.

Then Kaggle: scale model + data + the RL/verifier-as-reward loop. Target 600M-class on GPU; the box stays in
the few-K-to-few-M relational regime where this problem has, so far, actually lived.

## SIM-TO-REAL (`ground_arc.py`) — the recognizer TRANSFERS to real ARC (green light)

Enumerated the grammar over ARC-1 → **24 grammar-solvable tasks** (19 train, 5 eval). The v2 recognizer
(trained on synthetic blobs ONLY) on those REAL tasks:
- **feature top-3 = 24/24 (1.00)** — proposes the correct causal feature for every real task in its top-3;
- effect top-1 = 15/24 (0.62);
- propose(top-2 eff × top-3 feat) → induce → exact-verify **SOLVES 23/24 (0.96)** vs the enumerate-all ceiling
  24/24 — with a tiny candidate set instead of all 81 types × decomps.

**The relevance signal is DOMAIN-GENERAL**: "objects sharing a feature-value share an outcome" reads off real
ARC demos exactly as off synthetic, because the consistency structure is abstract (not the visual statistics).
So sim-to-real transfer of *relevance* needed **no** HF data — the HF ARC-ish sets are for widening curriculum
diversity + self-distillation as the grammar grows, not for closing a (small) sim-to-real gap.

**Honest scope:** the 24 overlap `gen2_base`'s menu (enumeration is cheap here, so the recognizer *navigates*
rather than *solves more*). The recognizer PAYS once the grammar is WIDENED past enumerable size (more
decomps/features/effects + COMPOSED relations) — which is also when it reaches ARC tasks *beyond* `gen2_base`.

**Next:** widen the grammar (compose relations + more features/effects) → retrain the recognizer → ground on
ARC measuring `beyond_base`, comparing recognizer-top-K vs blind-enumeration *under a fixed verify budget*
(the regime where the learner is necessary, not just tidy). Then fold in HF data + go to Kaggle.
