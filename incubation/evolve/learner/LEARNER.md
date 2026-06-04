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

- **v2 — explicit pairwise CONSISTENCY (`train_v2.py`, next).** Bake the inductive bias in: for each feature
  *j*, a *shared* net scores whether each object's outcome is a consistent function of value_j across object
  PAIRS (same value_j ⇒ same outcome; different ⇒ different); the feature head = argmax over per-feature
  consistency scores. Shared across features ⇒ generalizes to held-out features by construction. If v2 cracks
  held-out feature-acc, the architecture *can* represent + transfer relevance → scale to composed relations
  (too big to enumerate) + ARC grounding. If even v2 fails, the honest read is that learned relevance is a weak
  lever vs enumerate+verify, and the campaign's leverage stays in a richer verified library (→ the LLM-as-
  relation-proposer question we deferred, or the Kaggle-scale generator).
