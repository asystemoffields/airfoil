# Research log — compression-as-generalization

Autonomous run kicked off 2026-05-30 (Alex away for his dad's birthday).
Newest entries at the bottom. Each entry: what I tried, what happened, what next.

---

## v0 — toy unary-int DSL, frequency-code bits  (`induct.py`)
- Built a tiny DSL (unary int ops), defined motifs, solved a train set by search
  (14/14), then BPE-style library learning, measured description length in bits
  under a frequency code (-log2 p) fit on train.
- **Result:** related compressed only 1.27×; control *inflated* (0.50×, i.e. got
  2× more expensive). The frequency code *specialises* to the training
  distribution — base ops absorbed into macros become rare and so the standalone
  codes for them get expensive, penalising the control.
- **Read:** real phenomenon (no-free-lunch of a specialised code) but a muddy
  headline. Also noticed the search "optimises away" planted motif structure by
  finding shorter semantic equivalents — interesting, parked for later.
- **Next:** a cleaner metric where controls stay flat by construction.

## v1 — clean metric: expression length in learned symbols  (`induct_v1.py`)
- Metric = fewest library symbols needed to write a program (DP). A held-out
  program can only shrink if it *reuses* learned structure; an unrelated one
  stays at full length by construction. Train = 4 motifs + all depth-2 pairs;
  held-out = novel **depth-3** compositions; two controls (disjoint-ops and
  same-ops-scrambled-to-avoid-motifs).
- **Result (clean ✓):**
    - related        6.83 → 3.00 symbols  (**2.28× shorter**)
    - ctrl-disjoint  6.17 → 6.17          (1.00×, flat)
    - ctrl-scramble  6.00 → 6.00          (1.00×, flat)
  Learned exactly the motifs (dbl-inc, sqr-inc, dbl-dbl, inc-inc-inc, +inc-inc).
- **Read:** unambiguous structural transfer — abstractions from depth-2 training
  shorten unseen depth-3 compositions, and do nothing for unrelated work. This is
  the core thesis demonstrated with proper controls.
- **Next (v2):** make it bits, not just symbol-count — a two-part MDL that also
  *charges* for the library itself, and show the library cost is amortised across
  tasks (net corpus bits fall). Then v3: does the library make *solving* new tasks
  faster, not just describing them?

## v2 — two-part MDL in bits  (`induct_v2.py`)
- L_total = L(library: each macro spelled out in base ops × log2|BASE|) +
  L(data: uniform codebook, log2|vocab| bits per symbol).
- First cut used a frequency code → unstable on the tiny corpus (train cost
  wobbled UP before down, control drifted — v0's specialisation gremlin).
  Caught it, swapped to a stable uniform codebook.
- **Result (clean):**
    - train MDL    209.4 → 153.0 bits  (1.37×) — monotonic; the library MORE THAN
      pays for itself (net bits fall even counting its 28.4 bits). best_merge
      halts at 5 macros = exactly the recurring motifs (freq≥2 exhausted) → it
      stops abstracting when no reusable structure is left.
    - held-out related 106.0 → 62.3 bits (1.70×).
    - held-out control 93.1 → 124.5 bits — RISES.
- **Read:** (1) abstractions pay for themselves on structured data AND transfer
  (1.70× on unseen related). (2) Honest no-free-lunch: in a uniform-codebook bit
  measure a bigger library slightly taxes UNRELATED work (control inflates);
  v1's symbol-count lens hides this because it ignores codebook size. Both lenses
  agree: abstractions help related, never help control.
- **Next (v3):** (a) search-cost transfer — does the library make *solving* new
  tasks faster (fewer nodes enumerated)? (b) a train/test split where memorising
  whole training compositions ≠ generalising, to expose the overfit-vs-generalise
  boundary (the generalisation sweet spot, distinct from train-MDL).

## v3 — search-cost transfer  (`induct_v3.py`)
- New question: do abstractions make *solving* (search) new tasks cheaper, not
  just describing them? Iterative-deepening enumeration over a vocabulary,
  counting programs evaluated until one matches the examples. Base vocab (6 ops)
  vs learned vocab (6 + the 5 motif macros).
- **Result (clean, 0.37s):**
    - RELATED held-out: base median 25,761 nodes → learned 1,207 = **21× faster**
      (both 6/6 solved).
    - CONTROL: base 154 → learned 499 nodes (~3× slower; both solved).
- **Read:** abstractions are SEARCH ACCELERATORS where structure matches (21×) and
  a TAX where it doesn't (extra symbols = bigger branching, no shortcut) — the same
  double-edge as v2's bit cost. (Aside: the solver returns shortest *semantic*
  equivalents, so control functions solve in few nodes via short equivalents — the
  valid comparison is base-vs-learned *within* a set, which holds.)
- **Next (v4):** depth generalization (train depth ≤k, test depth >k) + the
  generalize-vs-memorize boundary: does minimizing description cost on TRAIN start
  *memorizing* whole training compositions instead of learning generic motifs?
  Find the abstraction level that maximizes *held-out* generalization.

## v4 — depth generalization + memorize-vs-generalize  (`induct_v4.py`)
- **A. Depth generalization:** trained on depth≤2; the motif library compresses
  novel depth-3/4/5/6 compositions at a FLAT **2.25×** (2.25/2.25/2.25/2.26).
  Constant per-motif ratio across depth = genuine compositional generalization
  beyond the trained depth, not interpolation.
- **B. Memorize vs generalize:** BPE naturally halts at the 4 generic motifs
  (whole-pair adjacencies occur once → freq<2 → no merge). Force-memorizing the
  10 seen training pairs as macros cuts TRAIN 24→14 symbols (1.71×) but leaves
  novel held-out (descending-index, never-trained adjacencies) UNCHANGED at 16.
  => compressing past the generic motifs is pure memorization.
- **Read:** the thesis's two halves converge — the shortest *reusable* description
  (generic motifs) is exactly what generalizes; pushing compression further just
  overfits the training set. The objective's own stopping point = the
  generalization optimum (the Occam result of v2, now tied to generalization).
  "Shortness that generalizes stops where reuse stops."
- **Next (v5):** richer DSL — lists with map/fold/filter + parameterized ops — to
  test whether the story survives in a combinatorially larger, more realistic
  program space (DreamCoder-style list functions). Where it gets real.

## v5 — richer domain: list-processing pipelines  (`induct_v5.py`)
- Swapped toy unary int ops for real list functions (map inc/dbl/sqr/neg,
  filter even/pos, reverse, tail, init, sort). Motifs are genuine idioms:
  "even dbl*" (double the evens), "pos sqr*" (square the positives),
  "inc*×3" (+3), "rev tail" (drop last). Same BPE+min-symbol harness.
- **Result (holds):** transfer related **2.28×**, both controls flat (1.00×);
  depth generalization flat **2.25×** at depths 2-6; learned exactly the real
  idioms. Identical to v1/v4. Sanity-checked semantics (A([1,-2,3,4,-5,6])=[-4,8,12]).
- **Read:** the mechanism is domain-agnostic — it keys off compositional
  structure, which real programs share. The thesis isn't an artifact of the toy DSL.
- **Caveat (honest):** v5 is still LINEAR pipelines (point-free composition), so
  the harness is unchanged. The real leap is abstraction over TREES — higher-order
  map/fold with sub-program arguments (true DreamCoder). That's v6/v7.
- **Next (v6):** either (a) the "domesticated learner" — a learned proposer that
  guides search (features→next symbol) vs uniform enumeration, measuring the
  search-efficiency gain (where a small model could play proposer); or (b) the
  jump to a higher-order tree DSL. Lean (a) first — smaller, and directly tests
  the "domesticated learner" idea from the design conversation.

## v6 — the domesticated learner  (`induct_v6.py`)
- Added a bigram proposer over the library symbols (fit on the training
  solutions) to ORDER a best-first search, vs v3's uniform enumeration. The
  objective is unchanged (find a program that fits); only the *search order* is
  guided by the learned model.
- **Result:** RELATED uniform 1,207 → guided **95** nodes = **12.7×** further
  speedup (on top of v3's 21× over base ≈ **270× cumulative**). CONTROL uniform
  499 → guided 1,206 (**0.4× — SLOWER**: the learned prior mis-prioritizes,
  confidently exploring motif-shaped branches when none apply).
- **Read:** the learner, domesticated to guide the search, earns its keep exactly
  where structure was learned and actively taxes where it wasn't — the same
  double-edge as v2/v3, now at the proposer level. Confirms the design-conversation
  framing: choosing compression as the objective doesn't banish the learner; it
  leashes it to guiding the program search, which is where sample-efficiency lives.
- **Next (v7):** the real leap — a higher-order TREE DSL (map/fold with sub-program
  arguments) to test abstraction over trees, not linear pipelines (true
  DreamCoder). Stretch: a context-conditioned proposer (small net / GGUF).
