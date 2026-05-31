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

## v8 — verifier wind tunnel  (`induct_v8.py`)  [logged out of order; see v7 above]
- The whole amortization loop rests on the verifier, which is free+perfect in
  program synthesis and cheap+noisy everywhere real. Gave the verifier a
  false-accept rate eps; measured oracle-checked solving accuracy vs eps and
  redundancy M (a wrong program survives only if it passes M independent checks
  → eps^M).
- **Result:**  M=1 / M=3 / M=5 / M=8
    - eps=0.00: 100 / 100 / 100 / 100
    - eps=0.10:   4 /  92 / 100 / 100
    - eps=0.20:   0 /  70 / 100 / 100
    - eps=0.40:   0 /  12 /  56 /  96
- **Findings:** (1) a 10% false-accept verifier ALONE collapses naive solving to
  4% — asymmetry: hundreds of short WRONG candidates precede the right one, so a
  rare fluke-accept ends the search wrong. (2) Redundancy rescues exponentially
  (eps^M): eps=0.1 → M=3=92%, M=5=100%. Rule M ≳ ln(R)/ln(1/eps); scales with the
  LOG of the search space, so it's cheap. (3) Floor: eps=0.4 needs M=8 for 96%;
  below ~chance there's nothing to amplify.
- **Read:** you don't need a RELIABLE verifier — a cheap UNRELIABLE one (err<~0.5)
  + a few confirmations suffices. Redundancy converts weak→strong exponentially
  (= self-consistency/B2, majority vote, biological consolidation needing
  repetition). Architectural law for the loop: **wrap the verifier in consensus
  before crystallizing anything** ("never consolidate a one-off," with a sizing
  formula).
- **HONEST CAVEAT (= the next experiment):** this assumes INDEPENDENT errors
  (noise re-rolled per check). Real verifier errors are often CORRELATED /
  systematic — the same wrong case fools the verifier identically every time, and
  then naive repetition does NOTHING. The fix is DIVERSE verifiers (different
  lenses), not repeated identical checks — the perspective-diverse-verification
  idea. That's v9.

## v9 — repetition vs diversity  (`induct_v9.py`)
- Modeled a SYSTEMATICALLY weak verifier as a partial test suite (covers 1 of 8
  cases; deterministically blind elsewhere). Repetition = same suite M times
  (identical verdict). Diversity = M *different* suites; a wrong program must fool
  the UNION of their covered cases.
- **Result (1-case verifiers, same budget M):**
    - repetition: 34/35/27/29/29% — flat. Volume can't fix a systematic blind spot.
    - diversity:  30/84/95/99/100% — climbs to certainty.
- **Read:** redundancy only buys v8's eps^M when errors are INDEPENDENT. For
  systematic errors you must MANUFACTURE independence — diverse lenses / evidence /
  framings, not the same check louder. v8 (re-rolled noise) = optimistic limit;
  v9 repetition = pessimistic limit (zero diversity); reality is between, and the
  engineering job is decorrelating the verifiers. **Verifier diversity, not volume,
  is the ballgame** — why biology cross-checks across senses, and Monte Carlo uses
  decorrelated restarts.
- **Next (v10):** wire a diverse-verifier consensus gate into the ACTUAL library
  loop — does it prevent the noisy-verifier library *poisoning* (compressing wrong
  solutions into garbage abstractions)? First true end-to-end loop test.

## v10 — library poisoning & the consensus gate  (`induct_v10.py`)
- First end-to-end loop test. Crystallize 4 primitive skills (the motifs) by
  solving each under a verifier gate, then solve held-out COMPOSITIONS (depth 2,3)
  purely by composing the crystallized skills (no new search). Verifier = partial
  test suite (v9 model); gate K = # of diverse suites that must agree to crystallize.
- **Result:**  skill / depth-2 / depth-3 correct
    - K=1: 29 / 15 / 10      K=2: 89 / 81 / 75
    - K=3: 96 / 92 / 90      K=5: 99 / 99 / 99
- **Findings:** (1) poisoning AMPLIFIES through reuse — a composition is correct
  only if every reused skill is, so accuracy compounds ~skill_acc^depth (29%→10%
  by depth 3); one bad primitive corrupts a whole family. (2) The diverse-consensus
  gate at crystallization restores skills→99%→compositions→99%. (3) Because reuse
  amplifies, the gate matters MORE than in one-shot solving — it protects every
  future composition at once.
- **Read:** the loop closes. A cheap weak verifier + diversity AT THE GATE buys a
  clean reusable library. In a reuse/amortization system, verification isn't
  per-task insurance — it's the wall between you and compounding structural error.
  This is v8+v9's payoff inside the actual loop.
- **Next (v11):** parameterized fragments (antiunification) earning their keep, then
  graduate toward a real abstraction benchmark (mini-ARC) — where generalization is
  measured, not assumed.

## v11 — parameterized fragments (antiunification)  (`induct_v11.py`)
- Built a corpus whose shared structure is a NON-CONTIGUOUS template: frame
  (sqr, ?, inc), middle op varies. Exact/BPE reuse only sees contiguous repeats →
  learns NOTHING here. Antiunification (positional least-general-generalization)
  recovers the schema (sqr, ?, inc).
- **Result on held-out novel-filler instances:** exact desc 3 symbols / ~205-node
  search; schema desc 2 tokens (**1.5×**) / ~4-node search (**51×** less). And it
  generalizes to fillers never seen in training.
- **Read:** parameterized fragments earn their keep exactly where exact ones are
  blind — when the shared structure is a template with a varying slot. The step
  from "reuse what recurs verbatim" to "reuse the pattern"; binding a schema's hole
  is what lets a few examples cover an open set (the essence of generalization).
- **Honest caveat:** clean best case — one schema, single-op holes, equal lengths,
  positional lgg. The general problem (many schemas needing CLUSTERING, choosing
  WHICH generalization among many candidate lggs, multi-op/tree-level holes) is
  genuinely hard — and is where this connects to real abstraction and a real
  benchmark.
- **Next (v12):** a mini-ARC subset or held-out list-functions suite (generalization
  measured, not assumed) + general antiunification. Best co-driven (real design
  choices).

## v12 — generalization MEASURED (held-out benchmark)  (`induct_v12.py`)
- Left "by construction". Trained by SOLVING 16 depth-2 list-program tasks at base
  level, then BPE-DISCOVERED the idioms by compressing the solutions (never told
  them). Evaluated on NOVEL depth-3 held-out tasks — test-near (novel compositions
  of discovered idioms) and test-far (a HELD-OUT idiom E) — vs a base-only baseline
  at a fixed 20k-node budget.
- **Discovery worked:** recovered exactly the 4 ground-truth idioms (even dbl*,
  pos inc*, rev tail, sqr* neg*) + one composite, purely by compression.
- **Result:** test-near base **35%** → learned **100%**; test-far base 25% →
  learned 50% (median nodes near-learned ~2431).
- **Read:** generalization measured — a library DISCOVERED (not planted) from
  depth-2 training extends the solvable frontier to novel *deeper* held-out tasks;
  strongly where they share learned structure (35→100%), weakly where they don't
  (25→50%, the honest boundary). The transition from toy demonstration to real
  held-out measurement.
- **Honest scope:** still a synthetic distribution (not an external benchmark like
  ARC); single solve-then-compress pass (not the full iterative wake-sleep);
  exact-match idioms (v11 antiunification not yet wired into the loop).
- **Next (v13):** iterative wake-sleep; wire in v11 antiunification + v6 proposer;
  then a real external benchmark (mini-ARC).

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

## v7 — the tree leap: abstraction over higher-order EXPRESSION TREES  (`induct_v7.py`)
- The leap the whole project was building toward. Programs are now TREES, not
  sequences: a higher-order DSL with `map(λx.body, xs)` / `filter(λx.pred, xs)`
  where the body/pred are SUB-PROGRAMS (lambda bodies over a bound var) built from
  leaf int ops (inc/dbl/tpl/sqr/neg/dec) and predicates (even/odd/pos). Real
  semantics, sanity-checked (`map(λx.2x+1,·)`, `filter(pos, map(λx.x²+1,·))`, and a
  genuinely tree-shaped depth-2 program all evaluate correctly).
- Abstraction = the tree analogue of BPE: greedily MINE the most-reused SUBTREE
  across the TRAIN trees, name it a library fragment, repeat until reuse runs out
  (count<2) — the exact stopping rule of v1/v4, lifted to trees. DL of a tree =
  node count where any subtree equal to a fragment collapses to 1 node (bottom-up).
  Ground fragments done cleanly; one-hole antiunification done as a small honest
  stretch (reported, NOT wired into the metric).
- Setup mirrors v1: TRAIN = motif bodies under single + depth-2 combinator stacks;
  TEST_RELATED = novel deeper (depth 3-4) trees REUSING the motif subtrees;
  ctrl-disjoint = leaf ops no motif uses (true no-shared-structure control);
  ctrl-scramble = SAME map-body leaf ops but arranged so no body-motif appears,
  filtered by the non-trained `odd` predicate.
- **Result (✓ holds, one honest caveat):**
    - related        12.50 -> 6.17 nodes  (**2.03×**)
    - ctrl-disjoint  12.50 -> 12.50       (1.00×, dead flat)
    - ctrl-scramble  12.50 -> 11.83       (1.06×, near-flat)
    - depth-gen: flat ~2.2× at depths 2-6 (2.40/2.29/2.22/2.18/2.15) — gently
      DECAYING toward an asymptote (deep trees accrue cheap leaf nodes the library
      can't collapse), not the perfectly-flat 2.25× of the linear v4/v5.
  Library mined exactly the intended motifs (inc∘dbl, inc∘sqr, dbl∘dbl as bodies;
  the two filters; the three full maps) plus the sub-fragment `dbl(x)`.
- **Read:** structural transfer LIFTS from linear pipelines to trees. Recurring
  subtrees discovered on shallow training shorten unseen deeper compositions and
  do literally nothing for the disjoint control. The thesis is not a property of
  the linear/BPE encoding — it keys off compositional reuse, which trees have too.
- **Caveat (the genuinely tree-specific finding, honestly):** ctrl-scramble is
  1.06×, not a perfect 1.00×. Cause: in a TREE, sharing a leaf op (`dbl`) means
  sharing the 2-node subtree `dbl(x)`, which the miner names — so any control that
  reuses leaf ops leaks a little via PARTIAL-subtree overlap. This has NO analogue
  in v1-v6: a linear "scramble" could perfectly avoid every motif bigram, but trees
  expose sub-motifs at every internal node. The disjoint control (no shared ops at
  all) is therefore the clean structural control here, and it is exactly flat. An
  earlier scramble that also reused the trained even/pos filters leaked more (1.15×);
  isolating to leaf-only overlap drops it to 1.06×, with the residual fully
  accounted for (5/6 trees contain `sqr(dbl(x))`, saving 1 node each).
- **Antiunification (stretch, correct but minor):** the least-general
  generalization of two same-shaped subtrees yields a one-hole schema, e.g.
  `map(λx.dbl(dbl(x)), map(λx.?(dbl(x)), xs))`. It works and is deterministic, but
  on this tiny corpus the highest-support one-hole schema only matches 2 subtrees,
  so it earns little — left out of the DL metric on purpose (ground fragments are
  where the measured gain lives). Real parameterized-fragment payoff needs a bigger,
  more varied corpus.
- **Next (v8):** (a) make antiunification EARN its keep — a corpus where a
  parameterized schema (map-over-any-op, fold-with-any-combiner) beats every ground
  fragment, and fold them into the DL metric to measure parameterized transfer; or
  (b) graduate toward a mini-ARC / abstraction-benchmark subset (the PLAN's other v7
  branch) now that tree abstraction is demonstrated. Lean (a): it's the honest next
  rung — ground subtrees transferred; the open question is whether HOLED subtrees do.
