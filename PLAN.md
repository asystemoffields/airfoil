# Compression-as-generalization — research plan

**Thesis (Alex's):** a system whose objective is to find the shortest reusable
description of its data will *generalize*, because the pressure to be short
forces it to discover abstractions that recur. "Within the particular is
contained the universal." The learner is *domesticated*: it only guides the
search for short programs, it doesn't roam the open space of all algorithms.

**The measurement that would prove it (the one Alex loved):**
> Does solving corner A make corner B cheaper to describe?
Learn abstractions from a TRAIN set; measure the description length of solutions
to a held-out TEST set. If structurally-*related* held-out tasks compress while
an unrelated *control* does not, the gain is genuine structural transfer, not a
free lunch.

## Roadmap (autonomous)
- [x] **v0** toy unary-int DSL, BPE-style library learning, frequency-code bits.
      Weak signal + messy control (frequency code specialises → control inflates).
- [ ] **v1** clean metric = *expression length in learned symbols* (control stays
      flat by construction); depth-3 held-out compositions; two controls
      (disjoint-ops + same-ops-scrambled). Target: related ≥1.5× shorter, controls flat.
- [x] **v2** two-part MDL in bits (library cost + uniform codebook data cost).
      Train MDL 209→153 (1.37×, library pays for itself); related 1.70×; control
      RISES (bigger codebook taxes unrelated work — honest no-free-lunch).
- [x] **v3** search-cost transfer: learned library solves held-out RELATED tasks
      in 21× fewer search nodes (25,761→1,207); CONTROL ~3× slower (branching tax,
      no shortcut). Abstractions = search accelerators where structure matches.
- [x] **v4** depth generalisation: trained depth≤2 → flat 2.25× compression at
      depths 3-6 (compositional, not interpolation). Memorize-vs-generalize:
      forcing compression past generic motifs cuts TRAIN 1.71× but leaves novel
      held-out unchanged → BPE's own stopping point = the generalization optimum.
- [x] **v5** richer domain: LINEAR list-processing pipelines (map/filter/rev/sort/
      drop, real semantics). Transfer 2.28×, controls flat, depth-gen flat 2.25× —
      identical to the toy DSL → thesis is domain-agnostic, not a toy artifact.
      (Still linear; true higher-order TREE abstraction = v6/v7.)
- [x] **v6** the domesticated learner: a bigram proposer over the library guides
      best-first search. Related 12.7× fewer nodes than uniform (~270× cumulative
      over naive); control 0.4× (guidance actively mis-prioritizes where no
      structure was learned). Objective unchanged — the learner just guides.
- [x] **v7** the tree leap: higher-order TREE DSL (map/filter over sub-program
      lambda bodies), abstraction by mining recurring SUBTREES (BPE-for-trees),
      DL = node count with fragments as 1 node. Related compresses **2.03×**,
      disjoint control dead flat (1.00×), depth-gen ~2.2× (depths 2-6). Honest
      tree-specific caveat: same-leaf-ops control isn't perfectly flat (1.06×) —
      sharing a leaf op shares the tiny subtree `dbl(x)`, a partial-subtree leak
      with no linear analogue. Antiunification (holed fragments) sketched & correct
      but earns little on this tiny corpus → deferred to v8.
- [x] **v8** verifier wind tunnel: the load-bearing assumption stress-tested. A
      10% false-accept verifier ALONE → 4% accuracy (many short wrong candidates →
      a fluke accept ends search wrong). Redundancy rescues exponentially (eps^M):
      eps=0.1 M=3→92%, M=5→100%. Rule M≳ln(R)/ln(1/eps). Reframe: you don't need a
      reliable verifier, you need a cheap unreliable one (err<~0.5) + a few
      confirmations. Caveat: assumes INDEPENDENT errors; correlated/systematic
      errors defeat naive voting → need DIVERSE verifiers (v9).
- [x] **v9** repetition vs diversity: a SYSTEMATICALLY weak verifier (partial test
      suite). Repetition flat ~29% across M; diversity (different cases) climbs
      30%→100% on the SAME budget. Redundancy only buys v8's eps^M when errors are
      independent — for systematic errors you must MANUFACTURE independence. Verifier
      diversity, not volume, is the lever.
- [x] **v10** library poisoning & the consensus gate (first end-to-end loop test).
      No gate (K=1): skills 29% correct → depth-3 compositions just 10% — poisoning
      AMPLIFIES with reuse (~skill_acc^depth). Diverse gate (K=5) → 99/99/99. Because
      reuse compounds error, the crystallization gate matters MORE than in one-shot
      solving — it protects every future composition at once. Loop closes.
- [x] **v11** parameterized (antiunification) fragments earn their keep: on a
      NON-CONTIGUOUS template corpus (frame sqr,?,inc), exact/BPE reuse learns
      NOTHING; antiunification recovers the schema → novel instances 1.5× shorter to
      describe and 51× less search (fill one hole), generalizing to unseen fillers.
      Clean best case (one schema, single-op holes); general antiunification = v12.
- [x] **v12** generalization MEASURED — held-out list-functions benchmark. Trained
      by SOLVING depth-2 tasks then BPE-DISCOVERING the idioms (recovered all 4,
      untold); evaluated on NOVEL depth-3 held-out tasks vs a base baseline at a
      fixed budget. test-near: base 35% → learned 100%. test-far (held-out idiom):
      25% → 50% (honest boundary). Discovered, held-out, baselined — not planted.
- [x] **v13** iterative wake-sleep — the loop BOOTSTRAPS: round 0→1, discovering
      idioms from its own solutions lifts the solvable frontier 58→83% (train) and
      46→79% (HELD-OUT) on a fixed budget with a frozen base, generalizing. BUT
      ungoverned it then bloats (junk composites) and held-out DECLINES — unbounded
      compression eats itself. The wake-sleep needs an Occam razor at the sleep.
- [x] **v14** govern the sleep — a two-part-MDL razor (keep a macro only if it
      reduces total description length). Head-to-head vs v13's hoarding: governed
      holds held-out at 79% with 6 macros; ungoverned bloats to 24 and declines to
      75%. Suppression buys STABILITY + PARSIMONY (not, here, further climbing).
      Unifies v2 (two-part MDL) + v4 (stop where reuse stops) as the loop's governor.
- [x] **v15** governance that CLIMBS: given a curriculum with genuine deeper
      structure (recurring 2-idiom phrases), the governed loop discovers a HIERARCHY
      in steps (idioms → phrases) — held-out 25→100%, deep-solve cost ~11× cheaper
      (2576→227 nodes) as phrases form, library stays lean. Honest scope: shows
      hierarchy use, NOT a clean phrases-vs-idioms isolation (caught myself
      budget-fiddling to fake a gap, stopped) — the payoff is search cost + tight-
      budget reach.
- [~] **v16** matched-library idioms-vs-phrases + budget SWEEP (no tunable knob).
      Right instrument; necessity regime is REAL (depth-6 idioms cap at 67% within
      100k while phrases reach 100%; depth-4 idioms catch up only at the high end).
      BUT confounded: some "deep" tasks COLLAPSE to short functions (shortest-
      equivalents at the task level), inflating the idiom baseline. Signal survives
      on incompressible tasks; clean isolation deferred to v17.
- [x] **v17** the clean necessity window — incompressible tasks BY CONSTRUCTION
      (each idiom has one sqr → depth-K task is degree-2^K → any program needs ≥K
      squarings → none shorter exists; eval mod prime on random points). Matched-
      library sweep, no collapse: depth-4 idioms catch up only at 10k (phrases from
      300); depth-6 idioms NEVER within 100k (phrases from 3k). Idiom cost ~|V|^depth
      runs away exponentially; deeper abstraction is REQUIRED, not just cheaper.
      The airfoil thesis quantified. Resolves v16's confound.
- [x] **v18** learned policy in the loop (inner-loop, KB-sized bigram). Policy-order
      vs uniform-order over the same governed library, on v17's incompressible domain:
      held-out deep tasks solved in ~14× fewer nodes (220→16); the policy (branching↓)
      and library (depth↓) co-evolve across rounds (round 0 identical → round 4 13.8×).
      The complete amortization loop running end-to-end on a frozen base.
- [~] **v18b** (staged) a frozen small GGUF as an OUTER-loop proposer — sparse
      high-level suggestions (a few calls/task, latency-amortized), vs the cheap inner
      policy. Module: `gguf_proposer.py` (loads SmolLM2-360M, ~72 tok/s). Measure
      next: does the LLM prior beat the learned policy enough to justify its latency?
- [~] **v19** EXTERNAL benchmark: real ARC (fchollet/ARC, 400 tasks), distribution
      not ours. **Cycle 1 baseline:** parameter-free geometric DSL (11 ops) + depth-3
      search solves **20/400 = 5.0%** (geometric slice; train-consistent programs also
      pass held-out test — no overfit). The honest floor.
- [~] **v19 cycle 2** +scale +inferred recolor → 6.8% (27/400). KEY FINDING: the
      cross-task airfoil LOOP did NOT move ARC — solutions are shallow (1-3 ops), so
      there's nothing deep to compress; the lever is DSL *breadth*. This maps the
      thesis boundary: the loop attacks DEPTH (compositional reuse, shown on v1-v18);
      ARC is hard along BREADTH (many distinct concepts). A measured scope boundary,
      not a failure.
- [x] **v19 cycle 3** boundary measured: cross-task library mined from solved ARC
      train → held-out transfer ≈0 (solve 6→6/200, nodes 10→10), vs ~14× on synthetic
      deep tasks (v18). The airfoil loop is a DEPTH tool, not a BREADTH tool — ARC is
      breadth-hard. The external benchmark mapped the scope. Thesis questions answered.
- [x] **v20** the BREADTH half as architecture test (feature-recognizer + bigger DSL
      on ARC, 3 conditions). PREDICTION FALSIFIED: small 5.8% / big-no-recog 6.5% /
      big+recog 6.0%; overfit 1/1/1. No overfit materialized (ARC's 3-5 train pairs
      constrain → nothing for recognition to regularize); the crude recognizer slightly
      HURT (excluded needed ops). Real lever = raw DSL coverage. REFRAME (= Alex's "this
      is perfect for an LLM"): a hand-coded recognizer is too dumb to BE the breadth
      half — genuine recognition is perception+world-knowledge = an LLM's job.
- [ ] **v21** LLM-as-recognizer: a model reads an ARC task → proposes relevant
      concepts/ops → narrows the depth-engine. Honest constraint: local 360M far too
      weak (clueless on grids); the laptop's RAM limit bites precisely on the BREADTH
      organ (which wants the big frozen model) while the depth half runs cheap. Test
      what a runnable local model (e.g. a 1-3B Q4) can recognize, and name the gap.
- [x] **v22** the airfoil STAIRCASE on ARC. **RESULT (NEGATIVE on ARC):** 360M+airfoil
      3.5% raw / 1.8% struct — both BELOW the 6.5% blind floor; the perception aid HURT
      (struct named on fewer tasks); 135M 1.5% (dropped); 1.7B arm abandoned (CPU-slow on
      a benchmark already negative). A small model's category-naming excludes needed ops,
      and ARC is breadth-hard — wrong benchmark for a depth engine. Doesn't kill the
      method; the decisive test is v23. Two bugs fixed en route (grid-explosion, downscale).
      ORIGINAL SPEC: two axes on real ARC. CAPABILITY =
      SmolLM2-135M → 360M → Qwen3-1.7B-PMRA (Alex's own frankentensor quant, pulled
      from HF). PERCEPTION = raw grid vs a "struct" rendering (sizes, area-ratio,
      color sets, symmetry) — a cheap perception prosthetic so the model need only
      NAME, not perceive. Fixes v20's dead end: an EQUAL-COMPUTE node budget (256 ≈
      full-DSL depth-2) so correctly narrowing the op-set BUYS depth (depth-2 over 15
      ops ≈ depth-3 over 6) — the only way the LLM's breadth can ADD over blind search.
      Q: does scaffold payoff scale with perception/size? ("imagine for a 60B.")
- [ ] **GATE (Alex, 2026-05-31):** proceed to a shippable release ONLY if v22 shows a
      JUSTIFYING lift SOMEWHERE on the ladder — a (model × perception) combo where
      +AIRFOIL beats BOTH floors (model-alone AND the 6.5% blind-search). No lift on
      the ladder → report the boundary, don't ship, rethink. ARC is breadth-hard
      (v19/v20), so a null here doesn't kill the method — it may just confirm ARC is
      the wrong target and push v23 forward as the real test.
- [x] **v23 — TRANSFER SUITE. RESULT (two findings):** 1.7B+airfoil vs ALONE = lists
      10→27%, strings 7→57%, numbers 7→31% (+17/+50/+25, ALL 3 domains; 360M net-negative)
      → the scaffold is a REAL, broad, size-scaling 3-8× amplifier of a small LLM. BUT
      +airfoil ≈ blind (vs-blind ≈0), captures only 42-73% of oracle → uplift is
      SEARCH-driven, not recognition-driven. Naming doesn't beat blind yet. Next lever =
      close the combo→oracle gap (v24). ORIGINAL SPEC: Prove METHOD
      transfer: the SAME propose→search→verify harness lifts a small frozen LLM across
      ≥3 DISTINCT example→program domains chosen where the airfoil is STRONG (depth-
      compositional), not breadth-hard. Domains: (1) list functions (map/filter/sort/
      rev/dedup/take-drop), (2) string transforms FlashFill-style (substring/concat/
      case/replace/split-join — literature-backed PBE), (3) numeric function induction
      ((x,y)→program, from v17's incompressible polynomials). Per domain, the v22
      three-baseline protocol: LLM-alone vs blind full-DSL search vs LLM+airfoil;
      domain-appropriate DSL but ONE harness. SHIP CRITERION: lift positive AND
      consistent across all three → not ARC-overfit → earns the HF card. Then (and only
      then) the rung-A/C "shippable airfoil" repo (harness + GGUF + GBNF grammar).
- [x] **v24 — close the recognition gap (voting + fallback). RESULT: did NOT close it.**
      Self-consistency majority/union voting + blind-fallback; best vs-blind +1.3 to +2.0
      (within ±3% noise). DIAGNOSIS from op-set sizes: the 1.7B OVER-INCLUDES (majority
      names ~20-25 of ~22-30 ops → no narrowing → combo=blind); 360M UNDER-includes
      (~3-4 → misses). The model HAS the breadth but can't SELECT which concepts are
      RELEVANT to a task — relevance-discrimination is the gap, not knowledge.
- [x] **v25 — force SELECTION. RESULT: decisive NO.** recall@2 at/below chance on 6-7-cat
      vocabularies (1.7B lists 2%, strings 6%); only 1.7B numbers (4 cats) beats chance
      (49%@2, 73%@3) but accuracy still ≈blind (precision↔depth squeeze). The model can't
      rank task-relevance — category-naming forces it through OUR invented ontology, a
      foreign interface. Knowledge present, selection absent.
- [ ] **v26 — PIVOT: LLM as PROGRAM proposer (native code).** Code-from-examples is
      in-distribution (pretraining), unlike our taxonomy. Model writes `def f(x): ...` →
      SANDBOX-exec (whitelisted builtins, no imports, timeout) → verify on train pairs →
      sample-K-and-keep. Compare vs model-alone AND blind DSL search. The unlock to test:
      a one-line `sorted(set(x))[::-1]` is a compressed deep composition blind search can't
      reach within budget → breadth (code knowledge) directly buys depth. If this beats
      blind, the uplift flips from search-driven to MODEL-driven.
      **OUTCOME: CUT mid-run** (Alex pivoted). Partial: 360M code-propose 1-2% (can't write
      valid programs for these compositions). Script kept; not a committed result.
- [x] **v27 — can the scaffold TEACH a better standalone LLM? in-context proxy. RESULT:
      NEGATIVE.** Scaffold-verified worked-examples fed in-context do NOT lift alone-accuracy
      on held-out tasks — they HURT (1.7B numbers 16.7→3.3 as M 0→8; M=0 best everywhere).
      Verified solutions of OTHER tasks don't teach THIS task's distinct rule (independent
      functions, no shared transferable skill in input→output form) + long context dilutes
      attention. Caveat: weak proxy for gradient distillation, but no positive signal and a
      principled reason for skepticism on these tasks.

## Next chapter — ARCHITECTURE, not scaffolding (Alex, 2026-05-31)
v22-v27 exhaust the "bolt something around a frozen model" levers — recognition, selection,
code-proposal, teaching-by-example all bounce off the model's INTRINSIC induction/general-
ization ceiling. Plain search+verify (no LLM) keeps winning. So the next lever is to change
what the model IS, trained in.
- [ ] **The "subconscious / incubation" layer (the live design).** A trained, non-goal-
      directed BACKGROUND recurrence (looped/latent) that recombines representations seeded
      by the current context and SURFACES useful connections — the complement of serial
      reasoning (which fixates), modeled on incubation / default-mode / sleep-replay. Key
      properties from the design convo: (1) ADDITIVE, never suppresses the obvious; (2) trades
      SIZE for TIME (extra cheap background steps ≈ horsepower a small model lacks — fits the
      north star); (3) decoupled training — background loop learns associative STRUCTURE
      (self-supervised), a SURFACING GATE learns WHEN to inject (task loss). CRUX = the gate's
      trigger (resonance / uncertainty-reduction vs surprise — OPEN, Alex deciding).
      FALSIFIABLE SIGNATURE: accuracy on remote-association tasks RISES with non-directed
      background steps, DISSOCIABLE from (and beating) equal-compute serial reasoning.
      MEASURE THE CURVE, NOT A POINT (Alex refinement 2026-05-31): sweep N for BOTH modes on
      BOTH task types; compare accuracy-vs-compute FRONTIERS. Three outcomes: (1) curves
      separate on remote, coincide on local = remote-association capability (headline);
      (2) non-directed Pareto-dominates everywhere (same acc, less compute) = EFFICIENCY win
      — KEEP, don't discard as "generic compute" (candidate cause: premature commitment is a
      compute tax directed-mode pays and non-directed doesn't); (3) frontiers coincide
      everywhere = the true null/kill. "Wins on local too" is NOT a kill by itself.
      MINIMAL TEST (CPU-feasible): tiny from-scratch transformer + background recurrence +
      gate on a synthetic "remote-bridge" task; ablate vanilla vs +background vs +gate.
      Also relates to: binding-vs-blending attention (multiplicative/tensor path), looped
      transformers, latent reasoning (Coconut), Titans surprise-memory, v13 wake-sleep
      brought INSIDE the model.

## Constraints
- Free / CPU-only. No paid APIs. No `sudo` (needs Alex's password) → no apt installs.
- GGUFs RUN locally — VERIFIED 2026-05-31: `llama-cpp-python` installs from a
  prebuilt CPU wheel (no cmake, no compile), and SmolLM2-360M-Q8 generates at
  ~72 tok/s on this Zen3 (AVX2). (The earlier "blocked: no cmake / AVX-512 crash"
  note was a STALE WINDOWS-ERA ASSUMPTION — corrected.) Real constraints: RAM
  (~7GB → small models, ≤~3-4B Q4) and, for an INNER-loop proposer, per-call
  LATENCY (a KB-sized learned policy is ~µs/call vs an LLM's tens-of-ms/token over
  thousands of calls). So a learned policy stays the inner-loop proposer; a local
  small GGUF is a real tool for outer-loop proposing, task/CoT generation, and ARC.
- Everything logged in LOG.md; anything downloaded/installed tracked in INVENTORY.md.

## Current chapter (2026-06-03): functionally-creative non-LLM ARC solver

The "subconscious / incubation" chapter above RAN — see `incubation/NOTES.md`,
executed through **step 17** (the deployable Hybrid-C end-to-end controller). Net of
that chapter: creativity = *inference-time search over a frozen causal world-model +
a structure-general value*, irreducible to reactive weights (triangulated three ways
— param-scaling, distillation, training-cycles all fail to put it in weights);
depth-scaled to depth-4; packaged as a progress-gated Hybrid-C (distilled fast path +
gated search slow path). The toy arc is closed; the open move is to make the engine
*genuinely* creative and prove it on a real benchmark.

**Goal.** Make the non-LLM creative engine ACTUALLY creative (not a gesture) and
smoke it on **ARC-AGI** — ARC-AGI-1 first, then ARC-AGI-2.

**Method — DIY-AlphaEvolve.** Agents are the *design-time* variation operator,
evolving a whole non-LLM solver. The deployed artifact = **search + small trained
nets, NO LLM at solve time**. Harness lives at `incubation/evolve/`
(`harness.py` = fitness: `solve@test` on a named ARC split, ARC 2-attempt rule +
partial credit, persists json logs; `seed_solver.py` = gen-0 = the current best-first
grid-distance DSL search).

**Creativity, defined operationally (falsifiable, no woo).** (1) an UNRESTRICTED
grasp of cause-and-effect — induce the *invariant causal mechanism*; cross-train-pair
invariance licenses it; the exact verifier = the held-out intervention — PLUS (2)
real-time INVENTION of that mechanism from experience (compose + abstract new rules,
not retrieve from a menu). The selection **GATE** measures *invented-and-generalizing*
mechanisms (they must survive an invention-OFF ablation), NOT coverage.

**Measured so far.**
- [x] **Seed baseline (gen-0) on ARC-AGI-1:** 24/400 train, **2/400 held-out eval**.
      Failure taxonomy ~93–99% "bucket A" = the DSL can't EXPRESS the rule
      (breadth/expressiveness ceiling, not search).
- [~] **Gen-1 (6 evolved operators):** roughly DOUBLED dev solve-rate and GENERALIZES
      (best, param-struct: **28/400 held-out eval = 7%, 14× the seed**). BUT honest
      attribution: the gains are generalizing **competence** (parametric concept-
      fitting); the **creative** mechanisms (experience reuse, novel multi-concept
      linking) contributed **~0**.
- [ ] **Next — a genuinely-different departure:** active **causal discovery**
      (intervention + invariance + non-directed exploration) layered on this material,
      toward a **learned grid causal world-model** — i.e. push past parametric concept-
      fitting into invented-and-generalizing mechanism, the thing the gate measures.
