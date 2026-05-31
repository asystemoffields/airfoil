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
- [ ] **v18** a learned proposer in the loop (v6's bigram → context-conditioned /
      small frozen LLM); then a real EXTERNAL benchmark (mini-ARC), distribution not
      ours. Co-driven.

## Constraints
- Free / CPU-only. No paid APIs. No `sudo` (needs Alex's password) → no apt installs.
- GGUFs ARE pullable (network confirmed); SmolLM2-360M / PMRA-1.7B available.
  But running them locally is blocked: `cmake` missing → can't build llama.cpp,
  and the pip llama-cpp-python wheel is AVX-512 (crashes this Zen3). So the
  symbolic core is the focus; LLM-as-proposer is deferred to when/if v6 needs it.
- Everything logged in LOG.md; anything downloaded/installed tracked in INVENTORY.md.
