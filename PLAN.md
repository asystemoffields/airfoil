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
- [ ] **v10** wire a diverse-verifier consensus gate into the actual library loop:
      does it prevent the noisy-verifier library POISONING (compressing wrong
      solutions into garbage abstractions)? The first end-to-end loop test.
- [ ] **v10** parameterized (antiunification) fragments EARN their keep; later,
      graduate toward an abstraction benchmark (mini-ARC subset).

## Constraints
- Free / CPU-only. No paid APIs. No `sudo` (needs Alex's password) → no apt installs.
- GGUFs ARE pullable (network confirmed); SmolLM2-360M / PMRA-1.7B available.
  But running them locally is blocked: `cmake` missing → can't build llama.cpp,
  and the pip llama-cpp-python wheel is AVX-512 (crashes this Zen3). So the
  symbolic core is the focus; LLM-as-proposer is deferred to when/if v6 needs it.
- Everything logged in LOG.md; anything downloaded/installed tracked in INVENTORY.md.
