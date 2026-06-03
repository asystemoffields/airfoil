#!/usr/bin/env python3
"""
Incubation line — STEP 0: build the task and VERIFY the fixation property.

Claim we're eventually testing (no brain words): non-goal-directed recurrent compute
solves problems needing a REMOTE association that equal goal-directed compute can't.
Before any of that, the task must actually induce FIXATION: a vanilla model must fail
by following the salient LOCAL pattern and ignoring DISTANT-but-decisive evidence — and
fail MORE as the decisive evidence gets farther away. If it doesn't, the task is no good.

Task RULESWITCH (sequence classification, predict the answer digit):
  [MARK]  x1->lure(x1)  x2->lure(x2) ... xF->lure(xF)  [QUERY] q   =>  answer
  - MARK (pos 0, the DISTANT decisive token) sets the TRUE rule: A => +1, B => +3 (mod 10).
  - The F local example pairs all demonstrate the OTHER (lure) rule -> a salient local
    pattern screaming "use the lure rule".
  - The query q is where the two rules DISAGREE; correct answer = true_rule(q), the lure
    answer = lure_rule(q). A model that rides the local pattern emits the lure (wrong).
  - F (filler length) = the DISTANCE from the decisive MARK to the query.

Fixation is confirmed if: accuracy falls toward chance/lure as F grows, AND the errors are
dominated by the lure answer (the model is following the local pattern, not guessing).

Vanilla tiny transformer, trained from scratch on CPU. Run with /data/llm/.venv/bin/python.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)

# vocab: 0..9 digits; specials
PAD, MARK_A, MARK_B, ARROW, SEP, QUERY = 10, 11, 12, 13, 14, 15
VOCAB = 16
RULES = {MARK_A: 1, MARK_B: 3}        # A: +1, B: +3  (mod 10)
FMAX = 12
SEQLEN = 2 + FMAX * 4 + 2             # MARK + F*(x ARROW y SEP) + QUERY q
DEVICE = "cpu"


def make_batch(bs, f_lo=2, f_hi=FMAX, gen=None):
    g = gen or torch.Generator().manual_seed(torch.randint(0, 1 << 30, (1,)).item())
    seqs = torch.full((bs, SEQLEN), PAD, dtype=torch.long)
    ans = torch.zeros(bs, dtype=torch.long)
    Fs = torch.zeros(bs, dtype=torch.long)
    for i in range(bs):
        true_mark = MARK_A if torch.rand(1, generator=g).item() < 0.5 else MARK_B
        lure_mark = MARK_B if true_mark == MARK_A else MARK_A
        tr, lr = RULES[true_mark], RULES[lure_mark]
        F = int(torch.randint(f_lo, f_hi + 1, (1,), generator=g).item())
        Fs[i] = F
        p = 0
        seqs[i, p] = true_mark; p += 1
        for _ in range(F):
            x = int(torch.randint(0, 10, (1,), generator=g).item())
            seqs[i, p] = x; seqs[i, p + 1] = ARROW
            seqs[i, p + 2] = (x + lr) % 10            # local examples follow the LURE rule
            seqs[i, p + 3] = SEP; p += 4
        q = int(torch.randint(0, 10, (1,), generator=g).item())
        seqs[i, p] = QUERY; seqs[i, p + 1] = q; p += 2
        ans[i] = (q + tr) % 10                         # correct = TRUE rule (the distant mark)
    return seqs.to(DEVICE), ans.to(DEVICE), Fs


def lure_answer(seq):
    """The answer a local-pattern follower would give (other rule applied to the query)."""
    q = seq[-1].item()
    mark = seq[0].item()
    lure_rule = RULES[MARK_B] if mark == MARK_A else RULES[MARK_A]
    return (q + lure_rule) % 10


class TinyTransformer(nn.Module):
    def __init__(self, d=96, heads=4, layers=3):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(SEQLEN, d)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d, batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d, 10)

    def forward(self, x):
        h = self.emb(x) + self.pos(torch.arange(x.size(1), device=x.device))
        h = self.enc(h)
        return self.head(h[:, -1])      # read out at the query-answer position


def main():
    flush = lambda *a: print(*a, flush=True)
    flush("=" * 72)
    flush("Incubation step 0 — does RULESWITCH induce FIXATION? (vanilla tiny transformer)")
    flush("=" * 72)
    model = TinyTransformer().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    lossf = nn.CrossEntropyLoss()
    n = sum(p.numel() for p in model.parameters())
    flush(f"  params={n/1e3:.0f}K  seqlen={SEQLEN}  train F~[2,{FMAX}]")

    model.train()
    for step in range(4000):
        x, y, _ = make_batch(128)
        opt.zero_grad()
        loss = lossf(model(x), y)
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            flush(f"  step {step:4d}  loss {loss.item():.3f}")

    # eval: accuracy and lure-fraction per filler-distance F
    model.eval()
    flush("\n  F (dist)  acc%   among-errors: %=lure   (chance=10%)")
    with torch.no_grad():
        for F in [2, 4, 6, 8, 10, 12]:
            x, y, _ = make_batch(600, f_lo=F, f_hi=F)
            pred = model(x).argmax(1)
            acc = (pred == y).float().mean().item()
            err = pred != y
            if err.sum() > 0:
                lures = torch.tensor([lure_answer(x[i].cpu()) for i in range(x.size(0))], device=DEVICE)
                lure_frac = (pred[err] == lures[err]).float().mean().item()
            else:
                lure_frac = 0.0
            flush(f"     {F:2d}     {acc*100:5.1f}      {lure_frac*100:5.1f}")

    flush("\n  FIXATION CONFIRMED if acc falls toward ~10% as F grows AND errors are")
    flush("  dominated by the lure (high %=lure) — i.e. the model rides the local pattern")
    flush("  and ignores the distant decisive mark. That makes RULESWITCH a valid substrate.")


if __name__ == "__main__":
    main()
