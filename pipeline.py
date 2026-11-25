"""
pipeline.py -- end-to-end prototype (needs synth_digits.py in same folder)

  data -> Naive Bayes (from scratch) -> facts -> rule engine -> verdict
  baseline = best single-feature threshold rule (no chi-square needed)

Run:  python pipeline.py            (synthetic experiments, 5 seeds)
      python pipeline.py --csv real.csv   (columns: person_id,label,sequence)
"""
import argparse
import csv
import math
import random
from collections import Counter

import synth_digits as sd

HUMAN, PRNG = "human", "prng"
VOCAB_UNI = [f"d{i}" for i in range(10)]
VOCAB_DI = [f"p{a}{b}" for a in range(10) for b in range(10)]


# ------------------------------------------------------------------ data
def weak_prng_sequence(rng, length):
    """Bad PRNG: LCG mod 2^31, last decimal digit. Parity strictly alternates."""
    x, out = rng.randrange(2 ** 31), []
    for _ in range(length):
        x = (1103515245 * x + 12345) % 2 ** 31
        out.append(str(x % 10))
    return "".join(out)


def make_rows(rng, people, seqs, length, repeat_keep, uniform_humans=False,
              prng_kind="good", sigma=0.25):
    base = [0.1] * 10 if uniform_humans else sd.digit_weights()
    rows = []
    for p in range(people):
        w = sd.personalize(rng, base, sigma)
        for _ in range(seqs):
            rows.append((f"h{p}", HUMAN,
                         sd.human_sequence(rng, length, w, repeat_keep)))
    for p in range(people):
        for _ in range(seqs):
            s = (sd.prng_sequence(rng, length) if prng_kind == "good"
                 else weak_prng_sequence(rng, length))
            rows.append((f"m{p}", PRNG, s))
    return rows


def split_by_person(rows, rng, test_frac=0.3):
    """Whole people go to train OR test, stratified by class."""
    train, test = [], []
    for label in (HUMAN, PRNG):
        people = sorted({r[0] for r in rows if r[1] == label})
        rng.shuffle(people)
        test_people = set(people[:max(1, int(len(people) * test_frac))])
        for r in rows:
            if r[1] == label:
                (test if r[0] in test_people else train).append((r[2], r[1]))
    return train, test


# --------------------------------------------------------- Naive Bayes
def tokens(seq, digrams=True):
    t = ["d" + c for c in seq]
    if digrams:
        t += ["p" + a + b for a, b in zip(seq, seq[1:])]
    return t


class NaiveBayes:
    def __init__(self, digrams=True, alpha=1.0):
        self.digrams, self.alpha = digrams, alpha

    def fit(self, data):
        vocab = VOCAB_UNI + (VOCAB_DI if self.digrams else [])
        self.logprior, self.loglik = {}, {}
        for c in (HUMAN, PRNG):
            seqs = [s for s, l in data if l == c]
            self.logprior[c] = math.log(len(seqs) / len(data))
            cnt = Counter(t for s in seqs for t in tokens(s, self.digrams))
            total = sum(cnt.values())
            denom = total + self.alpha * len(vocab)
            self.loglik[c] = {t: math.log((cnt[t] + self.alpha) / denom)
                              for t in vocab}
        return self

    def p_human(self, seq):
        sc = {c: self.logprior[c] + sum(self.loglik[c][t]
                                        for t in tokens(seq, self.digrams))
              for c in (HUMAN, PRNG)}
        m = max(sc.values())
        e = {c: math.exp(v - m) for c, v in sc.items()}
        return e[HUMAN] / (e[HUMAN] + e[PRNG])

    def predict(self, seq):
        return HUMAN if self.p_human(seq) >= 0.5 else PRNG


# ----------------------------------------------------- features/baseline
def seq_features(s):
    n = len(s)
    rep = sum(a == b for a, b in zip(s, s[1:])) / (n - 1)
    share = max(Counter(s).values()) / n
    zig = sum((a < b > c) or (a > b < c)
              for a, b, c in zip(s, s[1:], s[2:])) / (n - 2)
    return rep, share, zig


class ThresholdBaseline:
    """Best single-feature threshold rule, tuned on training data."""
    NAMES = ["repeat_rate", "max_digit_share", "zigzag_rate"]

    def fit(self, data):
        F = [seq_features(s) for s, _ in data]
        best = (-1, None)
        for i in range(3):
            vals = sorted(set(f[i] for f in F))
            step = max(1, len(vals) // 40)
            for thr in vals[::step]:
                for sign in (1, -1):
                    acc = sum(((sign * (f[i] - thr)) > 0) == (l == HUMAN)
                              for f, (_, l) in zip(F, data)) / len(data)
                    if acc > best[0]:
                        best = (acc, (i, thr, sign))
        self.i, self.thr, self.sign = best[1]
        return self

    def predict(self, seq):
        return HUMAN if self.sign * (seq_features(seq)[self.i] - self.thr) > 0 \
            else PRNG

    def describe(self):
        op = ">" if self.sign > 0 else "<"
        return f"human if {self.NAMES[self.i]} {op} {self.thr:.3f}"


# ------------------------------------------------------- rule engine
# (layer, name, must-have facts, must-NOT-have facts, conclusion)
# 'NOT' only refers to facts from lower layers, so each layer is a fixpoint.
RULES = [
    (1, "E1", ["avoids_repeats"], [], "some_evidence"),
    (1, "E2", ["digit_favoritism"], [], "some_evidence"),
    (1, "E3", ["over_alternates"], [], "some_evidence"),
    (1, "E4", ["avoids_repeats", "digit_favoritism"], [], "strong_evidence"),
    (1, "E5", ["avoids_repeats", "over_alternates"], [], "strong_evidence"),
    (1, "E6", ["digit_favoritism", "over_alternates"], [], "strong_evidence"),
    (2, "R1", ["length_short"], [], "weak_sample"),
    (2, "R2", ["weak_sample", "posterior_high"], [], "unreliable_confidence"),
    (3, "C1", ["posterior_high", "strong_evidence"], ["weak_sample"],
     "likely_human_made"),
    (3, "C2", ["posterior_high", "some_evidence"],
     ["strong_evidence", "weak_sample"], "possible_human_made"),
    (3, "C3", ["posterior_high"], ["some_evidence", "weak_sample"],
     "conflict_review"),
    (3, "C4", ["posterior_low", "strong_evidence"], ["weak_sample"],
     "conflict_review"),
    (3, "C5", ["posterior_low"], ["strong_evidence", "weak_sample"],
     "likely_machine_random"),
    (3, "C6", ["weak_sample"], [], "inconclusive"),
    (3, "C7", ["uncertain"], [], "inconclusive"),
    (4, "A1", ["likely_human_made"], [], "action_flag_for_review"),
    (4, "A2", ["possible_human_made"], [], "action_flag_for_review"),
    (4, "A3", ["inconclusive"], [], "action_collect_more_entries"),
    (4, "A4", ["conflict_review"], [], "action_manual_check"),
]
VERDICTS = ["likely_human_made", "possible_human_made", "likely_machine_random",
            "conflict_review", "inconclusive"]


def run_engine(facts):
    facts, trace = set(facts), []
    for layer in sorted({r[0] for r in RULES}):
        changed = True
        while changed:
            changed = False
            for ly, name, req, forb, concl in RULES:
                if (ly == layer and concl not in facts
                        and all(r in facts for r in req)
                        and not any(f in facts for f in forb)):
                    facts.add(concl)
                    trace.append(f"{name}: {' & '.join(req)}"
                                 + (f" & not({','.join(forb)})" if forb else "")
                                 + f" => {concl}")
                    changed = True
    return facts, trace


def pct(vals, q):
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]


class RuleLayer:
    """Thresholds come from PRNG training data (10% tails), not guesses."""

    def fit(self, train):
        F = [seq_features(s) for s, l in train if l == PRNG]
        self.rep_lo = pct([f[0] for f in F], 0.10)
        self.share_hi = pct([f[1] for f in F], 0.90)
        self.zig_hi = pct([f[2] for f in F], 0.90)
        return self

    def facts(self, seq, p_human):
        rep, share, zig = seq_features(seq)
        f = set()
        if p_human >= 0.8: f.add("posterior_high")
        elif p_human <= 0.2: f.add("posterior_low")
        else: f.add("uncertain")
        if len(seq) < 10: f.add("length_short")
        if rep <= self.rep_lo: f.add("avoids_repeats")
        if share >= self.share_hi: f.add("digit_favoritism")
        if zig >= self.zig_hi: f.add("over_alternates")
        return f

    def verdict(self, seq, p_human):
        facts, trace = run_engine(self.facts(seq, p_human))
        v = next((x for x in VERDICTS if x in facts), "inconclusive")
        return v, trace


# --------------------------------------------------------- evaluation
def confusion(y, p):
    tp = sum(a == HUMAN and b == HUMAN for a, b in zip(y, p))
    fn = sum(a == HUMAN and b == PRNG for a, b in zip(y, p))
    fp = sum(a == PRNG and b == HUMAN for a, b in zip(y, p))
    tn = sum(a == PRNG and b == PRNG for a, b in zip(y, p))
    return tp, fn, fp, tn


def acc(c):
    return (c[0] + c[3]) / max(1, sum(c))


def evaluate(train, test):
    nb = NaiveBayes(True).fit(train)
    nb_uni = NaiveBayes(False).fit(train)
    base = ThresholdBaseline().fit(train)
    rl = RuleLayer().fit(train)
    y = [l for _, l in test]
    out = {"nb": confusion(y, [nb.predict(s) for s, _ in test]),
           "nb_uni": confusion(y, [nb_uni.predict(s) for s, _ in test]),
           "base": confusion(y, [base.predict(s) for s, _ in test])}
    dec_y, dec_p, n_dec = [], [], 0
    for s, l in test:
        v, _ = rl.verdict(s, nb.p_human(s))
        if v in ("likely_human_made", "possible_human_made"):
            dec_y.append(l); dec_p.append(HUMAN)
        elif v == "likely_machine_random":
            dec_y.append(l); dec_p.append(PRNG)
    out["rules"] = confusion(dec_y, dec_p)
    out["coverage"] = len(dec_y) / len(test)
    out["base_rule"] = base.describe()
    return out, (nb, base, rl)


def add(c1, c2):
    return tuple(a + b for a, b in zip(c1, c2))


def run_grid(seeds, people, seqs):
    grid = [(f"E{i + 1 + (0 if rk == 1.0 else 5)}", L, rk)
            for rk in (1.0, 0.3) for i, L in enumerate((4, 8, 12, 20, 40))]
    print(f"{'exp':4s} {'len':>3s} {'repeat_keep':>11s} | {'NB uni+di':>9s} "
          f"{'NB uni':>7s} {'baseline':>8s} | {'rules acc':>9s} {'coverage':>8s}")
    totals = {}
    for name, L, rk in grid:
        agg = {k: (0, 0, 0, 0) for k in ("nb", "nb_uni", "base", "rules")}
        cov = 0
        for sd_ in range(seeds):
            rng = random.Random(1000 * sd_ + L)
            rows = make_rows(rng, people, seqs, L, rk)
            res, _ = evaluate(*split_by_person(rows, rng))
            for k in agg:
                agg[k] = add(agg[k], res[k])
            cov += res["coverage"] / seeds
        totals[name] = agg
        rules_acc = f"{acc(agg['rules']):9.3f}" if cov > 0 else f"{'n/a':>9s}"
        print(f"{name:4s} {L:3d} {rk:11.1f} | {acc(agg['nb']):9.3f} "
              f"{acc(agg['nb_uni']):7.3f} {acc(agg['base']):8.3f} | "
              f"{rules_acc} {cov:8.2f}")
    return totals


def run_failure_cases(seeds, people, seqs):
    """Train on the standard condition, test on shifted/bad conditions."""
    print("\nFAILURE / SHIFT CASES (train: length 20, repeat_keep 0.3)")
    cases = [
        ("F1 humans try to be uniform", dict(uniform_humans=True, repeat_keep=0.3)),
        ("F2 weak LCG instead of good PRNG", dict(prng_kind="weak", repeat_keep=0.3)),
        ("F3 humans never avoid repeats", dict(repeat_keep=1.0)),
    ]
    print(f"{'case':34s} {'NB':>6s} {'baseline':>8s} {'rules acc':>9s}")
    for name, kw in cases:
        agg = {k: (0, 0, 0, 0) for k in ("nb", "base", "rules")}
        for sd_ in range(seeds):
            rng = random.Random(7000 + sd_)
            tr_rows = make_rows(rng, people, seqs, 20, 0.3)
            train, _ = split_by_person(tr_rows, rng)
            nb = NaiveBayes(True).fit(train)
            base = ThresholdBaseline().fit(train)
            rl = RuleLayer().fit(train)
            args = dict(repeat_keep=0.3); args.update(kw)
            te_rows = make_rows(rng, people, seqs, 20, **args)
            _, test = split_by_person(te_rows, rng, test_frac=0.5)
            y = [l for _, l in test]
            agg["nb"] = add(agg["nb"], confusion(y, [nb.predict(s) for s, _ in test]))
            agg["base"] = add(agg["base"], confusion(y, [base.predict(s) for s, _ in test]))
            dy, dp = [], []
            for s, l in test:
                v, _ = rl.verdict(s, nb.p_human(s))
                if v in ("likely_human_made", "possible_human_made"):
                    dy.append(l); dp.append(HUMAN)
                elif v == "likely_machine_random":
                    dy.append(l); dp.append(PRNG)
            agg["rules"] = add(agg["rules"], confusion(dy, dp))
        print(f"{name:34s} {acc(agg['nb']):6.3f} {acc(agg['base']):8.3f} "
              f"{acc(agg['rules']):9.3f}   (NB human recall "
              f"{agg['nb'][0] / max(1, agg['nb'][0] + agg['nb'][1]):.2f}, "
              f"machine recall {agg['nb'][3] / max(1, agg['nb'][2] + agg['nb'][3]):.2f})")


def demo(rng):
    print("\nDEMO: derived verdict with rule trace")
    train, _ = split_by_person(make_rows(rng, 60, 5, 20, 0.3), rng)
    nb, rl = NaiveBayes(True).fit(train), RuleLayer().fit(train)
    for label, s in [("your 28 digits", "1928729356215297825739498678"),
                     ("fresh PRNG", sd.prng_sequence(rng, 20))]:
        p = nb.p_human(s)
        v, trace = rl.verdict(s, p)
        print(f"\n[{label}] P(human)={p:.3f} -> {v}")
        for t in trace:
            print("   ", t)


def run_csv(path):
    rows = [(r["person_id"], r["label"], r["sequence"])
            for r in csv.DictReader(open(path))]
    rng = random.Random(0)
    res, (_, base, _) = evaluate(*split_by_person(rows, rng))
    print(f"NB acc={acc(res['nb']):.3f}  baseline acc={acc(res['base']):.3f} "
          f"({res['base_rule']})  confusion(tp,fn,fp,tn)={res['nb']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--people", type=int, default=60)
    ap.add_argument("--seqs-per-person", type=int, default=5)
    ap.add_argument("--csv")
    a = ap.parse_args()
    if a.csv:
        run_csv(a.csv)
    else:
        t = run_grid(a.seeds, a.people, a.seqs_per_person)
        for k in ("E4", "E9"):
            tp, fn, fp, tn = t[k]["nb"]
            print(f"\nNB confusion {k} (summed over seeds): "
                  f"TP={tp} FN={fn} FP={fp} TN={tn}")
        run_failure_cases(a.seeds, a.people, a.seqs_per_person)
        demo(random.Random(123))
