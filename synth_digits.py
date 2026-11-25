"""
synth_digits.py -- synthetic "human-like" vs PRNG digit sequences.

Human bias prior: the survey chart "Pick a random number from 1-10"
(n=8604). Cite the original source of the chart in your report.

Both classes are generated per "person" so you can split train/test
BY PERSON (avoids leakage of one person's habits into both sets).

NOTE: this is SYNTHETIC data. A classifier trained on it learns this
script's assumptions, so use it for smoke tests and for a clearly
labeled "simulated" experiment, not as your main human-class results.

Usage:
    python synth_digits.py --people 40 --seqs-per-person 5 --length 20 \
        --seed 42 --out dataset.csv
"""
import argparse
import csv
import random
from collections import Counter

# Counts read off the chart (value -> respondents).
CHART_COUNTS = {0: 47, 1: 294, 2: 731, 3: 834, 4: 833, 5: 1058,
                6: 843, 7: 2405, 8: 933, 9: 460, 10: 166}


def digit_weights(ten_to_zero=True):
    """Collapse the 0-10 chart into digit weights 0-9.
    '10' is not a digit: by default it is folded into 0 (the '10' key
    on a keypad ends in 0). Use --drop-ten to discard it instead."""
    w = [0] * 10
    for value, count in CHART_COUNTS.items():
        if value == 10:
            if ten_to_zero:
                w[0] += count
        else:
            w[value] += count
    total = sum(w)
    return [x / total for x in w]


def personalize(rng, weights, sigma):
    """Each person deviates a bit from the population prior."""
    if sigma <= 0:
        return weights
    noisy = [w * rng.lognormvariate(0, sigma) for w in weights]
    total = sum(noisy)
    return [x / total for x in noisy]


def human_sequence(rng, length, weights, repeat_keep):
    """Draw digits from the biased prior.
    repeat_keep = probability that an immediate repeat is accepted.
      1.0 -> chart bias only (independent draws)
      <1  -> extra assumption (NOT from the chart): people avoid repeats."""
    seq = []
    while len(seq) < length:
        d = rng.choices(range(10), weights=weights)[0]
        if seq and d == seq[-1] and rng.random() > repeat_keep:
            continue
        seq.append(d)
    return "".join(map(str, seq))


def prng_sequence(rng, length):
    return "".join(str(rng.randrange(10)) for _ in range(length))


def summarize(name, seqs):
    digits = "".join(seqs)
    n = len(digits)
    counts = Counter(digits)
    pairs = sum(len(s) - 1 for s in seqs)
    repeats = sum(a == b for s in seqs for a, b in zip(s, s[1:]))
    expected = n / 10
    chi2 = sum((counts.get(str(d), 0) - expected) ** 2 / expected
               for d in range(10))
    freqs = " ".join(f"{d}:{counts.get(str(d), 0) / n:.3f}" for d in range(10))
    print(f"{name:6s} n={n}  repeat_rate={repeats / pairs:.3f}  "
          f"chi2(df=9)={chi2:.1f}")
    print(f"       digit freqs  {freqs}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", type=int, default=40,
                    help="number of simulated people PER class")
    ap.add_argument("--seqs-per-person", type=int, default=5)
    ap.add_argument("--length", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--person-noise", type=float, default=0.25,
                    help="lognormal sigma for person-to-person variation")
    ap.add_argument("--repeat-keep", type=float, default=1.0,
                    help="prob. of accepting an immediate repeat "
                         "(1.0 = chart only; try 0.3 to add repeat avoidance)")
    ap.add_argument("--drop-ten", action="store_true",
                    help="discard '10' instead of folding it into 0")
    ap.add_argument("--out", default="dataset.csv")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    base = digit_weights(ten_to_zero=not args.drop_ten)

    rows = []
    for p in range(args.people):
        w = personalize(rng, base, args.person_noise)
        for _ in range(args.seqs_per_person):
            rows.append((f"human_{p:03d}", "human",
                         human_sequence(rng, args.length, w, args.repeat_keep)))
    for p in range(args.people):
        for _ in range(args.seqs_per_person):
            rows.append((f"prng_{p:03d}", "prng",
                         prng_sequence(rng, args.length)))

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["person_id", "label", "sequence"])
        writer.writerows(rows)

    print(f"wrote {len(rows)} sequences to {args.out} (seed={args.seed})")
    summarize("human", [r[2] for r in rows if r[1] == "human"])
    summarize("prng", [r[2] for r in rows if r[1] == "prng"])


if __name__ == "__main__":
    main()
