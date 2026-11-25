"""
two_stage_detector.py

Two-Stage Fake (Human-Generated) vs. Real (Uniform Random) 4-Digit Detector:
  - Stage 1: Positional + Adjacent-Digram Naive Bayes (trained strictly on real human data)
  - Stage 2: Shannon Entropy & Cross-Entropy Deviations (Digit bias, 1..4 repeat multiplicity, Benford's Law)
  - Fusion : Gaussian likelihoods P(x | Fake) and P(x | Real) combined via the Law of Total Probability
"""

import math
import random
from collections import Counter

# Actual human data ONLY (4 rows x 10 numbers = 40 four-digit numbers)
HUMAN_ROWS = [
    ["2345", "8645", "6432", "1853", "5319", "7601", "3427", "6459", "1394", "8271"],
    ["1004", "8030", "3030", "1234", "2004", "1974", "1972", "4010", "2013", "2026"],
    ["1352", "1673", "7640", "6754", "2421", "3452", "9068", "8923", "5623", "7845"],
    ["3791", "2916", "1062", "9971", "2751", "1023", "5495", "9205", "9022", "6391"],
]
HUMAN_DATA = [num for row in HUMAN_ROWS for num in row]

DIGITS = "0123456789"
FIRST_DIGITS = "123456789"
MULTS = [1, 2, 3, 4]  # Max count of any repeated digit in a 4-digit number

UNIF_DIGIT = {d: 0.1 for d in DIGITS}
BENFORD = {str(d): math.log10(1.0 + 1.0 / d) for d in range(1, 10)}

# Exact multiplicity distribution for uniform 4-digit integers in 1000..9999 (9000 total)
RAND_MULT = {1: 4536 / 9000, 2: 4131 / 9000, 3: 324 / 9000, 4: 9 / 9000}


def max_multiplicity(num_str: str) -> int:
    return max(Counter(num_str).values())


def empirical_dist(items, keys, alpha: float = 0.0) -> dict:
    counts = Counter(items)
    denom = len(items) + alpha * len(keys)
    return {k: (counts[k] + alpha) / denom for k in keys}


def shannon_entropy(p: dict) -> float:
    return -sum(v * math.log2(v) for v in p.values() if v > 0)


def cross_entropy(p: dict, q: dict) -> float:
    return -sum(p[k] * math.log2(q[k]) for k in p if p[k] > 0)


def normal_pdf(x: float, mu: float, sigma: float) -> float:
    sigma = max(sigma, 1e-6)
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def mean_std(values) -> tuple[float, float]:
    mu = sum(values) / len(values)
    var = sum((v - mu) ** 2 for v in values) / max(1, len(values) - 1)
    return mu, math.sqrt(var)


# ==============================================================================
# STAGE 1: Positional + Digram Naive Bayes (Human vs. Exact Uniform 1000..9999)
# ==============================================================================
class Stage1NaiveBayes:
    def __init__(self, alpha: float = 1.0, digram_weight: float = 0.5):
        self.alpha = alpha
        self.digram_weight = digram_weight

    def fit(self, human_seqs: list[str]):
        self.pos_counts = [Counter() for _ in range(4)]
        self.di_counts = Counter()
        for s in human_seqs:
            for pos, ch in enumerate(s):
                self.pos_counts[pos][ch] += 1
            for a, b in zip(s, s[1:]):
                self.di_counts[a + b] += 1
        self.n_seqs = len(human_seqs)
        self.n_di = self.n_seqs * 3
        return self

    def sequence_llr(self, s: str) -> float:
        """Log2 likelihood ratio: log2 P(s | Human_NB) - log2 P(s | Uniform)."""
        lp_human, lp_rand = 0.0, 0.0
        for pos, ch in enumerate(s):
            k_choices = 9 if pos == 0 else 10  # Leading digit is 1..9; others 0..9
            p_h = (self.pos_counts[pos][ch] + self.alpha) / (self.n_seqs + self.alpha * k_choices)
            p_r = 1.0 / k_choices
            lp_human += math.log2(p_h)
            lp_rand += math.log2(p_r)

        for a, b in zip(s, s[1:]):
            p_h_di = (self.di_counts[a + b] + 0.5) / (self.n_di + 0.5 * 100)
            p_r_di = 0.01
            lp_human += self.digram_weight * math.log2(p_h_di)
            lp_rand += self.digram_weight * math.log2(p_r_di)

        return lp_human - lp_rand

    def batch_score(self, batch: list[str]) -> float:
        return sum(self.sequence_llr(s) for s in batch) / len(batch)

    def predict_proba_fake(self, batch: list[str], prior_fake: float = 0.5) -> float:
        total_llr = sum(self.sequence_llr(s) for s in batch)
        log_odds = total_llr + math.log2(prior_fake / (1.0 - prior_fake))
        log_odds = max(-50.0, min(50.0, log_odds))
        return 1.0 / (1.0 + 2.0 ** (-log_odds))


# ==============================================================================
# STAGE 2 + FUSION: Cross-Entropy Deviations + Law of Total Probability
# ==============================================================================
class TwoStageFakeDetector:
    """
    Combines Stage 1 (Naive Bayes LLR) and Stage 2 (Cross-Entropy Deviations)
    using Normal likelihoods and the Law of Total Probability:
        P(x) = P(x | Fake) * P(Fake) + P(x | Real) * P(Real)
        P(Fake | x) = [P(x | Fake) * P(Fake)] / P(x)
    """

    def __init__(self, prior_fake: float = 0.5, active_indices: tuple[int, ...] = (0, 1, 2, 3)):
        self.prior_fake = prior_fake
        self.prior_real = 1.0 - prior_fake
        # Feature indices: 0=Stage1_NB, 1=Digit_CE, 2=Multiplicity_CE, 3=Benford_CE
        self.active_indices = active_indices

    @staticmethod
    def _compute_vector(
        batch: list[str],
        nb_model: Stage1NaiveBayes,
        q_human_digit: dict,
        q_human_mult: dict,
    ) -> list[float]:
        # Stage 1 feature: Mean Naive Bayes log-likelihood ratio
        s1_nb = nb_model.batch_score(batch)

        # Stage 2a: Digit cross-entropy deviation H(P, Q_human) - H(P, Uniform)
        p_dig = empirical_dist("".join(batch), DIGITS, alpha=0.0)
        dev_digit = cross_entropy(p_dig, q_human_digit) - math.log2(10)

        # Stage 2b: Same-digit multiplicity (1..4 repeats) cross-entropy deviation
        p_mult = empirical_dist([max_multiplicity(s) for s in batch], MULTS, alpha=0.0)
        dev_mult = cross_entropy(p_mult, q_human_mult) - cross_entropy(p_mult, RAND_MULT)

        # Stage 2c: Benford's Law leading-digit cross-entropy deviation
        p_first = empirical_dist([s[0] for s in batch], FIRST_DIGITS, alpha=0.0)
        dev_benford = cross_entropy(p_first, BENFORD) - math.log2(9)

        return [s1_nb, dev_digit, dev_mult, dev_benford]

    def fit(self, human_train_batches: list[list[str]], rng: random.Random, null_samples: int = 500):
        batch_size = len(human_train_batches[0])
        flat_human = [num for b in human_train_batches for num in b]

        # Fit final Stage 1 NB and Stage 2 reference distributions on full human training split
        self.nb = Stage1NaiveBayes(alpha=1.0).fit(flat_human)
        self.q_human_digit = empirical_dist("".join(flat_human), DIGITS, alpha=1.0)
        self.q_human_mult = empirical_dist([max_multiplicity(s) for s in flat_human], MULTS, alpha=0.5)

        # Fit P(x | Fake) Gaussians via internal leave-one-batch-out so NB/CE scores aren't overfit
        fake_matrix = []
        for idx_b, target_batch in enumerate(human_train_batches):
            sub_flat = [num for j, b in enumerate(human_train_batches) if j != idx_b for num in b]
            sub_nb = Stage1NaiveBayes(alpha=1.0).fit(sub_flat)
            sub_q_dig = empirical_dist("".join(sub_flat), DIGITS, alpha=1.0)
            sub_q_mult = empirical_dist([max_multiplicity(s) for s in sub_flat], MULTS, alpha=0.5)
            fake_matrix.append(self._compute_vector(target_batch, sub_nb, sub_q_dig, sub_q_mult))

        self.fake_gaussians = [mean_std(col) for col in zip(*fake_matrix)]

        # Fit P(x | Real) Gaussians on true uniform random batches
        real_batches = [
            [str(rng.randint(1000, 9999)) for _ in range(batch_size)]
            for _ in range(null_samples)
        ]
        real_matrix = [
            self._compute_vector(b, self.nb, self.q_human_digit, self.q_human_mult)
            for b in real_batches
        ]
        self.real_gaussians = [mean_std(col) for col in zip(*real_matrix)]
        return self

    def predict_proba_fake(self, batch: list[str]) -> float:
        x = self._compute_vector(batch, self.nb, self.q_human_digit, self.q_human_mult)

        log_lik_fake = sum(
            math.log(max(normal_pdf(x[i], self.fake_gaussians[i][0], self.fake_gaussians[i][1]), 1e-300))
            for i in self.active_indices
        )
        log_lik_real = sum(
            math.log(max(normal_pdf(x[i], self.real_gaussians[i][0], self.real_gaussians[i][1]), 1e-300))
            for i in self.active_indices
        )

        # Law of Total Probability: P(x) = P(x|Fake)P(Fake) + P(x|Real)P(Real)
        a = log_lik_fake + math.log(self.prior_fake)
        b = log_lik_real + math.log(self.prior_real)
        m = max(a, b)
        p_x_and_fake = math.exp(a - m)
        p_x_total = p_x_and_fake + math.exp(b - m)
        return p_x_and_fake / p_x_total

    def inspect(self, batch: list[str]) -> dict:
        x = self._compute_vector(batch, self.nb, self.q_human_digit, self.q_human_mult)
        return {
            "stage1_nb_prob": round(self.nb.predict_proba_fake(batch, self.prior_fake), 4),
            "stage1_nb_llr": round(x[0], 4),
            "stage2_digit_ce_dev": round(x[1], 4),
            "stage2_mult_ce_dev": round(x[2], 4),
            "stage2_benford_ce_dev": round(x[3], 4),
            "final_p_fake": round(self.predict_proba_fake(batch), 4),
        }


# ==============================================================================
# K-Fold Cross-Validation Benchmark (Stage 1 vs. Stage 2 vs. Combined)
# ==============================================================================
def benchmark_kfold(batch_size: int = 5, k_folds: int = 4, n_seeds: int = 20):
    acc_s1, acc_s2, acc_both = [], [], []

    for seed in range(n_seeds):
        rng = random.Random(500 + seed)
        h_batches = [HUMAN_DATA[i : i + batch_size] for i in range(0, len(HUMAN_DATA), batch_size)]
        n = len(h_batches)
        r_batches = [[str(rng.randint(1000, 9999)) for _ in range(batch_size)] for _ in range(n)]

        idx = list(range(n))
        rng.shuffle(idx)
        folds = [idx[i::k_folds] for i in range(k_folds)]

        c1 = c2 = cb = total = 0
        for f in range(k_folds):
            tr_h = [h_batches[i] for j in range(k_folds) if j != f for i in folds[j]]
            te_h = [h_batches[i] for i in folds[f]]
            te_r = [r_batches[i] for i in folds[f]]

            # Fit once per fold, evaluate Stage 1 only (0,), Stage 2 only (1,2,3), and Both (0,1,2,3)
            model = TwoStageFakeDetector(prior_fake=0.5, active_indices=(0, 1, 2, 3)).fit(tr_h, rng)

            for true_label, test_batches in [(1, te_h), (0, te_r)]:
                for b in test_batches:
                    model.active_indices = (0,)
                    c1 += int((model.predict_proba_fake(b) >= 0.5) == true_label)

                    model.active_indices = (1, 2, 3)
                    c2 += int((model.predict_proba_fake(b) >= 0.5) == true_label)

                    model.active_indices = (0, 1, 2, 3)
                    cb += int((model.predict_proba_fake(b) >= 0.5) == true_label)
                    total += 1

        acc_s1.append(c1 / total)
        acc_s2.append(c2 / total)
        acc_both.append(cb / total)

    return sum(acc_s1) / n_seeds, sum(acc_s2) / n_seeds, sum(acc_both) / n_seeds


if __name__ == "__main__":
    print("=== Two-Stage K-Fold Cross-Validation (20 Seeds, Real Human Data Only) ===")
    for bs, kf in [(1, 5), (5, 4), (10, 4)]:
        s1, s2, both = benchmark_kfold(batch_size=bs, k_folds=kf, n_seeds=20)
        print(
            f"Batch size = {bs:2d} ({kf}-fold): "
            f"Stage 1 NB = {s1:.3f} | Stage 2 CE = {s2:.3f} | Combined P(Fake) = {both:.3f}"
        )

    # Full-model inspection on a human-like batch vs. a random batch
    rng_demo = random.Random(99)
    detector = TwoStageFakeDetector(prior_fake=0.5).fit(
        [HUMAN_DATA[i : i + 5] for i in range(0, 40, 5)], rng_demo
    )

    human_like = ["1984", "2019", "4729", "8361", "5104"]
    true_random = [str(rng_demo.randint(1000, 9999)) for _ in range(5)]

    print("\n=== Stage-by-Stage Inspection ===")
    print("Suspect batch:", human_like)
    print("  ->", detector.inspect(human_like))
    print("Random batch :", true_random)
    print("  ->", detector.inspect(true_random))