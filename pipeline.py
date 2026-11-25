"""
joint_batch_detector.py

Computes P(All k numbers in batch are Fake | x_1, ..., x_k) using:
  - Stage 1: Positional + Adjacent-Digram Naive Bayes (trained on real human numbers)
  - Stage 2: Item-level Shannon/Cross-Entropy deviations (digit bias, 1..4 repeat multiplicity, Benford)
  - Joint Batch Likelihood: Product of P(x_i | Fake) across all k items in the batch,
    combined via Bayes' Theorem and the Law of Total Probability.
"""

import math
import random
from collections import Counter

HUMAN_ROWS = [
    ["2345", "8645", "6432", "1853", "5319", "7601", "3427", "6459", "1394", "8271"],
    ["1004", "8030", "3030", "1234", "2004", "1974", "1972", "4010", "2013", "2026"],
    ["1352", "1673", "7640", "6754", "2421", "3452", "9068", "8923", "5623", "7845"],
    ["3791", "2916", "1062", "9971", "2751", "1023", "5495", "9205", "9022", "6391"],
]
HUMAN_DATA = [num for row in HUMAN_ROWS for num in row]

DIGITS = "0123456789"
FIRST_DIGITS = "123456789"
MULTS = [1, 2, 3, 4]
BENFORD = {str(d): math.log10(1.0 + 1.0 / d) for d in range(1, 10)}
RAND_MULT = {1: 4536 / 9000, 2: 4131 / 9000, 3: 324 / 9000, 4: 9 / 9000}


def max_multiplicity(s: str) -> int:
    return max(Counter(s).values())


def empirical_dist(items, keys, alpha: float = 0.0) -> dict:
    c = Counter(items)
    d = len(items) + alpha * len(keys)
    return {k: (c[k] + alpha) / d for k in keys}


def cross_entropy(p: dict, q: dict) -> float:
    return -sum(p[k] * math.log2(q[k]) for k in p if p[k] > 0)


def normal_pdf(x: float, mu: float, sigma: float) -> float:
    sigma = max(sigma, 1e-6)
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def mean_std(vals) -> tuple[float, float]:
    mu = sum(vals) / len(vals)
    var = sum((v - mu) ** 2 for v in vals) / max(1, len(vals) - 1)
    return mu, math.sqrt(var)


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
        lp_h, lp_r = 0.0, 0.0
        for pos, ch in enumerate(s):
            k = 9 if pos == 0 else 10
            lp_h += math.log2((self.pos_counts[pos][ch] + self.alpha) / (self.n_seqs + self.alpha * k))
            lp_r += math.log2(1.0 / k)
        for a, b in zip(s, s[1:]):
            lp_h += self.digram_weight * math.log2((self.di_counts[a + b] + 0.5) / (self.n_di + 0.5 * 100))
            lp_r += self.digram_weight * math.log2(0.01)
        return lp_h - lp_r


class JointBatchFakeDetector:
    """
    Evaluates the joint probability that an ENTIRE batch [x_1, ..., x_k] is Fake:
        P(Batch = Fake | x_1..x_k) =
            [P(Fake) * prod_i P(x_i | Fake)] / [P(Fake) * prod_i P(x_i | Fake) + P(Real) * prod_i P(x_i | Real)]
    """

    def __init__(self, prior_fake: float = 0.5):
        self.prior_fake = prior_fake
        self.prior_real = 1.0 - prior_fake

    @staticmethod
    def _item_features(s: str, nb_model: Stage1NaiveBayes, q_dig: dict, q_mult: dict) -> list[float]:
        # Stage 1: Positional + digram Naive Bayes log-likelihood ratio
        s1_nb = nb_model.sequence_llr(s)

        # Stage 2: Cross-entropy deviations for item s
        p_dig = empirical_dist(s, DIGITS, alpha=0.0)
        dev_dig = cross_entropy(p_dig, q_dig) - math.log2(10)

        p_mult = empirical_dist([max_multiplicity(s)], MULTS, alpha=0.0)
        dev_mult = cross_entropy(p_mult, q_mult) - cross_entropy(p_mult, RAND_MULT)

        p_first = empirical_dist([s[0]], FIRST_DIGITS, alpha=0.0)
        dev_benf = cross_entropy(p_first, BENFORD) - math.log2(9)

        return [s1_nb, dev_dig, dev_mult, dev_benf]

    def fit(self, human_train_seqs: list[str], rng: random.Random, null_items: int = 1000):
        self.nb = Stage1NaiveBayes().fit(human_train_seqs)
        self.q_dig = empirical_dist("".join(human_train_seqs), DIGITS, alpha=1.0)
        self.q_mult = empirical_dist([max_multiplicity(s) for s in human_train_seqs], MULTS, alpha=0.5)

        # Leave-one-out on human training items to fit unbiased P(x_i | Fake) Gaussians
        fake_mat = []
        for idx, s in enumerate(human_train_seqs):
            sub = human_train_seqs[:idx] + human_train_seqs[idx + 1 :]
            nb_sub = Stage1NaiveBayes().fit(sub)
            qd_sub = empirical_dist("".join(sub), DIGITS, alpha=1.0)
            qm_sub = empirical_dist([max_multiplicity(x) for x in sub], MULTS, alpha=0.5)
            fake_mat.append(self._item_features(s, nb_sub, qd_sub, qm_sub))

        self.fake_gauss = [mean_std(col) for col in zip(*fake_mat)]

        # True random items (1000..9999) to fit P(x_i | Real) Gaussians
        rand_seqs = [str(rng.randint(1000, 9999)) for _ in range(null_items)]
        real_mat = [self._item_features(s, self.nb, self.q_dig, self.q_mult) for s in rand_seqs]
        self.real_gauss = [mean_std(col) for col in zip(*real_mat)]
        return self

    def predict_proba_batch_fake(self, batch: list[str]) -> float:
        # Multiply likelihoods across EVERY item in the batch (sum in log-space)
        joint_ll_fake = 0.0
        joint_ll_real = 0.0

        for s in batch:
            x = self._item_features(s, self.nb, self.q_dig, self.q_mult)
            for xi, (mf, sf), (mr, sr) in zip(x, self.fake_gauss, self.real_gauss):
                joint_ll_fake += math.log(max(normal_pdf(xi, mf, sf), 1e-300))
                joint_ll_real += math.log(max(normal_pdf(xi, mr, sr), 1e-300))

        # Law of Total Probability for the entire batch
        a = joint_ll_fake + math.log(self.prior_fake)
        b = joint_ll_real + math.log(self.prior_real)
        m = max(a, b)
        p_batch_and_fake = math.exp(a - m)
        p_batch_total = p_batch_and_fake + math.exp(b - m)
        return p_batch_and_fake / p_batch_total


def evaluate_joint_kfold(batch_size: int = 10, k_folds: int = 4, n_seeds: int = 20):
    accs, mean_p_fake, mean_p_real = [], [], []

    for seed in range(n_seeds):
        rng = random.Random(700 + seed)
        h_shuffled = HUMAN_DATA[:]
        rng.shuffle(h_shuffled)

        h_batches = [h_shuffled[i : i + batch_size] for i in range(0, len(h_shuffled), batch_size)]
        n = len(h_batches)
        r_batches = [[str(rng.randint(1000, 9999)) for _ in range(batch_size)] for _ in range(n)]

        idx = list(range(n))
        rng.shuffle(idx)
        folds = [idx[i::k_folds] for i in range(k_folds)]

        correct = total = 0
        for f in range(k_folds):
            tr_seqs = [num for j in range(k_folds) if j != f for i in folds[j] for num in h_batches[i]]
            te_h = [h_batches[i] for i in folds[f]]
            te_r = [r_batches[i] for i in folds[f]]

            detector = JointBatchFakeDetector(prior_fake=0.5).fit(tr_seqs, rng)

            for b in te_h:
                p = detector.predict_proba_batch_fake(b)
                mean_p_fake.append(p)
                correct += int(p >= 0.5)
                total += 1

            for b in te_r:
                p = detector.predict_proba_batch_fake(b)
                mean_p_real.append(p)
                correct += int(p < 0.5)
                total += 1

        accs.append(correct / total)

    return sum(accs) / n_seeds, sum(mean_p_fake) / len(mean_p_fake), sum(mean_p_real) / len(mean_p_real)


if __name__ == "__main__":
    print("=== Joint Batch Probability P(All k Items Fake | x_1..x_k) ===")
    for bs, kf in [(1, 5), (5, 4), (10, 4)]:
        acc, pf, pr = evaluate_joint_kfold(batch_size=bs, k_folds=kf, n_seeds=20)
        print(
            f"Batch size = {bs:2d} ({kf}-fold CV): Accuracy = {acc:.3f} | "
            f"Avg P(Batch=Fake) on Human = {pf:.3f}, on Random = {pr:.3f}"
        )

    # Demo on a batch of 10 suspect numbers vs. 10 truly random numbers
    rng_demo = random.Random(42)
    full_model = JointBatchFakeDetector(prior_fake=0.5).fit(HUMAN_DATA, rng_demo)

    suspect_10 = ["1984", "2019", "4729", "8361", "5104", "1998", "2024", "7391", "3456", "6182"]
    random_10 = [str(rng_demo.randint(1000, 9999)) for _ in range(10)]

    print("\n=== Whole-Batch Demo ===")
    print("Suspect 10:", suspect_10)
    print(f"  -> P(Entire Batch is Fake) = {full_model.predict_proba_batch_fake(suspect_10):.4f}")
    print("Random  10:", random_10)
    print(f"  -> P(Entire Batch is Fake) = {full_model.predict_proba_batch_fake(random_10):.4f}")