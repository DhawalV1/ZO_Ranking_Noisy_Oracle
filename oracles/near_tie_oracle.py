import numpy as np

from .base_oracle import RankingOracle


class NearTieOracle(RankingOracle):

    def __init__(
        self,
        tau=2.0,
        flip_prob=0.3
    ):
        self.tau = tau
        self.flip_prob = flip_prob

        self.total_pairs = 0
        self.flipped_pairs = 0

    def rank(self, returns):

        returns = np.asarray(returns)

        m = len(returns)

        scores = returns.copy()

        ################################################
        # pairwise near-tie corruption
        ################################################

        for i in range(m):
            for j in range(i + 1, m):

                diff = abs(
                    returns[i]
                    -
                    returns[j]
                )

                if diff <= self.tau:

                    self.total_pairs += 1

                    if np.random.rand() < self.flip_prob:

                        self.flipped_pairs += 1

                        # swap ordering by adding a tiny perturbation
                        temp = scores[i]
                        scores[i] = scores[j]
                        scores[j] = temp

        ranking = np.argsort(
            -scores
        )

        return ranking

    def corruption_rate(self):

        if self.total_pairs == 0:
            return 0.0

        return (
            self.flipped_pairs
            /
            self.total_pairs
        )
    
    def reset_stats(self):

        self.total_pairs = 0
        self.flipped_pairs = 0