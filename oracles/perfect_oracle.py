import numpy as np

from .base_oracle import RankingOracle


class PerfectOracle(RankingOracle):

    def rank(self, returns):

        returns = np.asarray(returns)

        ranking = np.argsort(
            -returns
        )

        return ranking
    
    def reset_stats(self):

        self.total_pairs = 0
        self.flipped_pairs = 0