from abc import ABC
from abc import abstractmethod


class RankingOracle(ABC):

    @abstractmethod
    def rank(self, returns):
        pass