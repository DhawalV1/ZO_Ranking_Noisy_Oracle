# test_oracle.py

from oracles.near_tie_oracle import NearTieOracle

oracle = NearTieOracle(
    tau=2.0,
    flip_prob=0.5
)

returns = [
    10,
    9.5,
    7,
    2,
    1
]

for _ in range(10):

    ranking = oracle.rank(
        returns
    )

    print(ranking)