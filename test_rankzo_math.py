# test_rankzo_math.py

import numpy as np

from rankzo.graph_builder import build_edges
from rankzo.estimator import estimate_gradient


ranking = [2, 0, 1]

edges = build_edges(
    ranking
)

print(edges)

noises = np.array(
[
 [1,0],
 [0,1],
 [2,2]
]
)

g = estimate_gradient(
    noises,
    edges
)

print(g)