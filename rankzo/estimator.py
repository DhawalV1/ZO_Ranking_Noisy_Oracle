import numpy as np


def estimate_gradient(
    noises,
    edges
):

    """
    noises:
        shape (m,d)

    edges:
        [(winner,loser),...]

    returns:
        gradient estimate
    """

    d = noises.shape[1]

    if len(edges) == 0:
        return np.zeros(d)

    g = np.zeros(d)

    for winner, loser in edges:

        g += (
            noises[loser]
            -
            noises[winner]
        )

    g /= len(edges)

    return g