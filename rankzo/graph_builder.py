import itertools


def build_edges(ranking):

    """
    ranking:
        [best_idx,...,worst_idx]

    returns:
        edge list
    """

    edges = []

    n = len(ranking)

    for higher_pos in range(n):

        for lower_pos in range(
            higher_pos + 1,
            n
        ):

            winner = ranking[higher_pos]
            loser = ranking[lower_pos]

            edges.append(
                (winner, loser)
            )

    return edges