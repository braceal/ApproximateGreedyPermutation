import random
import numpy as np
from scipy.spatial import distance_matrix


def _mitchells_best_candidate(data, k, candidates=10, minkowski=2, threshold=1000000):
    """
    Approximates Poisson-disc distribution.
    """

    # Matrix containing the distance from every vector in x to every vector in y.
    dist_mat = distance_matrix(data, data, p=minkowski, threshold=threshold)

    radius = np.mean(dist_mat)

    sample_space = set(range(len(data)))
    # Start inds with a random element from the sample space
    selected = sample_space.pop()
    inds = {selected}

    while len(inds) < k:
        # List of sample candidates
        samples = random.sample(sample_space, candidates)

        max_dist = 0.
        for sample in samples:

            # Sum the distances from the candidate point
            # to all previous inds
            dist = 0.
            candidate_row = dist_mat[sample]
            for ind in inds:
                dist += candidate_row[ind]
            dist /= len(inds)

            if dist > max_dist:
                max_dist = dist
                selected = sample

        inds.add(selected)
        sample_space.remove(selected)

    return list(inds)


def mitchells_best_candidate(data, k, candidates=10, minkowski=2, threshold=1000000):
    """
    Approximates Poisson-disc distribution.
    """

    # Matrix containing the distance from every vector in x to every vector in y.
    dist_mat = distance_matrix(data, data, p=minkowski, threshold=threshold)

    radius = np.mean(dist_mat)

    sample_space = set(range(len(data)))
    # Start inds with a random element from the sample space
    selected = sample_space.pop()
    inds = {selected}

    while len(inds) < k:
        # List of sample candidates
        samples = random.sample(sample_space, candidates)

        max_dist = 0.
        for sample in samples:

            # Distance from previously selected point
            # to candidate sample
            dist = dist_mat[sample][selected]

            # dist + radius
            if dist + radius > max_dist:
                max_dist = dist
                selected = sample

        inds.add(selected)
        sample_space.remove(selected)

    return list(inds)
