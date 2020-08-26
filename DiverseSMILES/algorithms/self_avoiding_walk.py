import random
import numpy as np
from scipy.spatial import distance_matrix


def self_avoiding_walk(data, k, candidates=10, minkowski=2, threshold=1000000):
    """
    Self avoiding walk with a few tweaks.

    Samples points from the space such that:
        1. each point is unique
        2. sample_{i+1} must be of distance > radius from sample_i

    Generates at random a set of size `candidates` of candidates and
    finds the minimum distance candidate such that 1,2 are satisfied.
    """

    # Matrix containing the distance from every vector in x to every vector in y.
    dist_mat = distance_matrix(data, data, p=minkowski, threshold=threshold)

    radius = np.mean(dist_mat) + np.min(dist_mat)
    max_dist = np.max(dist_mat) + 1

    sample_space = set(range(len(data)))
    # Start inds with a random element from the sample space
    prev_selected = sample_space.pop()
    inds = {prev_selected}

    while len(inds) < k:

        min_dist = max_dist
        for sample in random.sample(sample_space, candidates):
            candidate_dist = dist_mat[prev_selected][sample]
            # Self avoiding
            if sample not in inds:
                # Candidate must be atleast radius dist away from
                # previously selected ind
                if candidate_dist > radius:
                    # Add the minimum distance point greater than radius
                    # that has not been seen before
                    if candidate_dist < min_dist:
                        min_dist = candidate_dist
                        selected = sample

        # Incase points from random sample don't satisfy contraints
        if selected == prev_selected:
            continue

        inds.add(selected)
        sample_space.remove(selected)
        prev_selected = selected

    return list(inds)