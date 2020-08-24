import numba
import numpy as np
from scipy.spatial import distance_matrix


@numba.jit
def _farthest_first_traversal(dist, k, row_ind=0, sample_edge=False):

    N = len(dist)

    # Collect indices of maximally distant vectors in the data array
    distant_inds = set()

    for i in range(N):
        old_row_ind = row_ind  # TODO: remove, for debugging
        # Row array
        row = dist[row_ind]
        maximal_dist = 0.
        # Set row_ind to maximal element in the row that
        # has not been seen before
        # all_dists = []
        # all_inds = []
        # cur_dists = []
        for j in range(N):
            if j not in distant_inds:

                # Sum of distances from all previous points to current point
                s = 0.
                if sample_edge:
                    for p in distant_inds:
                        s += dist[j][p]
                # all_dists.append(s)
                # all_inds.append(j)

                s += row[i]

                #print(f'sum s={s} j={j}')

                # Current maximal distance
                if s > maximal_dist:
                    maximal_dist = s
                    row_ind = j
                # cur_dists.append(row[j])

        assert (old_row_ind != row_ind) or i == 0
        # Add vector furthest away from current vector (row)
        distant_inds.add(row_ind)

        # Collect the first k only
        if len(distant_inds) >= k:
            return list(distant_inds)


# def fft(X,D,k):
#     """
#     X: input vectors (n_samples by dimensionality)
#     D: distance matrix (n_samples by n_samples)
#     k: number of centroids
#     out: indices of centroids
#     """
#     n=X.shape[0]
#     visited=[]
#     i=np.int32(np.random.uniform(n))
#     visited.append(i)
#     while len(visited)<k:
#         dist=np.mean([D[i] for i in visited],0)
#         for i in np.argsort(dist)[::-1]:
#             if i not in visited:
#                 visited.append(i)
#                 break
#     return np.array(visited)

def distance(A, B):
    return np.linalg.norm(A - B)


import random


def incremental_farthest_search(points, k):
    remaining_points = list(points[:])
    solution_set = []
    solution_inds = []
    solution_inds.append(random.randint(0, len(remaining_points) - 1))
    solution_set.append(remaining_points.pop(solution_inds[-1]))
    for _ in range(k - 1):
        distances = [distance(p, solution_set[0]) for p in remaining_points]
        for i, p in enumerate(remaining_points):
            for j, s in enumerate(solution_set):
                distances[i] = min(distances[i], distance(p, s))
        solution_inds.append(distances.index(max(distances)))
        solution_set.append(remaining_points.pop(solution_inds[-1]))
    return solution_set, solution_inds


def farthest_first_traversal(data, k, minkowski=2, threshold=1000000, sample_edge=False):
    """
    Furthest first traversal in O(n^2) time and space

    Parameters
    ----------
    data : np.ndarray
        Matrix of vectors (N, M) where the N vectors
        (each M dimensional) exist in a metric space.

    k : int
        Number of maximally distant vectors to return.

    minkowski : int
        Which Minkowski p-norm to use.

    threshold : int
        If M * N * K > threshold, algorithm uses a
        Python loop instead of large temporary arrays.

    Note: We choose a random starting vector but do not
          add immediately add it to the k returned vectors.
          The farthest_first_traversal algorithm takes the
          initial seed and first finds the farthest vector
          away from it to add to the k returned vectors.
          Thus, the first vector returned is guaranteed to
          be on the edge of the convex hull defined by the
          data.
    """

    # Matrix containing the distance from every vector in x to every vector in y.
    dist = distance_matrix(data, data, p=minkowski, threshold=threshold)

    # Randomly choose starting vector.
    row_ind = np.random.randint(low=0, high=len(data))

    # Helper function returns indices into data array
    return _farthest_first_traversal(dist, k, row_ind, sample_edge)
