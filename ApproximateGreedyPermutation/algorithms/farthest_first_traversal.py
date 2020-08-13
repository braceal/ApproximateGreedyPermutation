import numba
import numpy as np
from scipy.spatial import distance_matrix


@numba.jit
def _farthest_first_traversal(dist, row_ind=0):

    N = len(dist)

    # Collect indices of maximally distant vectors in the data array
    distant_inds = set()

    for i in range(N):
        old_row_ind = row_ind # TODO: remove, for debugging
        # Row array
        row = dist[row_ind]
        maximal_dist = 0.
        # Set row_ind to maximal element in the row that
        # has not been seen before
        for j in range(N):
            if j not in distant_inds:
                d = row[j]
                if d > maximal_dist and j not in distant_inds:
                    maximal = d
                    row_ind = j

        assert old_row_ind != row_ind
        # Add vector furthest away from current vector (row)
        distant_inds.add(row_ind)

        # Collect the first k only
        if len(distant_inds) >= k:
            return data[list(distant_inds)]

def farthest_first_traversal(data, k, minkowski=2, threshold=1000000):
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

    return _farthest_first_traversal(dist, row_ind)
