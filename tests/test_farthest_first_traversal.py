import unittest
import numpy as np
from ApproximateGreedyPermutation.algorithms import farthest_first_traversal


class Test_farthest_first_traversal(unittest.TestCase):

    def test_basic(self):
        data = np.array([[1, 2], [5, 6], [0, 0]])
        k_farthest = farthest_first_traversal(data, k=2)

        self.assertTrue(np.array([0, 0]) in k_farthest)
        self.assertTrue(np.array([5, 6]) in k_farthest)
        self.assertEqual(k_farthest.shape, (2, 2))


if __name__ == '__main__':
    unittest.main()
