import unittest
import numpy as np
from DiverseSMILES.algorithms import farthest_first_traversal


class Test_farthest_first_traversal(unittest.TestCase):

    def test_basic(self):
        data = np.array([[1, 2], [5, 6], [0, 0]])
        k_farthest = farthest_first_traversal(data, k=2)

        farthest = data[k_farthest]

        self.assertTrue(np.array([0, 0]) in farthest)
        self.assertTrue(np.array([5, 6]) in farthest)
        self.assertEqual(farthest.shape, (2, 2))


if __name__ == '__main__':
    unittest.main()
