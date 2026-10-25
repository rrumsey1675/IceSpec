"""An example of unit tests in Python
"""
import unittest

from hsi.module_example import *


class Test_module_example(unittest.TestCase):
    """Unit tests for module_example."""

    def test_sum_list(self):
        # Test empty list
        lst = []
        self.assertEqual(0, sum_list(lst))

        # Test single element list
        lst = [1]
        self.assertEqual(1, sum_list(lst))

        # Test multiple element list
        lst = list(range(1, 11))
        self.assertEqual(55, sum_list(lst))
