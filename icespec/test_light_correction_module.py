import unittest
from icespec.light_correction_module import correct
import numpy as np
import random


class TestTrim(unittest.TestCase):
    def test_small_arrays(self):
        # Define small example arrays
        img_data = np.array([[5, 10], [15, 20]])
        d_ref = np.array([[1, 2], [3, 4]])
        w_ref = np.array([[6, 12], [18, 24]])

        # Expected output computed manually
        # corrected_data = (img_data - d_ref) / (w_ref - d_ref)
        expected = np.array([
            [(5 - 1) / (6 - 1), (10 - 2) / (12 - 2)],
            [(15 - 3) / (18 - 3), (20 - 4) / (24 - 4)]
        ])

        # Run the function
        result = correct(img_data, d_ref, w_ref)

        # Assert arrays are equal
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_edge_case_division_by_zero(self):
        # Edge case: Division by zero
        img_data = np.array([[5, 10]])
        d_ref = np.array([[1, 2]])
        w_ref = np.array([[1, 2]])  # Denominator becomes zero

        # Expect division by zero to raise a warning
        with self.assertRaises(ZeroDivisionError):
            correct(img_data, d_ref, w_ref)

    def test_broadcasting_support(self):
        # Test broadcasting of inputs
        img_data = np.array([[5, 10], [15, 20]])
        d_ref = np.array([1, 2])  # Broadcasting
        w_ref = np.array([6, 12])  # Broadcasting

        # Expected output with broadcasting
        expected = np.array([
            [(5 - 1) / (6 - 1), (10 - 2) / (12 - 2)],
            [(15 - 1) / (6 - 1), (20 - 2) / (12 - 2)]
        ])

        # Run the function
        result = correct(img_data, d_ref, w_ref)

        # Assert arrays are equal
        np.testing.assert_allclose(result, expected, rtol=1e-5)

if __name__ == "__main__":
    unittest.main()