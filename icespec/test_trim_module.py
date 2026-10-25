# test_data_cropping_module.py
import unittest
from icespec.trim_module import trim
import numpy as np
import random


class TestTrim(unittest.TestCase):
    def test_trim(self):
        """
        Unit test for data_cropping function.
        """
        lines=1010
        samples=1024
        bands=224
        self.full_data = np.random.random(
            (lines, samples, bands)
        )  # Random 3D array for testing.

        d_ref = np.random.random(
            (lines, samples, bands)
        )

        w_ref = np.random.random(
            (lines, samples, bands)
        )

        dataSize = [450, 920, (random.randint(50, 110)), (random.randint(720, 1000))]
        # [imgStartLine, imgEndLine, imgLeft, imgRight]

        w_stdSize = [530, 820, 110, 720]
        #290, 610, 224

        data_cropped = trim(
            self.full_data, w_ref, d_ref, dataSize, w_stdSize, Debug=False
        )

        # Check if shapes are correct
        #self.assertEqual((290, 610), representative_band_cropped.shape)
        self.assertEqual((290, 610, 224), data_cropped[0].shape)
