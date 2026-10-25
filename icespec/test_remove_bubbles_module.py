# chatGPT
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from icespec.remove_bubbles_module import remove, masking, deblending_image, inpaint_data

class TestHyperspectralProcessing(unittest.TestCase):
    def setUp(self):
        # Create mock hyperspectral image (50x50 pixels, 3 bands)
        # This will serve as input data for testing
        self.mock_image = np.random.rand(50, 50, 3).astype(np.float32)
        self.mock_fwhm = [2.5, 3.0, 4.0]  # Example FWHM values per band

    @patch('remove_bubbles_module.deblending_image')
    @patch('remove_bubbles_module.inpaint_data')
    def test_remove(self, mock_inpaint_data, mock_deblending_image):
        # Mock the behavior of deblending_image and inpaint_data to isolate the test
        mock_deblending_image.side_effect = lambda img, idx, fwhm: np.zeros(img.shape, dtype=bool)
        mock_inpaint_data.side_effect = lambda band, mask: band  # Return the same data for simplicity

        # Call the remove function and capture the result
        result = remove(self.mock_image, self.mock_fwhm)

        # Assertions to ensure the function behaves as expected
        self.assertEqual(result.shape, self.mock_image.shape)  # Check output shape
        mock_deblending_image.assert_called()  # Ensure deblending_image was called
        mock_inpaint_data.assert_called()  # Ensure inpaint_data was called

    @patch('remove_bubbles_module.calcsigma.calculate_sigma', return_value=1.0)
    @patch('remove_bubbles_module.Background2D')
    @patch('remove_bubbles_module.detect_sources')
    def test_deblending_image(self, mock_detect_sources, mock_Background2D, mock_calculate_sigma):
        # Mock background subtraction and source detection to isolate the test
        mock_Background2D.return_value.background = np.zeros((50, 50))
        mock_detect_sources.return_value = MagicMock(data=np.zeros((50, 50), dtype=bool))

        # Use a single band from the mock image for testing
        band_data = self.mock_image[:, :, 0]  # Single band
        fwhm = self.mock_fwhm[0]  # FWHM for the first band

        # Call deblending_image and capture the result
        segm_mask = deblending_image(band_data, 0, fwhm)

        # Assertions to validate the output and mock interactions
        self.assertEqual(segm_mask.shape, band_data.shape)  # Check output shape
        mock_calculate_sigma.assert_called_once_with(fwhm)  # Ensure sigma calculation was called
        mock_Background2D.assert_called()  # Ensure background estimation was called
        mock_detect_sources.assert_called()  # Ensure source detection was called

    def test_inpaint_data(self):
        # Test the inpaint_data function with a simple mask
        band_data = self.mock_image[:, :, 0]  # Single band data
        mask_band = np.zeros((50, 50), dtype=bool)
        mask_band[10:20, 10:20] = True  # Create a small masked region

        # Call inpaint_data and capture the result
        inpainted_band = inpaint_data(band_data, mask_band)

        # Assertions to ensure the function behaves as expected
        self.assertEqual(inpainted_band.shape, band_data.shape)  # Check output shape
        self.assertTrue(np.all(inpainted_band >= 0))  # Ensure no negative values in the result

if __name__ == '__main__':
    unittest.main()
