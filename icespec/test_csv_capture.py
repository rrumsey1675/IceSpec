import unittest
import pandas as pd

from icespec.csv_capture_module import *

class Test_csv_capture(unittest.TestCase):
    "Unit tests for csv_capture_module"

    def test_getData(self):

        DataFrame = pd.read_csv('./processing/HSI_July2024.csv') #reading in file with core data


        self.assertIsNotNone(DataFrame, msg="DataFrame is empty")


