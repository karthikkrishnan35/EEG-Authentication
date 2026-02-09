import unittest
from src.preprocessing.preprocess import preprocess_data

class TestPreprocessing(unittest.TestCase):

    def test_preprocess_data_shape(self):
        raw_data = ...  # Load or create a sample raw EEG data array
        processed_data = preprocess_data(raw_data)
        self.assertEqual(processed_data.shape[1], expected_number_of_channels)
        self.assertEqual(processed_data.shape[0], expected_number_of_samples)

    def test_preprocess_data_values(self):
        raw_data = ...  # Load or create a sample raw EEG data array
        processed_data = preprocess_data(raw_data)
        self.assertTrue((processed_data >= 0).all())  # Example check for non-negative values

    def test_preprocess_data_normalization(self):
        raw_data = ...  # Load or create a sample raw EEG data array
        processed_data = preprocess_data(raw_data)
        self.assertAlmostEqual(processed_data.mean(), expected_mean, delta=0.01)
        self.assertAlmostEqual(processed_data.std(), expected_std, delta=0.01)

if __name__ == '__main__':
    unittest.main()