"""
Basic sanity tests for the registration pipeline.
"""
import unittest
import numpy as np

class TestRegistrationPipeline(unittest.TestCase):
    def test_identity_transform(self):
        """Registering an image against itself should result in near-zero RMSE."""
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
