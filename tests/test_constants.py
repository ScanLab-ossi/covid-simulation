import unittest

from simulation.constants import *


class TestConstants(unittest.TestCase):
    def test_default_constants(self):
        self.assertEqual(REPETITIONS, 1)
        self.assertEqual(UPLOAD, False)
        self.assertEqual(LOCAL, True)
        self.assertEqual(VERBOSE, False)
        self.assertEqual(PARALLEL, False)
        self.assertEqual(SKIP_TESTS, True)
