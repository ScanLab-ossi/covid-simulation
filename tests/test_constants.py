import unittest

from simulation.constants import *


class TestConstants(unittest.TestCase):
    def test_default_constants(self):
        self.assertEqual(settings["UPLOAD"], False)
        self.assertEqual(settings["LOCAL"], True)
        self.assertIsInstance(settings["VERBOSE"], bool)
        self.assertEqual(settings["PARALLEL"], False)
        self.assertEqual(settings["SKIP_TESTS"], True)
