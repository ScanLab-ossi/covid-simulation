import unittest

from simulation.helpers import one_array_pickle_to_set, timing


class TestHelpers(unittest.TestCase):
    def test_pickle_to_set(self):
        self.assertTrue(
            type(one_array_pickle_to_set("data/destination_ids_first_3days.pickle"))
            is set
        )

    def test_timing(self):
        pass
