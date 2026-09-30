"""The size report must measure the vocabulary the script would write."""
import unittest

from files.build_draft_vocab import select_vocab


class SelectVocabTests(unittest.TestCase):
    def test_a_frequent_pinned_id_does_not_consume_two_slots(self):
        # The old report did set(pinned) | ranked[:size - len(pinned)],
        # which is {1, 2} here: two ids, not four.
        special = {1, 2}
        ranked = [1, 2, 3, 4, 5]
        self.assertEqual(select_vocab(special, ranked, 4), {1, 2, 3, 4})

    def test_pinned_ids_that_are_not_frequent_still_fill_the_size(self):
        self.assertEqual(select_vocab({10}, [1, 2, 3], 3), {1, 2, 10})

    def test_extra_pinned_ids_are_kept_when_they_already_pass_the_size(self):
        self.assertEqual(select_vocab({1, 2, 3}, [4, 5], 2), {1, 2, 3})


if __name__ == "__main__":
    unittest.main()
