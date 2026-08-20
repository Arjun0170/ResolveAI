import unittest

import numpy as np

from resolveai.text import PAD_TOKEN, UNK_TOKEN, Vocabulary, build_label_mapping, tokenize


class TextTest(unittest.TestCase):
    def test_tokenize_normalizes_and_keeps_contractions(self) -> None:
        self.assertEqual(tokenize("Can't FIND card #42!"), ["can't", "find", "card", "42"])

    def test_vocabulary_is_frequency_then_alphabetically_ordered(self) -> None:
        vocabulary = Vocabulary.build(
            ["beta alpha beta", "gamma alpha"],
            max_size=5,
            min_frequency=1,
            max_length=3,
        )
        self.assertEqual(
            vocabulary.id_to_token,
            [PAD_TOKEN, UNK_TOKEN, "alpha", "beta", "gamma"],
        )
        encoded, lengths = vocabulary.encode_padded(["alpha missing", "beta"])
        np.testing.assert_array_equal(encoded, [[2, 1, 0], [3, 0, 0]])
        np.testing.assert_array_equal(lengths, [2, 1])

    def test_oos_is_last_label(self) -> None:
        mapping, labels = build_label_mapping(["beta", "oos", "alpha"])
        self.assertEqual(labels, ["alpha", "beta", "oos"])
        self.assertEqual(mapping["oos"], 2)


if __name__ == "__main__":
    unittest.main()
