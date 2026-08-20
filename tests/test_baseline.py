import tempfile
import unittest
from pathlib import Path

import numpy as np

from resolveai.baseline import MultinomialNaiveBayes


class BaselineTest(unittest.TestCase):
    def test_fit_predict_and_round_trip(self) -> None:
        rows = [[2, 2, 3], [2, 3], [4, 4, 5], [4, 5]]
        labels = np.asarray([0, 0, 1, 1])
        model = MultinomialNaiveBayes(alpha=1.0).fit(rows, labels, 2, 6)
        predictions = model.predict_logits([[2, 3], [4, 5]]).argmax(axis=1)
        np.testing.assert_array_equal(predictions, [0, 1])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            restored = MultinomialNaiveBayes.load(path)
            np.testing.assert_allclose(
                restored.predict_logits(rows),
                model.predict_logits(rows),
            )


if __name__ == "__main__":
    unittest.main()
