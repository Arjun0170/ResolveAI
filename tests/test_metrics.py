import unittest

import numpy as np

from resolveai.metrics import (
    apply_abstention,
    classification_report,
    ranking_report,
    softmax,
)


class MetricsTest(unittest.TestCase):
    def test_softmax_is_stable_and_normalized(self) -> None:
        probabilities = softmax(np.asarray([[1_000.0, 1_001.0]]))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertGreater(probabilities[0, 1], probabilities[0, 0])

    def test_classification_metrics(self) -> None:
        report = classification_report(
            np.asarray([0, 0, 1, 1]),
            np.asarray([0, 1, 1, 1]),
            num_classes=2,
        )
        self.assertEqual(report["accuracy"], 0.75)
        self.assertAlmostEqual(report["macro_recall"], 0.75)

    def test_abstention_and_ranking(self) -> None:
        predictions = apply_abstention(
            np.asarray([0, 1, 0]),
            np.asarray([0.9, 0.2, 0.8]),
            threshold=0.5,
            oos_id=2,
        )
        np.testing.assert_array_equal(predictions, [0, 2, 0])
        report = ranking_report(
            np.asarray([[2, 1, 0], [0, 1, 2]]),
            np.asarray([2, 1]),
        )
        self.assertEqual(report["recall_at_1"], 0.5)
        self.assertEqual(report["recall_at_3"], 1.0)
        self.assertEqual(report["mrr"], 0.75)


if __name__ == "__main__":
    unittest.main()
