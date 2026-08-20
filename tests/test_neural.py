import unittest

import torch

from resolveai.neural import NeuralTextClassifier


class NeuralModelTest(unittest.TestCase):
    def test_shapes_and_padding_are_finite(self) -> None:
        torch.manual_seed(3)
        model = NeuralTextClassifier(
            vocabulary_size=20,
            num_classes=4,
            embedding_dim=8,
            hidden_dim=12,
            dropout=0.0,
        ).eval()
        tokens = torch.tensor([[2, 3, 0], [4, 0, 0]])
        encoded = model.encode(tokens)
        logits = model(tokens)
        self.assertEqual(tuple(encoded.shape), (2, 12))
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
