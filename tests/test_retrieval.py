import unittest
from pathlib import Path

import numpy as np

from resolveai.retrieval import CppTopK, build_tfidf_matrix, tfidf_queries, top_k_indices
from resolveai.text import Vocabulary


class RetrievalTest(unittest.TestCase):
    def test_tfidf_prefers_matching_document(self) -> None:
        documents = ["lost phone location", "cash card payment"]
        vocabulary = Vocabulary.build(documents, min_frequency=1)
        matrix, inverse_document_frequency = build_tfidf_matrix(documents, vocabulary)
        query = tfidf_queries(["find lost phone"], vocabulary, inverse_document_frequency)
        scores = query @ matrix.T
        self.assertEqual(int(top_k_indices(scores, 1)[0, 0]), 0)

    def test_cpp_top_k_matches_numpy_when_built(self) -> None:
        library = Path("build/libresolve_topk.so")
        if not library.exists():
            self.skipTest("native backend has not been built")
        matrix = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]],
            dtype=np.float32,
        )
        query = np.asarray([1.0, 0.0], dtype=np.float32)
        expected = top_k_indices((matrix @ query)[None, :], 2)[0]
        actual, _ = CppTopK(library).search(matrix, query, 2)
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
