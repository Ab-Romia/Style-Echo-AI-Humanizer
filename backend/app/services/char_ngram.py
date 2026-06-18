"""
Character n-gram author profiler.

Character n-grams are the strongest cross-domain signal in authorship work:
they capture morphology, function-word habits, and punctuation rhythm without
keying on topic. This profiler fits a tf-idf space on an author's samples and
scores how close a candidate text sits to the author's average n-gram vector.
"""
from typing import List, Optional

import numpy as np


class CharNgramProfiler:
    """Fits a character n-gram tf-idf profile for one author and scores texts."""

    def __init__(self, ngram_range=(2, 4), min_df: int = 1):
        """
        Args:
            ngram_range: Character n-gram sizes to use.
            min_df: Minimum document frequency for a feature.
        """
        self.ngram_range = ngram_range
        self.min_df = min_df
        self._vectorizer = None
        self._author_vector = None

    def fit(self, samples: List[str]) -> "CharNgramProfiler":
        """
        Fit the tf-idf space on the author's samples and store the centroid.

        Args:
            samples: The author's writing samples.

        Returns:
            self.

        Raises:
            ValueError: When no non-empty samples are provided.
        """
        cleaned = [s for s in samples if s and s.strip()]
        if not cleaned:
            raise ValueError("Cannot fit a char n-gram profile on zero samples.")

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:
            raise RuntimeError(
                "scikit-learn is required for the char n-gram profiler."
            ) from exc

        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.ngram_range,
            min_df=self.min_df,
        )
        matrix = self._vectorizer.fit_transform(cleaned)
        # Author centroid is the mean tf-idf vector over the samples.
        self._author_vector = np.asarray(matrix.mean(axis=0)).ravel()
        return self

    def similarity(self, text: str) -> float:
        """
        Cosine similarity of a candidate text to the author centroid.

        Args:
            text: Candidate text.

        Returns:
            Similarity in [0, 1]. Returns 0.0 for empty input.

        Raises:
            RuntimeError: When called before fit.
        """
        if self._vectorizer is None or self._author_vector is None:
            raise RuntimeError("Call fit before similarity.")
        if not text or not text.strip():
            return 0.0

        candidate = np.asarray(
            self._vectorizer.transform([text]).todense()
        ).ravel()

        author_norm = np.linalg.norm(self._author_vector)
        candidate_norm = np.linalg.norm(candidate)
        if author_norm == 0 or candidate_norm == 0:
            return 0.0

        cosine = float(
            np.dot(self._author_vector, candidate)
            / (author_norm * candidate_norm)
        )
        return max(0.0, min(1.0, cosine))

    def get_author_vector(self) -> Optional[np.ndarray]:
        """Return the stored author centroid, or None before fit."""
        return self._author_vector
