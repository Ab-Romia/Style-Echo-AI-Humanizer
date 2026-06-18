"""
Linguistic Feature Extraction Module.

Extracts concrete linguistic features from text samples including:
- Sentence statistics (length, complexity, dependency depth)
- Lexical diversity (type-token ratio)
- POS distribution
- Punctuation patterns
- Readability scores
"""
import re
import statistics
from typing import Dict, List, Any
import spacy
import textstat
from collections import Counter


class LinguisticAnalyzer:
    """Analyzes text to extract linguistic features."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize the linguistic analyzer.

        Args:
            spacy_model: Name of the spaCy model to use
        """
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            # Model not found, will need to download
            print(f"spaCy model '{spacy_model}' not found. Please download it with: python -m spacy download {spacy_model}")
            raise

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive linguistic analysis on text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary containing all linguistic features
        """
        if not text or not text.strip():
            return self._empty_features()

        doc = self.nlp(text)

        features = {
            **self._analyze_sentences(doc),
            **self._analyze_lexical_diversity(doc),
            **self._analyze_pos_distribution(doc),
            **self._analyze_punctuation(text),
            **self._analyze_readability(text),
            **self._analyze_syntax_complexity(doc),
        }

        return features

    def _empty_features(self) -> Dict[str, Any]:
        """Return a zeroed feature set for empty input."""
        return {
            "avg_sentence_length": 0.0,
            "sentence_length_std": 0.0,
            "num_sentences": 0,
            "type_token_ratio": 0.0,
            "num_unique_words": 0,
            "num_total_words": 0,
            "pos_distribution": {},
            "pos_counts": {},
            "punctuation_patterns": {},
            "punctuation_density": 0.0,
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
            "gunning_fog": 0.0,
            "smog_index": 0.0,
            "coleman_liau_index": 0.0,
            "avg_dependency_depth": 0.0,
            "max_dependency_depth": 0,
        }

    def _analyze_sentences(self, doc) -> Dict[str, float]:
        """
        Analyze sentence-level statistics.

        Args:
            doc: spaCy Doc object

        Returns:
            Dictionary with sentence statistics
        """
        sentences = list(doc.sents)
        sentence_lengths = [len(sent) for sent in sentences]

        if not sentence_lengths:
            return {
                "avg_sentence_length": 0.0,
                "sentence_length_std": 0.0,
                "num_sentences": 0,
            }

        return {
            "avg_sentence_length": statistics.mean(sentence_lengths),
            "sentence_length_std": statistics.stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0.0,
            "num_sentences": len(sentences),
            "min_sentence_length": min(sentence_lengths),
            "max_sentence_length": max(sentence_lengths),
        }

    def _analyze_syntax_complexity(self, doc) -> Dict[str, float]:
        """
        Measure syntax complexity using dependency tree depth.

        Args:
            doc: spaCy Doc object

        Returns:
            Dictionary with complexity metrics
        """
        # Cap recursion so a malformed or cyclic parse cannot blow the stack.
        max_depth_cap = 100

        def get_tree_depth(token, depth: int = 0) -> int:
            """Get maximum depth of dependency tree from this token."""
            if depth >= max_depth_cap:
                return depth
            children = list(token.children)
            if not children:
                return depth
            return max(get_tree_depth(child, depth + 1) for child in children)

        sentences = list(doc.sents)
        if not sentences:
            return {
                "avg_dependency_depth": 0.0,
                "max_dependency_depth": 0,
            }

        depths = []
        for sent in sentences:
            # A sentence may have zero or several tokens whose head is itself
            # (fragments, parse errors). Measure depth from every such root and
            # take the deepest; skip the sentence if none exists.
            roots = [token for token in sent if token.head == token]
            if not roots:
                continue
            depths.append(max(get_tree_depth(root) for root in roots))

        return {
            "avg_dependency_depth": statistics.mean(depths) if depths else 0.0,
            "max_dependency_depth": max(depths) if depths else 0,
        }

    def _analyze_lexical_diversity(self, doc) -> Dict[str, float]:
        """
        Calculate lexical diversity using type-token ratio.

        Args:
            doc: spaCy Doc object

        Returns:
            Dictionary with lexical diversity metrics
        """
        # Filter out punctuation and spaces
        tokens = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]

        if not tokens:
            return {
                "type_token_ratio": 0.0,
                "num_unique_words": 0,
                "num_total_words": 0,
            }

        unique_tokens = set(tokens)

        return {
            "type_token_ratio": len(unique_tokens) / len(tokens),
            "num_unique_words": len(unique_tokens),
            "num_total_words": len(tokens),
        }

    def _analyze_pos_distribution(self, doc) -> Dict[str, Any]:
        """
        Analyze part-of-speech distribution.

        Args:
            doc: spaCy Doc object

        Returns:
            Dictionary with POS distribution
        """
        pos_counts = Counter(token.pos_ for token in doc if not token.is_space)
        total_tokens = sum(pos_counts.values())

        if total_tokens == 0:
            return {"pos_distribution": {}}

        # Calculate percentages
        pos_distribution = {
            pos: count / total_tokens for pos, count in pos_counts.items()
        }

        return {
            "pos_distribution": pos_distribution,
            "pos_counts": dict(pos_counts),
        }

    def _analyze_punctuation(self, text: str) -> Dict[str, Any]:
        """
        Analyze punctuation patterns.

        Args:
            text: Input text

        Returns:
            Dictionary with punctuation statistics
        """
        # Long-dash characters are built from their code points so this source
        # file contains no literal long dash: U+2014 is the em dash, U+2013 the
        # en dash, U+2026 the ellipsis.
        em_dash = chr(0x2014)
        en_dash = chr(0x2013)
        ellipsis_char = chr(0x2026)

        # Count specific punctuation marks
        comma_count = text.count(',')
        semicolon_count = text.count(';')
        colon_count = text.count(':')
        em_dash_count = text.count(em_dash) + text.count('--')
        en_dash_count = text.count(en_dash)
        ellipsis_count = text.count('...') + text.count(ellipsis_char)
        exclamation_count = text.count('!')
        question_count = text.count('?')
        period_count = text.count('.')

        # Count words for normalization
        word_count = len(text.split())

        if word_count == 0:
            return {
                "punctuation_patterns": {},
                "punctuation_density": 0.0,
            }

        return {
            "punctuation_patterns": {
                "comma_per_100_words": (comma_count / word_count) * 100,
                "semicolon_per_100_words": (semicolon_count / word_count) * 100,
                "colon_per_100_words": (colon_count / word_count) * 100,
                "em_dash_per_100_words": (em_dash_count / word_count) * 100,
                "en_dash_per_100_words": (en_dash_count / word_count) * 100,
                "ellipsis_per_100_words": (ellipsis_count / word_count) * 100,
                "exclamation_per_100_words": (exclamation_count / word_count) * 100,
                "question_per_100_words": (question_count / word_count) * 100,
            },
            "punctuation_density": (comma_count + semicolon_count + colon_count + em_dash_count) / word_count,
        }

    def _analyze_readability(self, text: str) -> Dict[str, float]:
        """
        Calculate readability scores.

        Args:
            text: Input text

        Returns:
            Dictionary with readability metrics
        """
        if not text.strip():
            return {
                "flesch_reading_ease": 0.0,
                "flesch_kincaid_grade": 0.0,
                "gunning_fog": 0.0,
                "smog_index": 0.0,
                "coleman_liau_index": 0.0,
            }

        return {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
            "gunning_fog": textstat.gunning_fog(text),
            "smog_index": textstat.smog_index(text),
            "coleman_liau_index": textstat.coleman_liau_index(text),
        }

    def analyze_multiple_samples(self, samples: List[str]) -> Dict[str, Any]:
        """
        Analyze multiple text samples and aggregate features.

        Args:
            samples: List of text samples

        Returns:
            Aggregated linguistic features across all samples
        """
        if not samples:
            return {"total_word_count": 0}

        all_features = [self.analyze_text(sample) for sample in samples]

        # Aggregate numeric features
        aggregated = {}

        # Get all numeric keys from the first sample
        if all_features:
            first_features = all_features[0]

            for key, value in first_features.items():
                if isinstance(value, (int, float)):
                    # Average numeric values; also record the spread across
                    # samples so the profile captures how consistent the
                    # author is on each axis, not just the mean.
                    values = [f[key] for f in all_features if key in f and isinstance(f[key], (int, float))]
                    aggregated[key] = statistics.mean(values) if values else 0.0
                    if not key.endswith("_std"):
                        aggregated[f"{key}_std"] = (
                            statistics.stdev(values) if len(values) > 1 else 0.0
                        )
                elif isinstance(value, dict):
                    # For nested dicts, average each nested value
                    aggregated[key] = {}
                    for nested_key in value.keys():
                        nested_values = [
                            f[key][nested_key] for f in all_features
                            if key in f and isinstance(f[key], dict) and nested_key in f[key]
                            and isinstance(f[key][nested_key], (int, float))
                        ]
                        if nested_values:
                            aggregated[key][nested_key] = statistics.mean(nested_values)

        # Add total word count
        total_words = sum(f.get("num_total_words", 0) for f in all_features)
        aggregated["total_word_count"] = total_words

        return aggregated
