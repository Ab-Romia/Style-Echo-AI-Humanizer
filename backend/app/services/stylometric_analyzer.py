"""
Stylometric Analysis Module.

Analyzes stylometric markers including:
- Function word frequencies
- N-gram patterns
- Contraction usage
- Passive vs active voice ratio
- Constituency parsing for sentence structures
"""
import re
import statistics
from typing import Dict, List, Any, Tuple
from collections import Counter
import spacy
from nltk import ngrams
from nltk.corpus import stopwords
import nltk


class StylometricAnalyzer:
    """Analyzes text for stylometric markers."""

    # Common function words to track
    FUNCTION_WORDS = [
        'the', 'and', 'but', 'so', 'or', 'if', 'when', 'that', 'this',
        'a', 'an', 'of', 'to', 'in', 'for', 'on', 'with', 'as', 'by',
        'at', 'from', 'which', 'who', 'what', 'where', 'while', 'because',
        'although', 'though', 'however', 'therefore', 'thus', 'hence',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'may', 'might',
        'can', 'could', 'must', 'ought'
    ]

    # Common contractions
    CONTRACTIONS = [
        "n't", "'s", "'re", "'ve", "'ll", "'d", "'m",
        "won't", "can't", "don't", "doesn't", "didn't", "isn't", "aren't",
        "wasn't", "weren't", "haven't", "hasn't", "hadn't", "wouldn't",
        "shouldn't", "couldn't", "mightn't", "mustn't", "I'm", "you're",
        "he's", "she's", "it's", "we're", "they're", "I've", "you've",
        "we've", "they've", "I'll", "you'll", "he'll", "she'll", "we'll",
        "they'll", "I'd", "you'd", "he'd", "she'd", "we'd", "they'd"
    ]

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize the stylometric analyzer.

        Args:
            spacy_model: Name of the spaCy model to use
        """
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"spaCy model '{spacy_model}' not found. Please download it with: python -m spacy download {spacy_model}")
            raise

        # Download NLTK data if not already present
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)

        self.stop_words = set(stopwords.words('english'))

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive stylometric analysis on text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary containing all stylometric features
        """
        doc = self.nlp(text)

        features = {
            **self._analyze_function_words(text),
            **self._analyze_ngrams(text),
            **self._analyze_contractions(text),
            **self._analyze_voice(doc),
            **self._analyze_sentence_starters(doc),
            **self._analyze_transition_words(text),
        }

        return features

    def _analyze_function_words(self, text: str) -> Dict[str, Any]:
        """
        Analyze function word frequencies.

        Args:
            text: Input text

        Returns:
            Dictionary with function word frequencies
        """
        # Tokenize and lowercase
        words = re.findall(r'\b\w+\b', text.lower())
        word_count = len(words)

        if word_count == 0:
            return {
                "function_word_frequencies": {},
                "function_word_ratio": 0.0,
            }

        # Count function words
        function_word_counts = {}
        total_function_words = 0

        for fw in self.FUNCTION_WORDS:
            count = words.count(fw)
            if count > 0:
                function_word_counts[fw] = count / word_count
                total_function_words += count

        return {
            "function_word_frequencies": function_word_counts,
            "function_word_ratio": total_function_words / word_count if word_count > 0 else 0.0,
        }

    def _analyze_ngrams(self, text: str, n_values: List[int] = [2, 3]) -> Dict[str, Any]:
        """
        Analyze n-gram patterns.

        Args:
            text: Input text
            n_values: List of n values to generate n-grams for

        Returns:
            Dictionary with n-gram statistics
        """
        # Tokenize
        words = re.findall(r'\b\w+\b', text.lower())

        if len(words) < 2:
            return {
                "bigram_patterns": {},
                "trigram_patterns": {},
            }

        result = {}

        for n in n_values:
            if len(words) >= n:
                n_grams = list(ngrams(words, n))
                n_gram_counts = Counter(n_grams)

                # Get top 20 most common n-grams
                top_ngrams = dict(n_gram_counts.most_common(20))

                # Convert tuples to strings for JSON serialization
                top_ngrams_str = {
                    ' '.join(gram): count for gram, count in top_ngrams.items()
                }

                if n == 2:
                    result["bigram_patterns"] = top_ngrams_str
                elif n == 3:
                    result["trigram_patterns"] = top_ngrams_str

        return result

    def _analyze_contractions(self, text: str) -> Dict[str, float]:
        """
        Analyze contraction usage rate.

        Args:
            text: Input text

        Returns:
            Dictionary with contraction metrics
        """
        # Count contractions
        contraction_count = 0
        for contraction in self.CONTRACTIONS:
            contraction_count += len(re.findall(r'\b' + re.escape(contraction) + r'\b', text, re.IGNORECASE))

        # Count total words
        word_count = len(re.findall(r'\b\w+\b', text))

        return {
            "contraction_rate": contraction_count / word_count if word_count > 0 else 0.0,
            "contraction_count": contraction_count,
        }

    def _analyze_voice(self, doc) -> Dict[str, float]:
        """
        Determine passive vs active voice ratio.

        Args:
            doc: spaCy Doc object

        Returns:
            Dictionary with voice statistics
        """
        passive_count = 0
        active_count = 0

        for sent in doc.sents:
            # Check for passive voice patterns
            # Passive voice typically has: auxiliary verb + past participle
            for token in sent:
                # Check if token is a past participle
                if token.tag_ == "VBN":
                    # Check if it has an auxiliary verb as a child or ancestor
                    ancestors = list(token.ancestors)
                    children = list(token.children)

                    for ancestor in ancestors:
                        if ancestor.lemma_ in ["be", "get"] and ancestor.pos_ == "AUX":
                            passive_count += 1
                            break
                    else:
                        for child in children:
                            if child.dep_ == "auxpass":
                                passive_count += 1
                                break

            # Count active voice (sentences with clear subjects and active verbs)
            has_subject = any(token.dep_ in ["nsubj", "nsubjpass"] for token in sent)
            has_verb = any(token.pos_ == "VERB" for token in sent)

            if has_subject and has_verb:
                active_count += 1

        total_sentences = passive_count + active_count
        passive_ratio = passive_count / total_sentences if total_sentences > 0 else 0.0

        return {
            "passive_voice_ratio": passive_ratio,
            "active_voice_ratio": 1 - passive_ratio if total_sentences > 0 else 0.0,
            "passive_constructions": passive_count,
        }

    def _analyze_sentence_starters(self, doc) -> Dict[str, Any]:
        """
        Analyze how sentences typically start.

        Args:
            doc: spaCy Doc object

        Returns:
            Dictionary with sentence starter patterns
        """
        starters = []

        for sent in doc.sents:
            tokens = [t for t in sent if not t.is_space and not t.is_punct]
            if tokens:
                first_token = tokens[0]
                starters.append(first_token.pos_)

        if not starters:
            return {"sentence_starter_pos": {}}

        starter_counts = Counter(starters)
        total = len(starters)

        starter_distribution = {
            pos: count / total for pos, count in starter_counts.items()
        }

        return {
            "sentence_starter_pos": starter_distribution,
        }

    def _analyze_transition_words(self, text: str) -> Dict[str, float]:
        """
        Analyze usage of transition words and phrases.

        Args:
            text: Input text

        Returns:
            Dictionary with transition word metrics
        """
        transition_words = [
            'however', 'moreover', 'furthermore', 'nevertheless', 'therefore',
            'thus', 'hence', 'consequently', 'additionally', 'meanwhile',
            'similarly', 'likewise', 'conversely', 'alternatively', 'specifically',
            'in fact', 'for example', 'for instance', 'in other words', 'in conclusion',
            'in summary', 'as a result', 'on the other hand', 'in contrast', 'in addition'
        ]

        text_lower = text.lower()
        word_count = len(re.findall(r'\b\w+\b', text))

        transition_count = 0
        for transition in transition_words:
            transition_count += len(re.findall(r'\b' + re.escape(transition) + r'\b', text_lower))

        return {
            "transition_word_rate": transition_count / word_count if word_count > 0 else 0.0,
            "transition_word_count": transition_count,
        }

    def analyze_multiple_samples(self, samples: List[str]) -> Dict[str, Any]:
        """
        Analyze multiple text samples and aggregate features.

        Args:
            samples: List of text samples

        Returns:
            Aggregated stylometric features across all samples
        """
        all_features = [self.analyze_text(sample) for sample in samples]

        # Aggregate features
        aggregated = {}

        if all_features:
            first_features = all_features[0]

            for key, value in first_features.items():
                if isinstance(value, (int, float)):
                    # Average numeric values
                    values = [f[key] for f in all_features if key in f and isinstance(f[key], (int, float))]
                    aggregated[key] = statistics.mean(values) if values else 0.0
                elif isinstance(value, dict):
                    # For nested dicts, merge and average
                    aggregated[key] = {}
                    all_keys = set()
                    for f in all_features:
                        if key in f and isinstance(f[key], dict):
                            all_keys.update(f[key].keys())

                    for nested_key in all_keys:
                        nested_values = [
                            f[key][nested_key] for f in all_features
                            if key in f and isinstance(f[key], dict) and nested_key in f[key]
                            and isinstance(f[key][nested_key], (int, float))
                        ]
                        if nested_values:
                            aggregated[key][nested_key] = statistics.mean(nested_values)

        return aggregated
