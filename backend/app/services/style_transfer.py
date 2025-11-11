"""
Style Transfer Engine.

Applies a user's specific writing style to text by:
- Matching sentence length patterns
- Adjusting vocabulary to user's typical words
- Replicating punctuation style
- Matching syntactic structures
- Adjusting passive/active voice ratios
"""
import re
import random
import statistics
from typing import Dict, List, Any, Optional, Tuple
import spacy
from app.models.style_profile import StyleProfile


class StyleTransferEngine:
    """Transfers writing style from a profile to input text."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize the style transfer engine.

        Args:
            spacy_model: Name of the spaCy model to use
        """
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"spaCy model '{spacy_model}' not found.")
            raise

    def apply_style(
        self,
        text: str,
        profile: StyleProfile,
        strength: float = 0.7,
    ) -> str:
        """
        Apply a user's writing style to text.

        Args:
            text: Input text to transform
            profile: StyleProfile to apply
            strength: How strongly to apply style (0-1)

        Returns:
            Transformed text
        """
        doc = self.nlp(text)

        # Apply transformations in sequence
        modified_text = text

        # 1. Adjust sentence lengths
        modified_text = self._adjust_sentence_lengths(
            modified_text, profile.linguistic_features, strength
        )

        # 2. Adjust punctuation style
        modified_text = self._adjust_punctuation(
            modified_text, profile.linguistic_features, strength
        )

        # 3. Match vocabulary formality
        modified_text = self._adjust_vocabulary_formality(
            modified_text, profile.stylometric_features, strength
        )

        # 4. Adjust passive/active voice
        modified_text = self._adjust_voice(
            modified_text, profile.stylometric_features, strength
        )

        return modified_text

    def _adjust_sentence_lengths(
        self, text: str, linguistic_features: Dict[str, Any], strength: float
    ) -> str:
        """
        Adjust sentence lengths to match user's patterns.

        Args:
            text: Input text
            linguistic_features: User's linguistic features
            strength: Transformation strength

        Returns:
            Modified text
        """
        doc = self.nlp(text)
        sentences = list(doc.sents)

        target_avg = linguistic_features.get("avg_sentence_length", 15)
        target_std = linguistic_features.get("sentence_length_std", 5)

        modified_sentences = []
        i = 0

        while i < len(sentences):
            sent = sentences[i]
            sent_length = len([t for t in sent if not t.is_punct and not t.is_space])

            # Check if sentence is too long
            if sent_length > target_avg + target_std and random.random() < strength:
                # Split the sentence
                split_sentences = self._split_sentence(sent)
                modified_sentences.extend(split_sentences)
                i += 1

            # Check if sentence is too short and can be merged
            elif (sent_length < target_avg - target_std and
                  i < len(sentences) - 1 and
                  random.random() < strength):
                # Merge with next sentence
                next_sent = sentences[i + 1]
                merged = self._merge_sentences(sent, next_sent)
                modified_sentences.append(merged)
                i += 2

            else:
                modified_sentences.append(sent.text)
                i += 1

        return " ".join(modified_sentences)

    def _split_sentence(self, sent) -> List[str]:
        """
        Split a long sentence into shorter ones.

        Args:
            sent: spaCy Span object

        Returns:
            List of sentence strings
        """
        # Find conjunction or comma to split on
        text = sent.text

        # Try to split on coordinating conjunctions
        for conj in [", and ", ", but ", ", or ", "; ", " – "]:
            if conj in text:
                parts = text.split(conj, 1)
                # Capitalize second part
                parts[1] = parts[1][0].upper() + parts[1][1:] if len(parts[1]) > 1 else parts[1].upper()
                # Ensure first part ends with period
                if not parts[0].endswith('.'):
                    parts[0] += '.'
                return parts

        # If no good split point, just return original
        return [text]

    def _merge_sentences(self, sent1, sent2) -> str:
        """
        Merge two sentences.

        Args:
            sent1: First sentence (spaCy Span)
            sent2: Second sentence (spaCy Span)

        Returns:
            Merged sentence string
        """
        text1 = sent1.text.rstrip('.!?')
        text2 = sent2.text

        # Choose a connector
        connectors = [" and ", ", and ", " – ", "; ", ", "]
        connector = random.choice(connectors)

        # Lowercase the second sentence start
        text2 = text2[0].lower() + text2[1:] if len(text2) > 1 else text2.lower()

        return text1 + connector + text2

    def _adjust_punctuation(
        self, text: str, linguistic_features: Dict[str, Any], strength: float
    ) -> str:
        """
        Adjust punctuation to match user's style.

        Args:
            text: Input text
            linguistic_features: User's linguistic features
            strength: Transformation strength

        Returns:
            Modified text
        """
        modified = text

        punct_patterns = linguistic_features.get("punctuation_patterns", {})

        # Adjust comma usage
        comma_rate = punct_patterns.get("comma_per_100_words", 5)
        current_commas = modified.count(',')
        words = len(modified.split())

        if words > 0:
            current_rate = (current_commas / words) * 100
            target_rate = comma_rate

            # If using too many commas
            if current_rate > target_rate * 1.5 and random.random() < strength:
                # Remove some commas
                modified = self._reduce_commas(modified, strength)

            # If using too few commas
            elif current_rate < target_rate * 0.5 and random.random() < strength:
                # Add some commas
                modified = self._add_commas(modified, strength)

        # Adjust em dash usage
        em_dash_rate = punct_patterns.get("em_dash_per_100_words", 0)
        if em_dash_rate > 0.5 and random.random() < strength:
            # User likes em dashes, replace some commas with em dashes
            modified = self._add_em_dashes(modified, strength)

        return modified

    def _reduce_commas(self, text: str, strength: float) -> str:
        """Remove some commas."""
        # Remove commas before 'and' and 'but' sometimes
        if random.random() < strength:
            text = re.sub(r',\s+(and|but)\s+', r' \1 ', text)
        return text

    def _add_commas(self, text: str, strength: float) -> str:
        """Add commas where appropriate."""
        # Add commas before 'and' and 'but' in compound sentences
        text = re.sub(r'\s+(and|but)\s+', r', \1 ', text)
        return text

    def _add_em_dashes(self, text: str, strength: float) -> str:
        """Replace some commas with em dashes."""
        # Find commas that could be em dashes (parenthetical phrases)
        pattern = re.compile(r',\s+([^,]+?),\s+')

        def replace_with_dash(match):
            if random.random() < strength * 0.3:
                return f" – {match.group(1)} – "
            return match.group(0)

        return pattern.sub(replace_with_dash, text)

    def _adjust_vocabulary_formality(
        self, text: str, stylometric_features: Dict[str, Any], strength: float
    ) -> str:
        """
        Adjust vocabulary to match user's formality level.

        Args:
            text: Input text
            stylometric_features: User's stylometric features
            strength: Transformation strength

        Returns:
            Modified text
        """
        # Check contraction rate
        contraction_rate = stylometric_features.get("contraction_rate", 0)

        modified = text

        # If user uses contractions, add them
        if contraction_rate > 0.02 and random.random() < strength:
            contractions_map = {
                "do not": "don't",
                "does not": "doesn't",
                "did not": "didn't",
                "is not": "isn't",
                "are not": "aren't",
                "was not": "wasn't",
                "were not": "weren't",
                "have not": "haven't",
                "has not": "hasn't",
                "had not": "hadn't",
                "will not": "won't",
                "would not": "wouldn't",
                "should not": "shouldn't",
                "could not": "couldn't",
                "cannot": "can't",
                "I am": "I'm",
                "you are": "you're",
                "it is": "it's",
                "we are": "we're",
                "they are": "they're",
            }

            for full, contraction in contractions_map.items():
                if random.random() < strength:
                    modified = re.sub(
                        r'\b' + full + r'\b',
                        contraction,
                        modified,
                        flags=re.IGNORECASE
                    )

        # If user doesn't use contractions, expand them
        elif contraction_rate < 0.01 and random.random() < strength:
            expansions_map = {
                "don't": "do not",
                "doesn't": "does not",
                "didn't": "did not",
                "isn't": "is not",
                "aren't": "are not",
                "wasn't": "was not",
                "weren't": "were not",
                "haven't": "have not",
                "hasn't": "has not",
                "hadn't": "had not",
                "won't": "will not",
                "wouldn't": "would not",
                "shouldn't": "should not",
                "couldn't": "could not",
                "can't": "cannot",
                "I'm": "I am",
                "you're": "you are",
                "it's": "it is",
                "we're": "we are",
                "they're": "they are",
            }

            for contraction, full in expansions_map.items():
                if random.random() < strength:
                    modified = re.sub(
                        r'\b' + re.escape(contraction) + r'\b',
                        full,
                        modified,
                        flags=re.IGNORECASE
                    )

        return modified

    def _adjust_voice(
        self, text: str, stylometric_features: Dict[str, Any], strength: float
    ) -> str:
        """
        Adjust passive/active voice ratio.

        Args:
            text: Input text
            stylometric_features: User's stylometric features
            strength: Transformation strength

        Returns:
            Modified text
        """
        passive_ratio = stylometric_features.get("passive_voice_ratio", 0.1)

        # For now, this is a simplified version
        # In production, would use more sophisticated passive-to-active conversion

        doc = self.nlp(text)
        sentences = []

        for sent in doc.sents:
            sent_text = sent.text

            # Detect passive constructions
            has_passive = False
            for token in sent:
                if token.tag_ == "VBN":  # Past participle
                    for ancestor in token.ancestors:
                        if ancestor.lemma_ == "be":
                            has_passive = True
                            break

            # If user rarely uses passive voice, try to convert to active
            if has_passive and passive_ratio < 0.1 and random.random() < strength:
                # Simple passive to active conversion
                # "The ball was thrown" -> "Someone threw the ball"
                # This is simplified - production would need better conversion
                sent_text = self._simple_passive_to_active(sent)

            sentences.append(sent_text)

        return " ".join(sentences)

    def _simple_passive_to_active(self, sent) -> str:
        """
        Simple passive to active voice conversion.

        Args:
            sent: spaCy Span object

        Returns:
            Converted sentence (best effort)
        """
        # This is a simplified version
        # Production would need more sophisticated conversion
        text = sent.text

        # Basic patterns
        patterns = [
            (r'(\w+)\s+was\s+(\w+ed)', r'Someone \2 \1'),
            (r'(\w+)\s+were\s+(\w+ed)', r'People \2 \1'),
            (r'(\w+)\s+is\s+(\w+ed)', r'Someone \2s \1'),
        ]

        for pattern, replacement in patterns:
            if re.search(pattern, text):
                text = re.sub(pattern, replacement, text, count=1)
                break

        return text
