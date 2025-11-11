"""
AI Detection Removal Pipeline.

Analyzes AI-generated text for common AI tells and introduces strategic
perturbations to make the text appear more human-written.
"""
import re
import random
from typing import List, Dict, Any, Tuple
import spacy


class AIDetectorRemover:
    """Removes common AI detection patterns from text."""

    # Common AI hedging phrases
    AI_HEDGING_PHRASES = [
        "it's important to note",
        "it's worth mentioning",
        "it should be noted",
        "it's worth noting",
        "importantly",
        "significantly",
        "notably",
        "it is crucial to understand",
        "one must consider",
        "it is essential to recognize",
    ]

    # Natural filler words to inject
    FILLER_WORDS = [
        "well", "actually", "basically", "honestly", "frankly",
        "I mean", "you know", "kind of", "sort of", "pretty much",
        "like", "just", "really", "literally", "essentially"
    ]

    # Common typo patterns (intentionally limited to avoid breaking meaning)
    COMMON_TYPOS = {
        "the": ["teh"],
        "and": ["adn"],
        "there": ["thre"],
        "their": ["thier"],
        "receive": ["recieve"],
        "definitely": ["definately"],
        "separate": ["seperate"],
        "occurrence": ["occurence"],
    }

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize the AI detector remover.

        Args:
            spacy_model: Name of the spaCy model to use
        """
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"spaCy model '{spacy_model}' not found.")
            raise

    def detect_ai_tells(self, text: str) -> Dict[str, Any]:
        """
        Detect common AI writing patterns in text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with detection metrics
        """
        doc = self.nlp(text)
        sentences = list(doc.sents)

        # Check for perfect grammar (very few errors)
        # Check for repetitive sentence structures
        sentence_structures = []
        for sent in sentences:
            structure = " ".join([token.pos_ for token in sent if not token.is_punct])
            sentence_structures.append(structure)

        unique_structures = len(set(sentence_structures))
        structure_diversity = unique_structures / len(sentence_structures) if sentence_structures else 1.0

        # Check for lists of three (AI loves these)
        lists_of_three = len(re.findall(r'\w+,\s*\w+,\s*and\s+\w+', text))

        # Check for hedging language
        hedging_count = 0
        text_lower = text.lower()
        for phrase in self.AI_HEDGING_PHRASES:
            hedging_count += text_lower.count(phrase)

        # Check for overly balanced parallel construction
        sentence_lengths = [len(sent) for sent in sentences]
        length_variance = sum((l - sum(sentence_lengths) / len(sentence_lengths)) ** 2
                              for l in sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0

        return {
            "structure_diversity": structure_diversity,
            "lists_of_three_count": lists_of_three,
            "hedging_phrase_count": hedging_count,
            "sentence_length_variance": length_variance,
            "ai_probability": self._calculate_ai_probability(
                structure_diversity, lists_of_three, hedging_count, length_variance
            ),
        }

    def _calculate_ai_probability(
        self, structure_diversity: float, lists_of_three: int,
        hedging_count: int, length_variance: float
    ) -> float:
        """
        Calculate a simple AI probability score.

        Args:
            structure_diversity: Diversity of sentence structures
            lists_of_three: Count of "A, B, and C" patterns
            hedging_count: Count of hedging phrases
            length_variance: Variance in sentence lengths

        Returns:
            AI probability score (0-1)
        """
        score = 0.0

        # Low structure diversity suggests AI
        if structure_diversity < 0.5:
            score += 0.3

        # Multiple lists of three suggest AI
        if lists_of_three > 2:
            score += 0.2

        # Hedging language suggests AI
        if hedging_count > 0:
            score += min(hedging_count * 0.1, 0.3)

        # Low variance suggests AI
        if length_variance < 10:
            score += 0.2

        return min(score, 1.0)

    def remove_ai_patterns(self, text: str, strength: float = 0.7) -> str:
        """
        Remove AI patterns and introduce human-like variations.

        Args:
            text: Input text
            strength: How aggressively to modify (0-1)

        Returns:
            Modified text
        """
        # Remove hedging phrases
        modified_text = self._remove_hedging_phrases(text)

        # Break up overly symmetrical structures
        modified_text = self._break_symmetry(modified_text, strength)

        # Add natural filler words
        modified_text = self._add_filler_words(modified_text, strength)

        # Introduce minor typos
        if strength > 0.5:
            modified_text = self._add_strategic_typos(modified_text, strength)

        # Vary sentence lengths more dramatically
        modified_text = self._vary_sentence_length(modified_text, strength)

        return modified_text

    def _remove_hedging_phrases(self, text: str) -> str:
        """Remove common AI hedging phrases."""
        modified = text

        for phrase in self.AI_HEDGING_PHRASES:
            # Remove the phrase and clean up punctuation
            pattern = re.compile(r'\b' + re.escape(phrase) + r'\b,?\s*', re.IGNORECASE)
            modified = pattern.sub('', modified)

        # Clean up double spaces
        modified = re.sub(r'\s+', ' ', modified)

        return modified

    def _break_symmetry(self, text: str, strength: float) -> str:
        """Break up overly balanced parallel construction."""
        # Replace lists of three with varied constructions
        def replace_list_of_three(match):
            if random.random() > strength:
                return match.group(0)

            parts = match.group(0).split(',')
            if len(parts) == 3:
                # Randomly restructure
                options = [
                    f"{parts[0].strip()} and {parts[1].strip()}, plus {parts[2].strip()}",
                    f"{parts[0].strip()}, {parts[1].strip()} – {parts[2].strip()}",
                    f"{parts[0].strip()} and {parts[1].strip()} (and {parts[2].strip()})",
                ]
                return random.choice(options)

            return match.group(0)

        pattern = re.compile(r'\w+,\s*\w+,\s*and\s+\w+')
        modified = pattern.sub(replace_list_of_three, text)

        return modified

    def _add_filler_words(self, text: str, strength: float) -> str:
        """Add natural filler words to sentences."""
        doc = self.nlp(text)
        sentences = list(doc.sents)

        modified_sentences = []

        for sent in sentences:
            sent_text = sent.text

            # Randomly add filler words (based on strength)
            if random.random() < strength * 0.3:  # 30% chance at full strength
                filler = random.choice(self.FILLER_WORDS)

                # Insert at the beginning or after first few words
                if random.random() < 0.5:
                    sent_text = f"{filler.capitalize()}, {sent_text[0].lower()}{sent_text[1:]}"
                else:
                    # Insert after first clause
                    words = sent_text.split()
                    if len(words) > 3:
                        insert_pos = random.randint(2, min(5, len(words) - 1))
                        words.insert(insert_pos, filler + ",")
                        sent_text = " ".join(words)

            modified_sentences.append(sent_text)

        return " ".join(modified_sentences)

    def _add_strategic_typos(self, text: str, strength: float) -> str:
        """Add occasional typos to match human error rate."""
        words = text.split()
        typo_rate = min(strength * 0.02, 0.02)  # Max 2% typo rate

        modified_words = []
        for word in words:
            # Check if we should add a typo
            if random.random() < typo_rate:
                word_lower = word.lower()

                # Check if we have a common typo for this word
                if word_lower in self.COMMON_TYPOS:
                    typo = random.choice(self.COMMON_TYPOS[word_lower])

                    # Preserve capitalization
                    if word[0].isupper():
                        typo = typo.capitalize()

                    modified_words.append(typo)
                    continue

            modified_words.append(word)

        return " ".join(modified_words)

    def _vary_sentence_length(self, text: str, strength: float) -> str:
        """Vary sentence lengths more dramatically."""
        doc = self.nlp(text)
        sentences = list(doc.sents)

        if len(sentences) < 2:
            return text

        modified_sentences = []
        i = 0

        while i < len(sentences):
            sent = sentences[i]
            sent_text = sent.text.strip()

            # Randomly merge short sentences
            if (i < len(sentences) - 1 and
                len(sent_text.split()) < 10 and
                random.random() < strength * 0.3):

                next_sent = sentences[i + 1].text.strip()

                # Merge with varied connectors
                connectors = [" and ", " – ", ", ", "; "]
                connector = random.choice(connectors)

                # Remove period from first sentence
                sent_text = sent_text.rstrip('.')

                merged = sent_text + connector + next_sent[0].lower() + next_sent[1:]
                modified_sentences.append(merged)
                i += 2
            else:
                modified_sentences.append(sent_text)
                i += 1

        return " ".join(modified_sentences)

    def analyze_and_remove(self, text: str, strength: float = 0.7) -> Dict[str, Any]:
        """
        Analyze text for AI patterns and remove them.

        Args:
            text: Input text
            strength: Modification strength (0-1)

        Returns:
            Dictionary with original metrics, modified text, and new metrics
        """
        # Analyze original
        original_metrics = self.detect_ai_tells(text)

        # Remove patterns
        modified_text = self.remove_ai_patterns(text, strength)

        # Analyze modified
        modified_metrics = self.detect_ai_tells(modified_text)

        return {
            "original_text": text,
            "modified_text": modified_text,
            "original_metrics": original_metrics,
            "modified_metrics": modified_metrics,
            "improvement": original_metrics["ai_probability"] - modified_metrics["ai_probability"],
        }
