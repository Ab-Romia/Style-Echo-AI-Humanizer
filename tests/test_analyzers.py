"""
Unit tests for the interpretable analyzers.

These import the heavy NLP stack (spaCy, scikit-learn), so they run once the
virtual environment is prepared. They check the parts of the engine that should
be deterministic and explainable.
"""
import pytest

from app.services.linguistic_analyzer import LinguisticAnalyzer
from app.services.stylometric_analyzer import (
    FUNCTION_WORD_LIST,
    StylometricAnalyzer,
)
from app.services.char_ngram import CharNgramProfiler


FIXED_SAMPLE = (
    "I write short sentences. I keep things plain and direct. When I have a "
    "longer thought, I let it run, but I do not overdo the commas."
)

# Two same-author samples and one clearly different author.
AUTHOR_A_1 = (
    "I keep my sentences short. I say what I mean. I do not dress it up. Plain "
    "words do the job, so I use them and move on."
)
AUTHOR_A_2 = (
    "I like a direct line. I trim what I can. I read it back, then I cut more. "
    "Short beats clever, every time."
)
AUTHOR_B = (
    "Notwithstanding the aforementioned considerations, one must acknowledge "
    "the multifaceted nature of the phenomenon under examination, which, upon "
    "careful deliberation, reveals considerable complexity."
)


@pytest.fixture(scope="module")
def linguistic():
    return LinguisticAnalyzer()


@pytest.fixture(scope="module")
def stylometric():
    return StylometricAnalyzer()


def test_linguistic_extracts_expected_keys(linguistic):
    features = linguistic.analyze_text(FIXED_SAMPLE)
    for key in (
        "avg_sentence_length",
        "type_token_ratio",
        "num_total_words",
        "flesch_reading_ease",
        "avg_dependency_depth",
    ):
        assert key in features
    assert features["num_total_words"] > 0
    assert 0.0 <= features["type_token_ratio"] <= 1.0


def test_linguistic_empty_text_is_guarded(linguistic):
    features = linguistic.analyze_text("")
    assert features["num_total_words"] == 0
    assert features["avg_sentence_length"] == 0.0


def test_multiple_samples_emit_std_fields(linguistic):
    agg = linguistic.analyze_multiple_samples([AUTHOR_A_1, AUTHOR_A_2])
    assert "avg_sentence_length" in agg
    assert "avg_sentence_length_std" in agg
    assert agg["avg_sentence_length_std"] >= 0.0
    assert agg["total_word_count"] > 0


def test_function_word_vector_is_fixed_length(stylometric):
    features = stylometric.analyze_text(FIXED_SAMPLE)
    vector = features["function_word_vector"]
    assert len(vector) == len(FUNCTION_WORD_LIST)
    assert features["function_word_vector_order"] == FUNCTION_WORD_LIST
    assert all(isinstance(v, float) for v in vector)


def test_function_word_vector_fixed_across_texts(stylometric):
    v1 = stylometric.analyze_text(AUTHOR_A_1)["function_word_vector"]
    v2 = stylometric.analyze_text(AUTHOR_B)["function_word_vector"]
    assert len(v1) == len(v2) == len(FUNCTION_WORD_LIST)


def test_char_ngram_same_author_scores_higher():
    profiler = CharNgramProfiler().fit([AUTHOR_A_1])
    same_author = profiler.similarity(AUTHOR_A_2)
    other_author = profiler.similarity(AUTHOR_B)
    assert same_author > other_author


def test_char_ngram_requires_fit():
    profiler = CharNgramProfiler()
    with pytest.raises(RuntimeError):
        profiler.similarity("anything")
