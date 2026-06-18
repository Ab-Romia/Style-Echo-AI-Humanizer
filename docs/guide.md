# How to build an authorship-stylometry and style-transfer system from scratch

This is the long version of VoicePrint: how the thing actually works, written so someone getting into NLP can rebuild it. Every code snippet here is pulled from the real modules in this repo, not pseudocode. If you read it end to end you will be able to measure a writing fingerprint from a handful of samples, score how far a new piece of text sits from it, and adapt a draft toward that voice while keeping the meaning.

## 1. What this is, and what you will be able to build

VoicePrint turns a small set of an author's own writing into a measured fingerprint, then uses that fingerprint two ways: to score how much a draft sounds like the author, and to guide a rewrite back toward that voice. The demo has two tabs. The first takes at least three samples and at least 500 words, then shows a radar of the author's normalized style axes plus a consistency number. The second takes a draft, shows the voice match before and after adaptation, a per-axis radar of draft against voice, and a sentence-level diff coloring each sentence by whether its voice match improved or regressed.

## 2. Why I built it

I wanted to measure my own voice instead of guessing at it. When I write fast, my drafts drift: longer sentences, fewer contractions, more hedging. I could feel the drift but I could not see it. Stylometry already knows how to put a number on a writing identity, so I built a tool that learns my voice from my own samples and tells me, concretely, how far a given draft has wandered, then helps pull it back. The target is always my own measured voice, never anyone else's.

## 3. Intuition: what a writing fingerprint is

When you think about someone's writing style, you probably think about word choice and subject matter. Those are the worst features to use, because they track topic. An essay about ships and an essay about gardens differ on content words even when the same person wrote both.

The features that carry identity are the ones the writer is not thinking about. Function words are the clearest case: how often someone reaches for "the", "of", "but", "however", whether they prefer "while" or "whereas". Nobody chooses these consciously, and they barely move with subject matter, so they leak personal habit rather than topic. Character n-grams are similar: the short overlapping character sequences in a text capture spelling habits, morphology, contraction patterns, and punctuation rhythm without keying on what the text is about. That is the whole game in stylometry. Find the signal that survives a topic change, and ignore the signal that does not.

## 4. The data, and the honest trap

The trap in any authorship experiment is topic leakage. If you split a single book into random chunks and put some chunks in training and some in testing, your classifier can ace the test by memorizing that book's vocabulary, and you learn nothing about whether it generalizes to a new piece of writing by the same author.

The experiment in `experiments/authorship_attribution.py` avoids most of that by splitting by work, not by random chunk. The corpus is NLTK Gutenberg, cut into non-overlapping 400-word chunks, with five authors:

| Author      | Train docs | Test docs | Test source                  |
| ----------- | ---------- | --------- | ---------------------------- |
| Austen      | 726        | 353       | held-out work (sense)        |
| Carroll     | 59         | 26        | tail chunk of same work      |
| Chesterton  | 457        | 173       | held-out work (thursday)     |
| Melville    | 456        | 196       | tail chunk of same work      |
| Shakespeare | 157        | 57        | held-out work (macbeth)      |

For authors with two or more works (Austen, Chesterton, Shakespeare) the classifier trains on some of their works and is tested on a different, fully held-out work. That removes the easy win of memorizing one book's vocabulary. Carroll and Melville have only one work each in the corpus, so their test set is the last 30 percent of that one book. Train and test there come from the same text, which leaks topic, so their per-author scores are optimistic and should be read as a soft upper bound. The point of writing this down is that you should always know which of your numbers are honest and which are not. See `experiments/results.md` for the full breakdown.

## 5. Features, one family at a time

The interpretable backbone lives in two modules: `linguistic_analyzer.py` for surface and syntactic features, `stylometric_analyzer.py` for the authorship-flavored ones, plus a small `char_ngram.py` profiler.

### Lexical stats and type-token ratio

Sentence length and lexical diversity are cheap and informative. Type-token ratio is unique words over total words: high means a varied vocabulary, low means the writer repeats common words. From `linguistic_analyzer.py`:

```python
def _analyze_lexical_diversity(self, doc) -> Dict[str, float]:
    tokens = [token.text.lower() for token in doc
              if not token.is_punct and not token.is_space]
    if not tokens:
        return {"type_token_ratio": 0.0, "num_unique_words": 0, "num_total_words": 0}
    unique_tokens = set(tokens)
    return {
        "type_token_ratio": len(unique_tokens) / len(tokens),
        "num_unique_words": len(unique_tokens),
        "num_total_words": len(tokens),
    }
```

What it captures: vocabulary range. What it costs: TTR is sensitive to text length, since longer texts naturally repeat more, so it is best compared across samples of similar size. Sentence length comes from `statistics.mean` over `len(sent)` for each `doc.sents`, with the standard deviation kept too, because how much your sentence length varies is itself a habit.

### Character n-grams

Character n-grams are the strongest single cross-domain authorship family. The profiler in `char_ngram.py` fits a tf-idf space on the author's samples and stores their mean vector:

```python
self._vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=self.ngram_range,   # (2, 4)
    min_df=self.min_df,
)
matrix = self._vectorizer.fit_transform(cleaned)
self._author_vector = np.asarray(matrix.mean(axis=0)).ravel()
```

The `char_wb` analyzer builds n-grams from inside word boundaries, so it picks up word edges and spacing as well as internal letter patterns. A candidate text is scored by cosine to the author centroid. What it captures: morphology, spelling and contraction habits, punctuation rhythm, all without content words. What it costs: the feature space is large and a little opaque, and it needs enough text to be stable.

### Function-word vectors

This is the heart of topic-independent stylometry. `stylometric_analyzer.py` defines a fixed 130-word list so every profile produces a vector of the same length in the same order, which is what lets you compare two writers coordinate by coordinate:

```python
words = re.findall(r'\b\w+\b', text.lower())
word_count = len(words)
word_freq = Counter(words)
function_word_vector = [
    word_freq.get(fw, 0) / word_count for fw in FUNCTION_WORD_LIST
]
```

Each coordinate is the relative frequency of one function word. What it captures: unconscious grammatical habit, the part of style that survives a topic change. What it costs: function words alone cannot separate every author, as the error analysis below shows.

### Readability and punctuation

Readability indices (Flesch reading ease, Flesch-Kincaid grade, Gunning fog, and a couple more) come straight from `textstat` and place a text on an easy-to-dense scale. Punctuation is counted per 100 words: commas, semicolons, colons, exclamation marks, question marks. The comma and semicolon rates turn out to matter a lot for how a voice feels, and the rule rewriter later nudges comma rate directly. What these capture: register and rhythm. What they cost: little; they are fast string counts.

## 6. A simple interpretable baseline

Before reaching for anything neural, fit a linear model on these features and see how far it gets. The experiment uses multinomial logistic regression with a fixed seed and reports three conditions:

| Condition      | Macro-F1 | Accuracy |
| -------------- | -------- | -------- |
| char n-grams   | 0.996    | 0.999    |
| function words | 0.684    | 0.889    |
| combined       | 0.996    | 0.999    |

Read these carefully. The char n-gram score of 0.996 is partly real and partly an artifact. The five authors here are stylistically far apart, and two of them leak topic through the single-work split, so the task is easier than open-world attribution. Do not quote 0.996 as a general accuracy. The result worth keeping is the function-word condition. With no content words at all, a 130-dimensional frequency vector reaches macro-F1 0.684 and accuracy 0.889 against a 0.20 five-class baseline. That gap is the whole thesis of stylometry: identity lives in the words you do not think about.

## 7. Error analysis

The honest failure is in the function-word condition. The most confused pair is Carroll mistaken for Chesterton: 16 test documents truly by Carroll were predicted as Chesterton. The confusion matrix under function words (rows true, columns predicted):

| true \ pred | Austen | Carroll | Chesterton | Melville | Shakespeare |
| ----------- | ------ | ------- | ---------- | -------- | ----------- |
| Austen      | 348    | 0       | 1          | 4        | 0           |
| Carroll     | 7      | 0       | 16         | 3        | 0           |
| Chesterton  | 2      | 0       | 159        | 12       | 0           |
| Melville    | 5      | 0       | 13         | 176      | 2           |
| Shakespeare | 6      | 0       | 6          | 12       | 33          |

Carroll has the fewest training documents (59) and never gets predicted at all; the model has not seen enough of his function-word habits to carve out a region for him, so his test documents fall into the nearest dense neighbor, which is Chesterton's English prose. This is the expected limit of function words on their own. They are a strong topic-independent signal, not a complete one, which is exactly why VoicePrint pairs them with a neural embedding.

## 8. The neural fingerprint

Interpretable features are honest and readable, but they miss a lot of what makes a voice. For the fingerprint itself, VoicePrint uses StyleDistance, a 2024 style embedding built to be content-independent: it is trained so that two texts in the same voice on different topics land near each other, and two texts on the same topic in different voices land apart. That is precisely the property a fingerprint needs.

You build an author centroid by embedding each sample, L2-normalizing, averaging, and normalizing again so cosine reduces to a dot product. From `style_embedding.py`:

```python
embeddings = self.embed(samples)
normalized = self._normalize_rows(embeddings)
centroid = normalized.mean(axis=0)
norm = np.linalg.norm(centroid)
return centroid / norm if norm else centroid
```

Voice match for a candidate is then cosine to that centroid, clamped to [0, 1]:

```python
vector = self._normalize_rows(self.embed([text]))[0]
cosine = float(np.dot(vector, centroid / np.linalg.norm(centroid)))
return max(0.0, min(1.0, cosine))
```

The same method works per sentence, which is what powers the diff later. If you want to read further into content-independent author representations, the work around LUAR (universal author representations trained at scale), the Wegmann et al. style embeddings (trained to ignore topic with a contrastive setup), and LISA (style descriptions used as an interpretable style space) all attack the same problem from different angles, and StyleDistance sits in that lineage.

## 9. Style transfer as content-preserving rewriting

Once you can measure voice, adapting toward it is a rewriting problem, not a generation problem. The meaning is fixed; only the style moves.

The in-context approach in `llm_rewriter.py` does this in a single model call. The prompt carries three pieces: the source draft, three to five real excerpts from the author's own samples as exemplars, and the measured profile rendered as plain-English constraints. The constraints come from `profile_to_constraints.py`, which turns numbers into instructions a model can follow:

```python
avg_len = linguistic.get("avg_sentence_length")
if avg_len:
    lines.append(
        f"- Average sentence length is about {avg_len:.0f} words; "
        "keep sentences in that range."
    )
```

The prompt then asks for a meaning-preserving rewrite and nothing else:

```python
"Rewrite the draft below so it reads in this author's voice. "
"Preserve the meaning, the facts, and the structure of the "
"argument. Do not add new claims. Return only the rewritten draft."
```

The client is OpenAI-compatible, so the same path runs against OpenAI or OpenRouter by switching `base_url`. No key is shipped; it is read from an argument or `OPENAI_API_KEY`, and if none is present the rewriter raises and the caller falls back.

The fallback is `rule_rewriter.py`, a deterministic rewriter that only touches two signals it can move without breaking meaning: contraction usage toward the author's measured rate, and comma rate toward the author's comma rate. It expands or contracts where the author's rate clearly calls for it, adds or drops commas only before coordinating conjunctions, and leaves the text alone when the signal is ambiguous.

This is a clean break from where the repo started. The 2021 prototype tried to transform style with regex synonym swaps and random punctuation insertion, and it even injected deliberate misspellings. That approach is a dead end: synonym swaps change meaning, random punctuation is not a style, and injected typos are just damage. Word-level surface edits cannot carry voice, because voice lives in structure and rhythm, not in a thesaurus lookup. Modern in-context rewriting works because the model sees real exemplars and measured constraints together and rewrites the whole passage at once, with the rule rewriter kept only for the two edits a deterministic pass can make safely.

## 10. How the result is evaluated

A rewrite is only useful if it moves voice without losing meaning, so VoicePrint measures both. Voice match is computed before and after with StyleDistance cosine to the author centroid. Meaning preservation is checked against the source draft, not against any reference text, since the only thing the rewrite must preserve is what the author actually said. The per-sentence diff in `feedback.py` scores each adapted sentence against the source sentence in the same position and labels it improved, regressed, or same, with a small dead zone so tiny numeric wobble does not flip the label:

```python
if score > before_score + threshold:
    label = "improved"
elif score < before_score - threshold:
    label = "regressed"
else:
    label = "same"
```

That gives an honest, sentence-by-sentence picture instead of a single feel-good number.

## 11. Limits and ethics

A few things to keep straight. This works on your own writing; you analyze and adapt text you wrote. It is voice adaptation, not impersonation, and the rewrite optimizes toward your own measured voice, a positive target you can inspect, not away from anything. Authorship signals are probabilistic: a high voice match is evidence, not proof, and the measured numbers in this repo come from a small five-author set with a known topic-leak caveat. Stylometry is dual-use, like most identity tools, so the honest framing matters: measure your own voice, keep the caveats visible, and do not oversell a score.

## Run it yourself

Setup and the live demo link are in the [README](../README.md). To regenerate the measured numbers cited here, run the experiment script:

```bash
python experiments/authorship_attribution.py
```
