# Authorship attribution results

## Dataset

Corpus: NLTK Gutenberg. Documents are non-overlapping 400-word chunks.

| Author | Train docs | Test docs | Test source |
| --- | --- | --- | --- |
| Austen | 726 | 353 | held-out work (sense) |
| Carroll | 59 | 26 | tail chunk of same work (topic leak) |
| Chesterton | 457 | 173 | held-out work (thursday) |
| Melville | 456 | 196 | tail chunk of same work (topic leak) |
| Shakespeare | 157 | 57 | held-out work (macbeth) |
| TOTAL | 1855 | 805 | |

Authors: 5 (Austen, Carroll, Chesterton, Melville, Shakespeare).

## Split policy

Authors with two or more works (Austen, Chesterton, Shakespeare) are split by work: the classifier trains on a subset of an author's works and is tested on a different, fully held-out work by the same author. This removes the easy win of memorizing the vocabulary of a single book.

Single-work authors (Melville, Carroll) have no second work to hold out, so their test documents are the final 30 percent of the one work. Train and test there come from the same book, which leaks topic and vocabulary. Their per-author scores are therefore optimistic and should be read as a soft upper bound, not as cross-domain performance.

## Conditions

| Condition | Macro-F1 | Accuracy |
| --- | --- | --- |
| char n-grams | 0.996 | 0.999 |
| function words | 0.684 | 0.889 |
| combined | 0.996 | 0.999 |

Feature settings: character n-grams use TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), min_df=2, sublinear_tf=True). Function words use relative frequencies over a 130-word list. Classifier is multinomial LogisticRegression with fixed random_state=42.

## Interpretation

The character n-gram score is very high. That is partly a real result and partly an artifact of this small five-author set. The five authors here are stylistically far apart (Austen, Chesterton, Shakespeare, Melville, Carroll), and two of them are single-work authors whose test documents leak topic. Both effects make the task easier than open-world authorship attribution, so the char n-gram number should not be read as a general accuracy claim. The more transferable lesson is the function-word result: with no content words at all, a 130-dimensional function-word frequency vector still separates these authors well above chance (20 percent for five classes), which is the topic-independent signal stylometry relies on.

## Honest failure case

Under the topic-independent function-word condition, the most-confused author pair is Carroll mistaken for Chesterton: 16 test documents truly by Carroll were predicted as Chesterton. Function words alone cannot fully separate every pair, which is the expected and honest limitation of this feature on its own.

Per-author confusion under the function-word condition (rows are true authors, columns are predicted):

| true \ pred | Austen | Carroll | Chesterton | Melville | Shakespeare |
| --- | --- | --- | --- | --- | --- |
| Austen | 348 | 0 | 1 | 4 | 0 |
| Carroll | 7 | 0 | 16 | 3 | 0 |
| Chesterton | 2 | 0 | 159 | 12 | 0 |
| Melville | 5 | 0 | 13 | 176 | 2 |
| Shakespeare | 6 | 0 | 6 | 12 | 33 |

