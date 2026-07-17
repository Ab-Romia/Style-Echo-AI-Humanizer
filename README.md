---
title: VoicePrint
emoji: 🖋️
colorFrom: green
colorTo: indigo
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
pinned: false
short_description: Measure your writing voice and adapt your own drafts toward it.
preload_from_hub:
  - StyleDistance/styledistance
---

# VoicePrint

VoicePrint measures the stylometric fingerprint of how you write, learned from your own samples, then adapts your own drafts toward that measured voice. It is for writers who want their drafts to sound like them, and for anyone curious how much of a writing identity survives once you strip the topic out.

## Try it

Live demo: [huggingface.co/spaces/Ab-Romia/voiceprint](https://huggingface.co/spaces/Ab-Romia/voiceprint)

Run it locally:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # if the model is not already present
python app.py
```

Then open the local URL Gradio prints (default http://localhost:7860). Paste at least three samples of your own writing to build a profile, then paste a draft to adapt.

## One measured result

With no content words at all, just the relative frequencies of 130 function words, a plain logistic-regression classifier reached macro-F1 0.684 and accuracy 0.889 separating five authors. Five-class chance is 0.20, so the function-word signal sits well above it. That is the number I trust most here, because function words do not track topic, so the signal carries across subjects rather than memorizing one book's vocabulary.

## Why I built this

I kept noticing that my own drafts drifted out of my voice when I wrote them in a hurry, and I wanted a way to measure that drift instead of guessing at it. Authorship stylometry already has the tools to put a number on a writing fingerprint. So I built something that measures my voice from my own writing, scores how far a draft sits from it, and nudges the draft back, without pretending to be anyone else.

## How it works

```
your samples
   -> stylometric profile     interpretable features: sentence stats, lexical
                              diversity, function words, char n-grams,
                              readability, punctuation
   -> StyleDistance           neural style centroid (content-independent);
      fingerprint             voice match = cosine to that centroid
   -> in-context rewrite       source draft + your own exemplars + rendered
      (or rule fallback)       constraints, one model call; no key falls back
                              to a deterministic rule rewriter
   -> validation              re-score voice match before vs after, check
      and report              meaning preserved against your draft, per-sentence diff
```

The interpretable side stays readable: you can see the average sentence length, the contraction rate, the punctuation habits. The neural side handles what those features miss. StyleDistance is a 2024 style embedding trained to be content-independent, so two passages on different topics in the same voice land near each other. I build a centroid from your samples and score a candidate by cosine to it.

## Results

Corpus: NLTK Gutenberg, non-overlapping 400-word chunks, five authors (Austen, Carroll, Chesterton, Melville, Shakespeare). Authors with more than one work are tested on a fully held-out work; the two single-work authors are tested on the tail of their only book.

| Condition      | Macro-F1 | Accuracy |
| -------------- | -------- | -------- |
| char n-grams   | 0.996    | 0.999    |
| function words | 0.684    | 0.889    |
| combined       | 0.996    | 0.999    |

Feature settings: character n-grams use `TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), min_df=2, sublinear_tf=True)`; function words use relative frequencies over a 130-word list; the classifier is multinomial logistic regression with `random_state=42`.

The honest reading: the 0.996 char n-gram score is inflated. This is a small five-author set whose authors are stylistically far apart, and two of them are single-work authors whose test text leaks topic and vocabulary from training. Both effects make the task easier than open-world authorship attribution, so do not read 0.996 as a general accuracy claim. The transferable result is the function-word condition: with zero content words, a 130-dimensional frequency vector still separates these authors well above the 0.20 five-class baseline. That is the topic-independent signal stylometry relies on.

## What it does not do

- The evaluation set is small (five authors) and the numbers above are not open-world accuracy.
- Two authors are single-work, so their per-author scores leak topic and read as a soft upper bound, not cross-domain performance.
- The rewrite is only as good as the model key you bring. With no key, the rule rewriter makes a few conservative edits (contraction rate and comma rate) and otherwise leaves your text alone.
- This is voice adaptation on your own writing, not impersonation of someone else, and authorship signals are probabilistic, not proof.

## Reproduce the experiment

```bash
python experiments/authorship_attribution.py
```

This downloads the NLTK Gutenberg corpus, splits by work, fits the three feature conditions with a fixed seed, and reports the macro-F1, accuracy, and the per-author confusion matrix recorded in `experiments/results.md`.

## Tech

Python, Gradio, spaCy, textstat, scikit-learn, NLTK, sentence-transformers with StyleDistance, an OpenAI-compatible client for the bring-your-own-key rewrite (OpenAI or OpenRouter), and FastAPI for the backend service.

---

Built by [Ab-Romia](https://github.com/Ab-Romia/VoicePrint).
