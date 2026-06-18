# VoicePrint Rebuild: Design Spec

Date: 2026-06-18
Owner: Abdelrahman Abouroumia (Ab-Romia)
Status: Approved direction, ready for implementation plan

## Goal

Transform the `Style-Echo-AI-Humanizer` repository from an "AI text humanizer that
bypasses detectors" into **VoicePrint**, a credible authorship-stylometry engine that
measures a writer's voice from their own samples and helps them adapt their own drafts
toward that baseline. The project must read as a serious NLP portfolio centerpiece that
stands out to senior engineers at top tech companies, paired with a teaching guide and a
live demo. Everything authored solely as Ab-Romia, with no AI attribution anywhere.

## Positioning (the load-bearing decision)

Drop every reference to AI detectors, "humanizing," "bypass," or "evade." The project is
about authorship analysis and personal-voice adaptation. Rewrites optimize *toward the
author's own voice centroid* (a positive, explainable target), never *away from* a
detector. A consent / own-text note appears in the demo UI: you analyze and adapt your
own writing.

Framing guard: before shipping, grep the entire repo for `bypass`, `detect`, `humaniz`,
`GPTZero`, `Originality` and confirm zero user-facing matches.

## Locked decisions

- **Name:** VoicePrint everywhere. Recommend renaming the GitHub repo
  `Style-Echo-AI-Humanizer` to `VoicePrint` (owner executes the rename + push; the
  portfolio links are set to the final slug only after the live URLs are confirmed).
- **Demo:** Gradio HF Space as the canonical always-live demo. A polished Next.js
  frontend matching romia.dev follows in the next loop.
- **Guide:** in-repo `docs/guide.md` deep build guide, plus a real blog post on romia.dev.
- **Blog:** build a real `/blog` + `/blog/[slug]` route and a `BLOG_POSTS` array in
  `resume.ts` (first post is the VoicePrint writeup). Fix the false claim in CLAUDE.md
  that `BLOG_POSTS` already exists. (Next loop.)
- **Rewrite LLM:** bring-your-own-key, targeting OpenAI and OpenRouter (both
  OpenAI-compatible, so one client with a switchable `base_url`). No key is ever
  committed. Keyless users get the rule-based fallback.
- **Sequencing:** Pass 1 ships the working honest engine, the live Gradio Space, the
  in-repo docs/guide, the README, and the resume.ts project update. The Next.js frontend
  and the `/blog` route ship in the next loop.

## Engine design

### Data flow

```
samples (>= 3, ~500+ words total)
  -> stylometric profile    interpretable feature extraction (4 families)
  -> style fingerprint      StyleDistance neural centroid (cached) + normalized radar vector
  -> conditioned transform  profile rendered as NL constraints + few-shot exemplars
                            -> LLM rewrite (BYO-key)  |  rule-based fallback (no key)
  -> validation             re-embed output, cosine to author centroid, per-axis radar
                            delta, semantic preservation vs the source draft
  -> report                 voice-match %, before/after delta, sentence-level diff
```

### KEEP and elevate (credible analytical backbone)

- `linguistic_analyzer.py`: spaCy + textstat sentence stats, dependency-tree depth, TTR,
  POS distribution, normalized punctuation, readability indices. Fix: input guards
  (empty / over-long sentences), guard the single-root dependency assumption, cap
  recursion depth, and emit per-feature variance across samples (consistency), not just
  the mean.
- `stylometric_analyzer.py`: function-word frequencies, n-gram patterns, contraction
  rate, passive/active voice, sentence-starter POS, transition rate. Fix: expose function
  words as a fixed-length, content-independent vector (fixed stoplist) instead of a sparse
  dict, so distance scoring and the validator align principled.
- `embedding_analyzer.py`: centroid + cosine + consistency. Fix: use spaCy `doc.sents`
  instead of splitting on `.`; cache and persist the centroid.
- `validator.py`: weighted quality score. Fix: compare semantic preservation against the
  user's source draft (not against any AI text); document and calibrate the magic scaling
  constants and threshold instead of leaving them arbitrary.

### FIX (make it run, the bar everything else depends on)

- Restore `backend/app/models/style_profile.py` (201 lines) from
  `git show 144e95c:backend/app/models/style_profile.py`, plus an empty
  `backend/app/models/__init__.py`. This is the contract every call site expects
  (`StyleProfile`, `StyleProfileStore`, the module singleton `profile_store`). Confirmed
  recoverable.
- `.gitignore` line 33: change bare `models/` to anchored `/models/` and add `/data/`
  (weight/data dirs only). Verify `git check-ignore backend/app/models/style_profile.py`
  returns nothing afterward. Confirmed: that path is currently ignored.

### CUT (non-negotiable)

- Delete `ai_detector_remover.py` entirely: the injected typos (`teh`, `adn`, `thier`,
  `recieve`), filler-word stuffing, the fake `_calculate_ai_probability` score, and the
  whole `remove_ai_patterns` flow.
- In `style_transfer.py`: delete `_simple_passive_to_active` (fabricates broken subjects)
  and the random em-dash insertion. Keep only contraction normalization and
  punctuation-rate matching, folded into the rule-based fallback.
- Remove the duplicate in-memory `user_profiles` dict in `app.py`; rely on `profile_store`.
- Lock CORS `allow_origins` to known frontend origins, not `*`.
- Strip dead Postgres/Redis/Chroma settings from `config.py` and drop those dependencies.

### ADD (the modern engine)

- `style_embedding.py`: load **StyleDistance/styledistance** (RoBERTa-base, ~125M,
  content-independent, 2024 SOTA on PAN authorship verification) via sentence-transformers.
  Build and cache the author centroid; headline "voice match %" is
  cosine(candidate, centroid). Optionally lazy-load a second embedder (LUAR) as an
  ensemble confidence signal; only StyleDistance is resident by default to respect free
  Space RAM.
- `char_ngram.py`: scikit-learn `TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4))`
  for character n-grams (the strongest cross-domain authorship family). Feeds the
  fingerprint and an optional logistic-regression author-verifier baseline.
- `rewrite/` package:
  - `profile_to_constraints.py`: render the numeric profile into natural-language
    constraints (for example "average sentence length about 14 words; semicolons used
    sparingly; prefers contractions"). The renderer must never instruct em-dash usage.
  - `llm_rewriter.py`: BYO-key in-context style transfer. Prompt = source draft + 3 to 5
    of the author's real exemplars + rendered constraints, one LLM call. OpenAI-compatible
    client with switchable `base_url` (OpenAI / OpenRouter). No key shipped, no GPU.
  - `rule_rewriter.py`: deterministic fallback (contraction + punctuation-rate matching).
- `feedback.py`: before/after radar delta and per-sentence diff, each sentence colored by
  whether its per-sentence style-match to the centroid improved or regressed. Honest
  framing: voice-guided adaptation, not perfect impersonation.
- `store/sqlite_store.py`: thin SQLite-backed store (profiles + serialized centroid)
  replacing the in-memory singleton. Lightweight; no heavy DB stack.

## Repo structure after rebuild

```
VoicePrint/
  app.py                         # Gradio entry (the HF Space app)
  pyproject.toml                 # single source of truth for deps
  requirements.txt               # pinned; HF reads this directly
  Dockerfile                     # CPU image; preloads spaCy model + StyleDistance
  README.md                      # HF YAML card front-matter + demo-forward readme
  DEPLOY.md                      # single deploy doc (HF Spaces + Docker)
  docs/guide.md                  # the in-repo build guide
  LICENSE
  .gitignore                     # /models/ /data/ anchored, *.pt/*.pth
  backend/app/
    main.py                      # FastAPI; CORS locked
    config.py                    # only real settings
    models/style_profile.py      # restored
    models/__init__.py
    api/routes.py                # ai_detection_score field removed
    schemas/profile.py
    services/
      linguistic_analyzer.py     # keep + fix
      stylometric_analyzer.py    # keep + fix
      embedding_analyzer.py      # keep + fix
      style_embedding.py         # add (StyleDistance)
      char_ngram.py              # add
      validator.py               # keep + fix
      voiceprint_service.py      # fix imports; new flow
      feedback.py                # add
      rewrite/{profile_to_constraints,llm_rewriter,rule_rewriter}.py
    store/sqlite_store.py        # add
  tests/{test_api,test_analyzers}.py
  frontend/                      # Next.js 16 demo (next loop)
```

Deleted: `ai_detector_remover.py`, `README_HF.md`, `HF_DEPLOY_QUICK.md`, `QUICKSTART.md`,
`TESTING_REPORT.md` (contains "Tester: Claude" and a false "PRODUCTION READY" claim),
`prepare_hf_space.sh`, `requirements-hf.txt` (consolidated), `backend/requirements.txt`
(consolidated).

## Dependencies (2026, CPU-only)

Modern pinned set: gradio (6.x), torch (CPU wheel via the PyTorch CPU index),
transformers, sentence-transformers (5.x), spaCy 3.8 + the pinned `en_core_web_sm` 3.8
wheel, textstat, scikit-learn, nltk, pydantic, fastapi, uvicorn, pytest. numpy 2 is the
one risky bump: gate it behind an import/CI smoke test and fall back to 1.26 if torch,
spaCy, or scikit-learn conflict. Do not include psycopg2, sqlalchemy, chromadb, redis, or
pandas (all dead).

## Gradio HF Space

SDK Gradio 6.x. HF Spaces config lives in the README YAML front-matter (`title`, `emoji`,
`sdk: gradio`, `sdk_version`, `app_file: app.py`, `short_description`, and
`preload_from_hub: StyleDistance/styledistance`). Hardware cpu-basic (free).

App layout (`gr.Blocks`, restrained, no emoji walls):
- Tab 1, Build your profile: multi-sample input (>= 3), a one-line consent note, Build
  button. Output: radar chart of normalized stylometric axes + consistency number, held
  in `gr.State`.
- Tab 2, Adapt a draft: paste a draft, optional BYO API-key field (password type, never
  logged), Adapt button. Output: rewritten text, a voice-match gauge (before/after),
  per-axis radar delta, and a sentence-level diff (`gr.HighlightedText`). Keyless falls
  back to the rule rewriter and says so plainly.
- Footer: "Built by Ab-Romia" + GitHub link. No detector copy anywhere.

Embedded on the portfolio via the existing `demo-embed.tsx` iframe at the real Space
subdomain (lazy-loaded skeleton).

## Docs and portfolio

### In-repo guide (`docs/guide.md`)

Demo screenshot first; why I built this (first person); intuition for a writing
fingerprint; dataset with the topic-leakage caveat called out; features one family at a
time, each ending in working code under ~100 lines plus a separation plot (lexical, char
n-grams, syntactic, readability); an interpretable baseline (logistic regression / SVM)
with honest held-out metrics (macro-F1, accuracy, per-author confusion) against a named
baseline and stated dataset size; error analysis with one honest failure; the neural
fingerprint (StyleDistance, why content-independence matters); style transfer as
content-preserving rewriting, contrasted explicitly against the 2021 regex approach this
repo started from; evaluation across style/content/fluency; limits and dual-use ethics.
No "state of the art," no unbacked superlatives.

### README

Demo-forward and problem-first: one-line what-and-for-whom; a 3-line try-it block + live
Space link; one headline measured result; why I built this; how-it-works with one diagram
(SVG/ASCII, no React Flow); a results table (metric, baseline, dataset size); honest
limitations; what I would do differently; reproduce steps; footer. No emoji headers, no
"PRODUCTION READY," no roadmap fluff.

### romia.dev edits

- `resume.ts` PROJECTS VoicePrint entry: retitle to a stylometry framing, retag
  (`NLP`, `Stylometry`, `StyleDistance`, `Python`), rewrite description and the
  `caseStudy` fields away from any "humanizer" wording, add a decision card naming
  StyleDistance and the BYO-key in-context rewrite, and set `github` / `demo` /
  `embedDemo.src` to the confirmed live URLs only.
- Next loop: add `BLOG_POSTS` to `resume.ts` and `/blog` + `/blog/[slug]` routes mirroring
  the project case-study pages; add the VoicePrint writeup as the first post; update
  `sitemap.ts`; fix the CLAUDE.md claim.

## Voice and no-AI-traces rules

First person as Ab-Romia. No em-dashes anywhere. No AI/Claude/LLM attribution in commits,
code comments, docs, or PRs. Commits authored solely as `Ab-Romia <aabouroumia@gmail.com>`.
Kill-list on all prose: no "delve," "leverage," "utilize," "robust," "seamless,"
"comprehensive," "it is important to note," no not-X-but-Y, no reflexive tricolons, no
emoji bullets, minimal bolding. Vary sentence length, active voice, exact numbers.

## Order of operations

0. Make it run: fix `.gitignore`, restore the models package, confirm the app boots and
   `pytest` is green.
1. Make it honest: delete `ai_detector_remover.py`, the broken passive-to-active and
   em-dash insertion; strip all detector copy; lock CORS; remove the duplicate dict.
2. Modernize deps: consolidate to `pyproject.toml` + a single `requirements.txt`, bump to
   2026 pins, gate numpy 2 behind a compat check.
3. Add the engine: `style_embedding.py`, `char_ngram.py`, the `rewrite/` package,
   `feedback.py`, `sqlite_store.py`.
4. Demo: rebuild `app.py` as the Gradio Space + HF YAML card; confirm the live subdomain.
5. Docs + portfolio: README, DEPLOY.md, `docs/guide.md`, resume.ts project edits.
6. Next loop: Next.js frontend, `/blog` route + first post, CLAUDE.md fix.

## Risks and mitigations

- HF free CPU RAM / cold start: preload only StyleDistance; lazy-load extras; CPU torch
  wheel.
- numpy 2 breaking bump: gate behind a smoke test; fall back to 1.26.
- Broken-import cascade: restoring the models package is the linchpin; confirm boot before
  any other phase.
- Live-URL mismatch: confirm the deployed Space subdomain and the GitHub repo name before
  touching resume.ts; a dead link is worse than no link.
- Scope creep: SQLite only; Gradio Space is canonical; frontend and blog are the next
  loop.
- AI-slop / framing slip: kill-list pass on all copy; grep for `bypass`/`detect`/`humaniz`
  before shipping.
