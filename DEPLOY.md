# Deploying VoicePrint

Three ways to run it: the Hugging Face Space (the canonical live demo), local from source, and the FastAPI backend or Docker image.

## Hugging Face Spaces

The Space is configured entirely by the YAML card at the top of `README.md`: `sdk: gradio`, `app_file: app.py`, the `sdk_version`, and `preload_from_hub: StyleDistance/styledistance`, which downloads the style model when the Space builds so the first request is not slow. Hardware is cpu-basic (free); the app is CPU-only by design and the requirements pin the CPU PyTorch wheel.

To deploy, push the repo to the Space git remote:

```bash
git remote add space https://huggingface.co/spaces/<user>/<space-name>
git push space main
```

The Space installs from `requirements.txt`, reads the YAML card, and launches `app.py`. The rewrite key is optional and is never committed. If you want a default key available to the Space, set it as a Space secret named `OPENAI_API_KEY` in the Space settings; the rewriter reads it from the environment, and without it the app falls back to the rule rewriter. Users can also paste their own key in the UI for a single session.

## Local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # only if the pinned wheel did not install it
python app.py
```

Open the local URL Gradio prints (default http://localhost:7860). To use the LLM rewrite locally, either paste a key in the optional field in the second tab, or export `OPENAI_API_KEY` (and `OPENAI_BASE_URL` for OpenRouter) before launching. With no key, the rule rewriter runs.

## FastAPI backend

The Gradio app is the demo; the same engine is also exposed as a REST API. From the `backend` directory:

```bash
cd backend
uvicorn voiceprint.main:app --host 0.0.0.0 --port 8000
```

## Docker

Build and run the image:

```bash
docker build -t voiceprint .
docker run -p 7860:7860 voiceprint
```

The image is CPU-based, installs the Python dependencies, downloads the needed NLTK data at build time, copies the backend and `app.py`, and starts the Gradio app on port 7860. To enable the LLM rewrite in the container, pass the key at run time with `-e OPENAI_API_KEY=...`; do not bake it into the image.
