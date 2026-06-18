# VoicePrint frontend

A Next.js interface for VoicePrint is planned here. The Gradio app at the repo
root is the canonical demo for now; this will be a polished web client that
talks to the FastAPI backend.

## Planned

- Paste writing samples and build a voice profile.
- The voice fingerprint as a radar chart.
- Adapt a draft and see the voice match before and after, with a per-sentence diff.

## Stack

- Next.js 16 (App Router) and TypeScript
- Tailwind CSS
- A lightweight chart library for the fingerprint

Until this ships, run the Gradio demo from the repo root with `python app.py`.
