"""
VoicePrint Gradio app (Hugging Face Space entry).

Two tabs:
  Build your profile: paste at least three samples of your own writing and
    build a voice fingerprint.
  Adapt a draft: paste a draft and adapt it toward your measured voice, with a
    voice-match gauge before and after and a sentence-level diff.

You analyze and adapt your own writing.
"""
import sys
from pathlib import Path

import gradio as gr

# Make the backend package importable.
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from app.services.voiceprint_service import VoicePrintService  # noqa: E402

service = VoicePrintService()


def build_profile(sample1, sample2, sample3, sample4, sample5, profile_name):
    """Build a voice profile from the pasted samples."""
    samples = [s.strip() for s in (sample1, sample2, sample3, sample4, sample5) if s and s.strip()]

    if len(samples) < 3:
        return "Please provide at least 3 writing samples.", None, {}

    total_words = sum(len(s.split()) for s in samples)
    if total_words < 500:
        return (
            f"Need at least 500 words total. You provided {total_words}.",
            None,
            {},
        )

    try:
        profile = service.build_profile(
            user_id="demo-user",
            samples=samples,
            profile_name=profile_name or "My voice",
        )
    except ValueError as exc:
        return str(exc), None, {}
    except Exception as exc:
        return f"Could not build the profile: {exc}", None, {}

    from app.services.feedback import normalize_axes

    axes = normalize_axes(
        profile.linguistic_features, profile.stylometric_features
    )
    consistency = profile.embedding_metadata.get("consistency_metrics", {})
    mean_sim = consistency.get("mean_pairwise_similarity", 0.0)

    summary = (
        f"Profile built. {len(samples)} samples, {total_words} words.\n\n"
        f"Average sentence length: "
        f"{profile.linguistic_features.get('avg_sentence_length', 0):.1f} words\n"
        f"Lexical diversity: "
        f"{profile.linguistic_features.get('type_token_ratio', 0):.2f}\n"
        f"Contraction rate: "
        f"{profile.stylometric_features.get('contraction_rate', 0):.1%}\n"
        f"Sample consistency: {mean_sim:.2f}\n\n"
        "Fingerprint axes (0 to 1):\n"
        + "\n".join(f"  {name}: {value:.2f}" for name, value in axes.items())
    )
    return summary, profile.profile_id, axes


def adapt_draft(profile_id, source_draft, api_key, base_url, model):
    """Adapt a draft toward the built voice."""
    if not profile_id:
        return "Build your profile first.", "", []

    if not source_draft or len(source_draft.strip()) < 10:
        return "Please paste a draft of at least 10 characters.", "", []

    try:
        result = service.adapt_draft(
            profile_id=profile_id,
            source_draft=source_draft,
            api_key=api_key or None,
            base_url=base_url or None,
            model=model or None,
            use_llm=True,
        )
    except ValueError as exc:
        return str(exc), "", []
    except Exception as exc:
        return f"Could not adapt the draft: {exc}", "", []

    path = result["rewrite_path"]
    path_note = (
        "Rewritten with the LLM rewriter."
        if path == "llm"
        else "No API key provided, so this used the rule-based rewriter."
    )

    gauge = (
        f"{path_note}\n\n"
        f"Voice match before: {result['voice_match_before']:.1%}\n"
        f"Voice match after:  {result['voice_match_after']:.1%}\n"
        f"Meaning preserved (vs your draft): "
        f"{result['validation']['semantic_preservation']:.1%}"
    )

    # Map the per-sentence diff to gr.HighlightedText tuples.
    highlights = [
        (item["sentence"] + " ", item["label"])
        for item in result["feedback"]["sentence_diff"]
    ]
    if not highlights:
        highlights = [(result["adapted_text"], "same")]

    return gauge, result["adapted_text"], highlights


with gr.Blocks(title="VoicePrint") as demo:
    gr.Markdown(
        "# VoicePrint\n"
        "Measure your writing voice from your own samples, then adapt your own "
        "drafts toward that baseline."
    )

    profile_id_state = gr.State(value=None)

    with gr.Tabs():
        with gr.Tab("Build your profile"):
            gr.Markdown(
                "Paste at least three samples of your own writing "
                "(500+ words total). You analyze your own writing here."
            )
            name_box = gr.Textbox(label="Profile name (optional)", lines=1)
            s1 = gr.Textbox(label="Sample 1", lines=4)
            s2 = gr.Textbox(label="Sample 2", lines=4)
            s3 = gr.Textbox(label="Sample 3", lines=4)
            s4 = gr.Textbox(label="Sample 4 (optional)", lines=4)
            s5 = gr.Textbox(label="Sample 5 (optional)", lines=4)
            build_btn = gr.Button("Build profile", variant="primary")
            build_output = gr.Textbox(label="Fingerprint summary", lines=14)
            axes_output = gr.JSON(label="Radar axes")

            build_btn.click(
                fn=build_profile,
                inputs=[s1, s2, s3, s4, s5, name_box],
                outputs=[build_output, profile_id_state, axes_output],
            )

        with gr.Tab("Adapt a draft"):
            gr.Markdown(
                "Paste one of your own drafts to adapt it toward your voice. An "
                "API key is optional: without one, a rule-based rewriter runs."
            )
            draft_box = gr.Textbox(label="Your draft", lines=8)
            with gr.Accordion("Optional: bring your own LLM key", open=False):
                key_box = gr.Textbox(label="API key", type="password", lines=1)
                base_url_box = gr.Textbox(
                    label="Base URL (OpenAI default, or OpenRouter)", lines=1
                )
                model_box = gr.Textbox(label="Model name", lines=1)
            adapt_btn = gr.Button("Adapt", variant="primary")
            gauge_output = gr.Textbox(label="Voice match", lines=5)
            adapted_output = gr.Textbox(label="Adapted draft", lines=8)
            diff_output = gr.HighlightedText(
                label="Sentence diff",
                color_map={
                    "improved": "green",
                    "regressed": "red",
                    "same": "gray",
                },
            )

            adapt_btn.click(
                fn=adapt_draft,
                inputs=[profile_id_state, draft_box, key_box, base_url_box, model_box],
                outputs=[gauge_output, adapted_output, diff_output],
            )

    gr.Markdown(
        "Built by Ab-Romia. "
        "[GitHub](https://github.com/Ab-Romia/VoicePrint)"
    )


if __name__ == "__main__":
    import os

    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        theme=gr.themes.Soft(),
    )
