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
import matplotlib

matplotlib.use("Agg")  # headless backend for a server
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Make the backend package importable.
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from voiceprint.services.voiceprint_service import VoicePrintService  # noqa: E402

service = VoicePrintService()

# Human-readable labels for the six normalized fingerprint axes.
AXIS_LABELS = {
    "avg_sentence_length": "Sentence length",
    "type_token_ratio": "Lexical diversity",
    "contraction_rate": "Contractions",
    "passive_voice_ratio": "Passive voice",
    "function_word_ratio": "Function words",
    "transition_word_rate": "Transitions",
}


def make_radar(series, title):
    """
    Draw one or more normalized fingerprints as overlaid radar polygons.

    Args:
        series: list of (label, axes_dict, color) where axes_dict maps each
            axis name to a value in [0, 1].
        title: chart title.

    Returns:
        A matplotlib Figure.
    """
    axis_keys = list(AXIS_LABELS.keys())
    labels = [AXIS_LABELS[k] for k in axis_keys]
    angles = np.linspace(0, 2 * np.pi, len(axis_keys), endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True})
    for label, axes_dict, color in series:
        values = [float(axes_dict.get(k, 0.0)) for k in axis_keys]
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=label)
        ax.fill(angles, values, color=color, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", ""])
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=12, pad=18)
    if len(series) > 1:
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=8)
    fig.tight_layout()
    return fig


def build_profile(sample1, sample2, sample3, sample4, sample5, profile_name):
    """Build a voice profile from the pasted samples."""
    samples = [s.strip() for s in (sample1, sample2, sample3, sample4, sample5) if s and s.strip()]

    if len(samples) < 3:
        return "Please provide at least 3 writing samples.", None, None

    total_words = sum(len(s.split()) for s in samples)
    if total_words < 500:
        return (
            f"Need at least 500 words total. You provided {total_words}.",
            None,
            None,
        )

    try:
        profile = service.build_profile(
            user_id="demo-user",
            samples=samples,
            profile_name=profile_name or "My voice",
        )
    except ValueError as exc:
        return str(exc), None, None
    except Exception as exc:
        return f"Could not build the profile: {exc}", None, None

    from voiceprint.services.feedback import normalize_axes

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
        + "\n".join(
            f"  {AXIS_LABELS.get(name, name)}: {value:.2f}"
            for name, value in axes.items()
        )
    )
    radar = make_radar(
        [("Your voice", axes, "#10B981")], "Your voice fingerprint"
    )
    return summary, profile.profile_id, radar


def adapt_draft(profile_id, source_draft, api_key, base_url, model):
    """Adapt a draft toward the built voice."""
    if not profile_id:
        return "Build your profile first.", "", [], None

    if not source_draft or len(source_draft.strip()) < 10:
        return "Please paste a draft of at least 10 characters.", "", [], None

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
        return str(exc), "", [], None
    except Exception as exc:
        return f"Could not adapt the draft: {exc}", "", [], None

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

    radar_data = result["feedback"]["radar"]
    radar = make_radar(
        [
            ("Your voice", radar_data["author"], "#6366F1"),
            ("Before", radar_data["before"], "#9CA3AF"),
            ("After", radar_data["after"], "#10B981"),
        ],
        "Draft vs your voice",
    )

    return gauge, result["adapted_text"], highlights, radar


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
            with gr.Row():
                build_output = gr.Textbox(label="Fingerprint summary", lines=14)
                radar_output = gr.Plot(label="Voice fingerprint")

            build_btn.click(
                fn=build_profile,
                inputs=[s1, s2, s3, s4, s5, name_box],
                outputs=[build_output, profile_id_state, radar_output],
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
            with gr.Row():
                gauge_output = gr.Textbox(label="Voice match", lines=5)
                adapt_radar = gr.Plot(label="Draft vs your voice")
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
                outputs=[gauge_output, adapted_output, diff_output, adapt_radar],
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
