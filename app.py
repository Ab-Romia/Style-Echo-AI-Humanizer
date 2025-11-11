"""
VoicePrint Gradio Interface for Hugging Face Spaces

This is the main interface for deploying VoicePrint on Hugging Face Spaces.
"""
import gradio as gr
import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

# Import with error handling
try:
    from app.services.voiceprint_service import VoicePrintService
    from app.models.style_profile import profile_store
    SERVICE_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not load full service: {e}")
    SERVICE_AVAILABLE = False

# Initialize service if available
voiceprint_service = None
if SERVICE_AVAILABLE:
    try:
        voiceprint_service = VoicePrintService()
        print("✓ VoicePrint service initialized successfully")
    except Exception as e:
        print(f"Warning: Service initialization failed: {e}")
        SERVICE_AVAILABLE = False

# Store for user profiles (in-memory for demo)
user_profiles = {}


def create_profile(user_id: str, sample1: str, sample2: str, sample3: str, profile_name: str = None):
    """Create a new style profile from writing samples."""
    if not SERVICE_AVAILABLE:
        return "⚠️ Service not available. Please check model installation.", None

    if not user_id:
        return "❌ Please provide a user ID", None

    samples = [s.strip() for s in [sample1, sample2, sample3] if s.strip()]

    if len(samples) < 3:
        return "❌ Please provide at least 3 writing samples", None

    total_words = sum(len(s.split()) for s in samples)
    if total_words < 500:
        return f"❌ Need at least 500 words total. You provided {total_words} words.", None

    try:
        profile = voiceprint_service.create_style_profile(
            user_id=user_id,
            samples=samples,
            profile_name=profile_name or f"Profile for {user_id}"
        )

        # Store profile ID for this user
        user_profiles[user_id] = profile.profile_id

        result = f"""
✅ **Profile Created Successfully!**

**Profile ID:** {profile.profile_id}
**User ID:** {user_id}
**Total Words Analyzed:** {profile.linguistic_features.get('total_word_count', 0)}
**Number of Samples:** {len(samples)}

**Your Writing Style:**
- Average sentence length: {profile.linguistic_features.get('avg_sentence_length', 0):.1f} words
- Lexical diversity: {profile.linguistic_features.get('type_token_ratio', 0):.2f}
- Contraction rate: {profile.stylometric_features.get('contraction_rate', 0):.2%}
- Passive voice usage: {profile.stylometric_features.get('passive_voice_ratio', 0):.2%}

You can now use this profile to humanize AI-generated text!
"""
        return result, profile.profile_id

    except Exception as e:
        return f"❌ Error creating profile: {str(e)}", None


def humanize_text(user_id: str, profile_id: str, ai_text: str, strength: float):
    """Humanize AI-generated text using a style profile."""
    if not SERVICE_AVAILABLE:
        return "⚠️ Service not available. Please check model installation."

    if not profile_id:
        # Try to get profile from user_id
        profile_id = user_profiles.get(user_id)
        if not profile_id:
            return "❌ Please create a profile first or provide a profile ID"

    if not ai_text or len(ai_text.strip()) < 10:
        return "❌ Please provide text to humanize (at least 10 characters)"

    try:
        result = voiceprint_service.humanize_text(
            profile_id=profile_id,
            ai_text=ai_text,
            strength=strength,
            preserve_meaning=True
        )

        output = f"""
✅ **Text Humanized Successfully!**

**Original AI Detection Score:** {result['ai_removal_metrics']['original_ai_score']:.2%}
**After Humanization:** {result['ai_removal_metrics']['improved_ai_score']:.2%}
**Improvement:** {result['ai_removal_metrics']['improvement']:.2%}

**Style Similarity Score:** {result['validation']['style_similarity']:.2%}
**Semantic Preservation:** {result['validation']['semantic_preservation']:.2%}
**Overall Quality:** {result['validation']['overall_quality_score']:.2%}

---

**Your Humanized Text:**

{result['humanized_text']}

---

**Suggestions:**
{chr(10).join('• ' + s for s in result['suggestions'])}
"""
        return output

    except ValueError as e:
        return f"❌ {str(e)}"
    except Exception as e:
        return f"❌ Error humanizing text: {str(e)}"


def analyze_text(text: str):
    """Quick analysis of any text."""
    if not SERVICE_AVAILABLE:
        return "⚠️ Service not available. Please check model installation."

    if not text or len(text.strip()) < 50:
        return "❌ Please provide at least 50 characters of text to analyze"

    try:
        analysis = voiceprint_service.quick_analysis(text)

        output = f"""
📊 **Text Analysis**

**Basic Stats:**
- Word count: {analysis['word_count']}
- Average sentence length: {analysis['linguistic_features'].get('avg_sentence_length', 0):.1f} words
- Lexical diversity: {analysis['linguistic_features'].get('type_token_ratio', 0):.2f}

**Readability:**
- Flesch Reading Ease: {analysis['linguistic_features'].get('flesch_reading_ease', 0):.1f}
- Flesch-Kincaid Grade: {analysis['linguistic_features'].get('flesch_kincaid_grade', 0):.1f}

**AI Detection:**
- AI Probability: {analysis['ai_detection'].get('ai_probability', 0):.2%}
- Structure Diversity: {analysis['ai_detection'].get('structure_diversity', 0):.2f}
- Hedging Phrases: {analysis['ai_detection'].get('hedging_phrase_count', 0)}

**Writing Style:**
- Contraction rate: {analysis['stylometric_features'].get('contraction_rate', 0):.2%}
- Passive voice: {analysis['stylometric_features'].get('passive_voice_ratio', 0):.2%}
- Function word ratio: {analysis['stylometric_features'].get('function_word_ratio', 0):.2%}
"""
        return output

    except Exception as e:
        return f"❌ Error analyzing text: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="VoicePrint: AI Text Humanizer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎯 VoicePrint: AI Text Humanizer with Style Learning

    Transform AI-generated text to match your unique writing voice. VoicePrint learns from your writing samples
    and applies your personal style while bypassing AI detection.

    ## How it works:
    1. **Create Profile**: Provide 3+ writing samples (500+ words total)
    2. **Humanize Text**: Paste AI-generated text and let VoicePrint transform it
    3. **Analyze**: Check any text for AI patterns and writing style metrics
    """)

    with gr.Tabs():
        # Tab 1: Create Profile
        with gr.Tab("📝 Create Style Profile"):
            gr.Markdown("""
            ### Step 1: Build Your Writing Profile
            Provide 3-10 samples of your writing (emails, essays, articles, etc.).
            Minimum 500 words total across all samples.
            """)

            with gr.Row():
                with gr.Column():
                    user_id_input = gr.Textbox(
                        label="User ID",
                        placeholder="your_username",
                        info="Unique identifier for your profile"
                    )
                    profile_name_input = gr.Textbox(
                        label="Profile Name (Optional)",
                        placeholder="My Casual Writing Style"
                    )

            sample1 = gr.Textbox(
                label="Writing Sample 1",
                placeholder="Paste your first writing sample here...",
                lines=5
            )
            sample2 = gr.Textbox(
                label="Writing Sample 2",
                placeholder="Paste your second writing sample here...",
                lines=5
            )
            sample3 = gr.Textbox(
                label="Writing Sample 3",
                placeholder="Paste your third writing sample here...",
                lines=5
            )

            create_btn = gr.Button("🚀 Create Profile", variant="primary")
            profile_output = gr.Textbox(label="Profile Creation Result", lines=10)
            profile_id_output = gr.Textbox(label="Profile ID (save this!)", visible=True)

            create_btn.click(
                fn=create_profile,
                inputs=[user_id_input, sample1, sample2, sample3, profile_name_input],
                outputs=[profile_output, profile_id_output]
            )

        # Tab 2: Humanize Text
        with gr.Tab("✨ Humanize AI Text"):
            gr.Markdown("""
            ### Step 2: Transform AI Text to Your Style
            Paste AI-generated text below and watch it transform into your writing voice.
            """)

            with gr.Row():
                with gr.Column():
                    humanize_user_id = gr.Textbox(
                        label="User ID",
                        placeholder="your_username"
                    )
                    humanize_profile_id = gr.Textbox(
                        label="Profile ID (optional if you just created one)",
                        placeholder="Leave blank if you just created a profile"
                    )

            ai_text_input = gr.Textbox(
                label="AI-Generated Text",
                placeholder="Paste the AI-generated text you want to humanize...",
                lines=8
            )

            strength_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.7,
                step=0.1,
                label="Transformation Strength",
                info="0.0 = minimal changes, 1.0 = maximum transformation"
            )

            humanize_btn = gr.Button("🎨 Humanize Text", variant="primary")
            humanize_output = gr.Textbox(label="Result", lines=15)

            humanize_btn.click(
                fn=humanize_text,
                inputs=[humanize_user_id, humanize_profile_id, ai_text_input, strength_slider],
                outputs=humanize_output
            )

        # Tab 3: Analyze Text
        with gr.Tab("🔍 Analyze Text"):
            gr.Markdown("""
            ### Quick Text Analysis
            Analyze any text to see linguistic features, AI detection probability, and writing style metrics.
            """)

            analyze_input = gr.Textbox(
                label="Text to Analyze",
                placeholder="Paste any text here to analyze...",
                lines=8
            )

            analyze_btn = gr.Button("📊 Analyze", variant="primary")
            analyze_output = gr.Textbox(label="Analysis Results", lines=12)

            analyze_btn.click(
                fn=analyze_text,
                inputs=analyze_input,
                outputs=analyze_output
            )

    gr.Markdown("""
    ---
    ### 📚 About VoicePrint

    VoicePrint uses advanced NLP techniques including:
    - Linguistic feature extraction (sentence patterns, vocabulary, punctuation)
    - Stylometric analysis (function words, n-grams, voice patterns)
    - Sentence-BERT embeddings for semantic style matching
    - AI detection removal (removes hedging phrases, perfect grammar patterns)
    - Style transfer engine (applies your specific writing patterns)

    **Built by:** Ab-Romia
    **GitHub:** [Style-Echo-AI-Humanizer](https://github.com/Ab-Romia/Style-Echo-AI-Humanizer)
    **License:** MIT
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
