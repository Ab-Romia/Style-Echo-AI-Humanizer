---
title: VoicePrint AI Text Humanizer
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.16.0
app_file: app.py
pinned: false
license: mit
---

# VoicePrint: AI Text Humanizer with Style Learning

Transform AI-generated text to match your unique writing voice. VoicePrint learns from your writing samples and applies your personal style while bypassing AI detection.

## 🚀 Features

- **Style Learning**: Analyzes 3-10 writing samples to build a complete profile of your writing voice
- **AI Detection Removal**: Identifies and removes common AI tells (hedging phrases, perfect grammar, repetitive structures)
- **Style Transfer**: Applies your specific writing patterns to transform text
- **Real-time Analysis**: Check any text for AI patterns and writing style metrics
- **Validation**: Ensures output matches your style (85%+ similarity) while preserving meaning

## 📖 How to Use

### 1. Create Your Style Profile

Navigate to the "Create Style Profile" tab and:
1. Enter a unique User ID
2. Paste 3 or more writing samples (minimum 500 words total)
3. Click "Create Profile"
4. Save your Profile ID for later use

**Tips for best results:**
- Use authentic samples of your writing (emails, blog posts, essays)
- Include diverse examples showing different contexts
- Aim for 200-300 words per sample

### 2. Humanize AI-Generated Text

Go to the "Humanize AI Text" tab and:
1. Enter your User ID
2. Paste the AI-generated text you want to transform
3. Adjust the transformation strength (0.7 is recommended)
4. Click "Humanize Text"

The system will:
- Remove AI detection patterns
- Apply your writing style
- Show before/after AI detection scores
- Provide quality metrics

### 3. Analyze Any Text

Use the "Analyze Text" tab to:
- Check AI detection probability
- View linguistic features
- See readability scores
- Analyze writing style patterns

## 🔬 How It Works

VoicePrint uses advanced NLP techniques:

**Linguistic Analysis:**
- Sentence length patterns with standard deviation
- Syntax complexity (dependency tree depth)
- Lexical diversity (type-token ratio)
- Part-of-speech distribution
- Punctuation habits
- Readability scores

**Stylometric Analysis:**
- Function word frequencies
- N-gram patterns
- Contraction usage
- Passive vs active voice ratio
- Sentence starter patterns

**AI Detection Removal:**
- Identifies perfect grammar patterns
- Removes hedging phrases
- Breaks up repetitive structures
- Adds natural variations

**Style Transfer:**
- Matches sentence length patterns
- Adjusts vocabulary formality
- Replicates punctuation style
- Balances passive/active voice

**Validation:**
- Sentence-BERT embeddings for style matching
- Cosine similarity with style centroid
- Semantic preservation checks
- Quality score calculation

## 🛠️ Technology Stack

- **Gradio**: Interactive web interface
- **spaCy**: Linguistic parsing and analysis
- **Sentence-BERT**: Semantic embeddings
- **NLTK**: Stylometric features
- **TextStat**: Readability metrics
- **PyTorch & Transformers**: Deep learning models

## 📚 Research Foundation

This project builds on established research:

- **Style Transfer**: Jin et al. (2021) - "Deep Learning for Text Style Transfer: A Survey"
- **Stylometry**: Juola (2006) - "Stylometric Analysis of Literary Texts"
- **AI Detection**: Mitchell et al. (2023) - "DetectGPT: Zero-Shot Machine-Generated Text Detection"
- **Embeddings**: Reimers & Gurevych (2019) - "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"

## 🎯 Use Cases

- **Academic Writing**: Make AI-assisted drafts sound like your own work
- **Content Creation**: Maintain consistent voice across AI-generated content
- **Email Communication**: Transform formal AI suggestions to your casual style
- **Social Media**: Convert generic AI posts to your unique voice
- **Professional Writing**: Ensure AI tools match your professional tone

## ⚠️ Limitations

- Requires 500+ words of writing samples for accurate profiling
- Works best with English text
- Profile quality depends on sample diversity
- May need multiple iterations for optimal results
- Semantic meaning preservation prioritized over extreme transformation

## 🔒 Privacy

- All processing happens in-session (no permanent storage)
- Profiles are stored temporarily in memory
- No data is collected or shared
- Your writing samples are not saved

## 📞 Support & Feedback

- **GitHub**: [Style-Echo-AI-Humanizer](https://github.com/Ab-Romia/Style-Echo-AI-Humanizer)
- **Issues**: Report bugs or request features on GitHub
- **Author**: Ab-Romia (aabouroumia@gmail.com)

## 📜 License

MIT License - See [LICENSE](https://github.com/Ab-Romia/Style-Echo-AI-Humanizer/blob/main/LICENSE) for details

---

**Note**: This tool is for educational and research purposes. Please use responsibly and ethically. Always ensure proper attribution and follow academic integrity guidelines.
