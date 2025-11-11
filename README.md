# VoicePrint: AI Text Humanizer with Style Learning

Transform AI-generated text to match your unique writing voice. VoicePrint learns from your writing samples and applies your personal style to any AI-generated content while bypassing detection.

## Problem Statement

AI-generated text has become increasingly detectable through sophisticated detection tools like GPTZero and Originality.ai. Current "humanizer" tools fail in two critical ways:

1. **Generic Output**: They produce text that doesn't match any specific writing style, making it obvious the content was processed
2. **Poor Detection Bypass**: Simple paraphrasing isn't enough to fool modern AI detectors that analyze linguistic patterns

VoicePrint solves both problems by learning your actual writing patterns and applying them systematically while removing AI fingerprints.

## Key Features

- **Style Learning**: Analyzes 3-10 writing samples to build a complete profile of your writing voice
- **Linguistic Analysis**: Extracts sentence patterns, vocabulary preferences, and punctuation habits
- **AI Detection Removal**: Identifies and removes common AI tells (hedging phrases, perfect grammar, repetitive structures)
- **Style Transfer**: Applies your specific writing patterns to transform text
- **Validation**: Ensures output matches your style (85%+ similarity) while preserving meaning

## Quick Start

### Backend Setup

```bash
cd backend

# Run the setup script
./setup.sh

# Start the server
./start.sh
```

The API will be running at `http://localhost:8000`. Visit `http://localhost:8000/docs` for the interactive API documentation.

### Test the API

```bash
# Make sure the server is running first
python test_api.py
```

## How It Works

### 1. Style Profile Creation

When you provide 3-10 writing samples (minimum 500 words total), VoicePrint extracts:

**Linguistic Features:**
- Average sentence length with standard deviation
- Sentence complexity (dependency tree depth)
- Lexical diversity (type-token ratio)
- Part-of-speech distribution
- Punctuation patterns (commas, semicolons, em dashes, etc.)
- Readability scores (Flesch-Kincaid, Gunning Fog, etc.)

**Stylometric Markers:**
- Function word frequencies (the, and, but, so, etc.)
- N-gram patterns (common phrase preferences)
- Contraction usage rate
- Passive vs active voice ratio
- Sentence starter patterns
- Transition word usage

**Semantic Embeddings:**
- Sentence-BERT embeddings for each sample
- Style centroid vector (averaged embeddings)
- Consistency metrics across samples

### 2. AI Detection Removal

The system analyzes input text for AI tells:
- Perfect grammar patterns
- Repetitive sentence structures (especially "lists of three")
- Overly balanced parallel construction
- Hedging language ("it's important to note", "it's worth mentioning")
- Lack of natural variation

Then applies strategic changes:
- Removes AI hedging phrases
- Breaks up symmetrical structures
- Adds natural filler words ("well", "actually", "I mean")
- Introduces minor typos at human rates (0-2%)
- Varies sentence length dramatically

### 3. Style Transfer

For each sentence in the text:
- Adjusts length to match your average ± standard deviation
- Replaces words outside your typical vocabulary
- Matches your formality level (contractions vs formal language)
- Restructures syntax to match your preferred patterns
- Adjusts passive/active voice ratio
- Replicates your punctuation style

### 4. Validation

Output is validated by:
- Computing cosine similarity with your style centroid (target: 85%+)
- Checking linguistic features align with your profile
- Verifying readability scores fall within your range
- Confirming semantic meaning is preserved (BERT score)

If validation fails, the system automatically retries with adjusted parameters.

## API Endpoints

### Create Style Profile
```bash
POST /api/v1/profiles
{
  "user_id": "your_user_id",
  "samples": ["text sample 1", "text sample 2", "text sample 3"],
  "profile_name": "My Writing Style"
}
```

### Humanize Text
```bash
POST /api/v1/humanize
{
  "profile_id": "profile_id_from_creation",
  "text": "AI-generated text here...",
  "strength": 0.7,
  "preserve_meaning": true
}
```

### Quick Analysis
```bash
POST /api/v1/analyze
{
  "text": "Any text to analyze..."
}
```

### Get Profile
```bash
GET /api/v1/profiles/{profile_id}
```

### List User Profiles
```bash
GET /api/v1/users/{user_id}/profiles
```

### Delete Profile
```bash
DELETE /api/v1/profiles/{profile_id}
```

## Architecture

```
VoicePrint/
├── backend/
│   ├── app/
│   │   ├── api/           # FastAPI routes
│   │   ├── models/        # Data models (StyleProfile)
│   │   ├── schemas/       # Pydantic schemas
│   │   ├── services/      # Core business logic
│   │   │   ├── linguistic_analyzer.py      # Extract linguistic features
│   │   │   ├── stylometric_analyzer.py     # Extract stylometric markers
│   │   │   ├── embedding_analyzer.py       # Sentence-BERT embeddings
│   │   │   ├── ai_detector_remover.py      # Remove AI patterns
│   │   │   ├── style_transfer.py           # Apply writing style
│   │   │   ├── validator.py                # Validate output quality
│   │   │   └── voiceprint_service.py       # Main orchestrator
│   │   ├── config.py      # Configuration
│   │   └── main.py        # FastAPI app
│   ├── requirements.txt   # Python dependencies
│   ├── setup.sh          # Setup script
│   ├── start.sh          # Start server script
│   └── test_api.py       # Test suite
└── frontend/             # (Coming soon: Next.js interface)
```

## Technology Stack

**Backend:**
- FastAPI for the REST API
- PyTorch & HuggingFace Transformers for embeddings
- Sentence-BERT for semantic analysis
- spaCy for linguistic parsing
- NLTK for stylometric features
- textstat for readability metrics

**Planned:**
- PostgreSQL for user data
- ChromaDB for embedding storage
- Redis for caching
- Next.js frontend with TypeScript
- TailwindCSS for styling

## Related Work & Citations

This project builds on established research in several areas:

**Style Transfer:**
- Jin et al. (2021) - "Deep Learning for Text Style Transfer: A Survey" - Justification for neural approach
- Syed et al. (2020) - "Evaluating Prose Style Transfer with the Bible" - Evaluation metrics

**Stylometric Analysis:**
- Juola (2006) - "Stylometric Analysis of Literary Texts" - Function word and n-gram features
- Mahmood et al. (2019) - "Adversarial Authorship Attribution" - Feasibility of mimicking styles

**AI Detection:**
- Mitchell et al. (2023) - "DetectGPT: Zero-Shot Machine-Generated Text Detection" - AI detection patterns
- Uchendu et al. (2020) - "The Limitations of Stylometry for Detecting Machine-Generated Fake News" - Detection weaknesses
- OpenAI (2023) - GPT-4 Technical Report - Documented AI text characteristics

**Embeddings:**
- Reimers & Gurevych (2019) - "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"

**Paraphrasing:**
- Li et al. (2018) - "Paraphrase Generation with Deep Reinforcement Learning"

## Example Usage

```python
import requests

# 1. Create a profile from your writing samples
response = requests.post("http://localhost:8000/api/v1/profiles", json={
    "user_id": "user123",
    "samples": [
        "Here's how I actually write. I tend to use shorter sentences. Sometimes fragments, even.",
        "Look, I'm not saying this is perfect, but it works for me. The key is being natural.",
        "You know what bugs me? When people write like robots. Just write how you'd talk."
    ]
})

profile_id = response.json()["profile_id"]

# 2. Humanize AI-generated text
ai_text = """
It is important to note that effective communication requires careful consideration
of multiple factors. Furthermore, one must ensure that the message is conveyed clearly
and concisely. The implementation of proper techniques is essential for success.
"""

response = requests.post("http://localhost:8000/api/v1/humanize", json={
    "profile_id": profile_id,
    "text": ai_text,
    "strength": 0.7
})

print(response.json()["humanized_text"])
# Output will match your casual, direct writing style
```

## Development Roadmap

- [x] Core linguistic analysis engine
- [x] Stylometric feature extraction
- [x] Sentence-BERT embedding system
- [x] AI detection removal pipeline
- [x] Style transfer engine
- [x] Validation system
- [x] REST API with FastAPI
- [ ] PostgreSQL integration
- [ ] ChromaDB for embeddings
- [ ] Redis caching layer
- [ ] Next.js frontend
- [ ] Chrome extension
- [ ] User authentication
- [ ] Rate limiting
- [ ] API key management
- [ ] Freemium model implementation

## Known Limitations

1. **Passive to Active Conversion**: Current implementation uses simplified patterns. Could be improved with more sophisticated syntactic transformations.

2. **Vocabulary Replacement**: Currently focuses on formality level. Future versions will include specific word-level vocabulary matching.

3. **In-Memory Storage**: Uses in-memory storage for profiles. Production needs PostgreSQL + ChromaDB.

4. **No LLM Integration**: Currently doesn't use GPT-4 for paraphrasing. Can be added for stronger transformations.

## Contributing

This is an educational project demonstrating AI text transformation and style learning. Feel free to fork and experiment!

## License

MIT License - See LICENSE file for details

## Contact

Built by Ab-Romia - [aabouroumia@gmail.com](mailto:aabouroumia@gmail.com)

## Acknowledgments

Thanks to the researchers whose work made this possible, and to the open-source communities behind spaCy, HuggingFace, FastAPI, and all the other tools used in this project.
