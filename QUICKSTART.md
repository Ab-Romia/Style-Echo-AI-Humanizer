# VoicePrint Quick Start Guide

Get up and running with VoicePrint in 5 minutes.

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- 2-3 GB free disk space (for ML models)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Ab-Romia/Style-Echo-AI-Humanizer.git
cd Style-Echo-AI-Humanizer
```

### 2. Set Up the Backend

```bash
cd backend
./setup.sh
```

This script will:
- Create a Python virtual environment
- Install all required packages
- Download the spaCy language model
- Download NLTK data
- Create a `.env` file from the template

**Note**: The setup might take a few minutes to download all the ML models.

### 3. Start the Server

```bash
./start.sh
```

The API will start at `http://localhost:8000`

### 4. Test the API

Open a new terminal and run:

```bash
python test_api.py
```

You should see output showing successful tests of:
- Server health check
- Quick text analysis
- Profile creation
- Text humanization
- Profile management

## Using the API

### Interactive Documentation

Visit `http://localhost:8000/docs` in your browser to see the interactive API documentation where you can test all endpoints.

### Example: Humanize Your First Text

1. **Create a profile** with your writing samples:

```bash
curl -X POST "http://localhost:8000/api/v1/profiles" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "demo_user",
    "samples": [
      "Your first writing sample here...",
      "Your second writing sample here...",
      "Your third writing sample here..."
    ],
    "profile_name": "Demo Profile"
  }'
```

Save the `profile_id` from the response.

2. **Humanize AI text** using your profile:

```bash
curl -X POST "http://localhost:8000/api/v1/humanize" \
  -H "Content-Type: application/json" \
  -d '{
    "profile_id": "YOUR_PROFILE_ID_HERE",
    "text": "It is important to note that AI-generated text often contains characteristic patterns.",
    "strength": 0.7,
    "preserve_meaning": true
  }'
```

## Common Issues

### Issue: "spaCy model not found"

**Solution**: Run the download command manually:
```bash
source venv/bin/activate
python -m spacy download en_core_web_sm
```

### Issue: "NLTK data not found"

**Solution**: Download NLTK data manually:
```bash
source venv/bin/activate
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Issue: "Port 8000 already in use"

**Solution**: Either kill the process using port 8000 or change the port in `.env`:
```bash
# In backend/.env
API_PORT=8001
```

### Issue: Installation taking too long

**Solution**: The first setup downloads large ML models (PyTorch, Transformers, etc.). This is normal and only happens once. Grab a coffee!

## What's Next?

- Read the full [README.md](README.md) for detailed documentation
- Check out the [API documentation](http://localhost:8000/docs) for all available endpoints
- Experiment with different strength values (0.0 to 1.0) to find what works best
- Try analyzing your own writing samples to see what features are extracted

## Getting Help

- Check the [README.md](README.md) for detailed information
- Look at the test file (`test_api.py`) for example usage
- Open an issue on GitHub if you find bugs

Happy humanizing!
