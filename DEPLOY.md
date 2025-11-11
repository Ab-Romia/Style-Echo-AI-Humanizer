# VoicePrint Deployment Guide

This guide covers deploying VoicePrint to Hugging Face Spaces and other platforms.

## 🚀 Deploy to Hugging Face Spaces

### Prerequisites

1. A Hugging Face account (sign up at https://huggingface.co)
2. Git installed on your machine

### Method 1: Direct Upload (Easiest)

1. **Create a new Space:**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Name it "voiceprint" (or your preferred name)
   - Select "Gradio" as the SDK
   - Choose "Public" or "Private"
   - Click "Create Space"

2. **Upload files:**
   - Click "Files" tab in your Space
   - Upload these files:
     - `app.py`
     - `requirements-hf.txt` (rename to `requirements.txt`)
     - `README_HF.md` (rename to `README.md`)
     - The entire `backend/` directory

3. **Your Space will automatically build and deploy!**

### Method 2: Git Clone & Push (Recommended)

1. **Clone your new Space:**
   ```bash
   git clone https://huggingface.co/spaces/Ab-Romia/voiceprint
   cd voiceprint
   ```

2. **Copy VoicePrint files:**
   ```bash
   # From your VoicePrint repo
   cp app.py ../voiceprint/
   cp requirements-hf.txt ../voiceprint/requirements.txt
   cp README_HF.md ../voiceprint/README.md
   cp -r backend ../voiceprint/
   ```

3. **Commit and push:**
   ```bash
   cd ../voiceprint
   git add .
   git commit -m "Initial VoicePrint deployment"
   git push
   ```

4. **Watch it build:**
   - Go to your Space URL: https://huggingface.co/spaces/Ab-Romia/voiceprint
   - The build will start automatically
   - Wait 5-10 minutes for models to download

### Method 3: Using This Repo Directly

1. **Create Space with repo URL:**
   - When creating a Space, use "Import from repository"
   - Enter: `https://github.com/Ab-Romia/Style-Echo-AI-Humanizer`
   - Select files to import

2. **Configure Space:**
   - Make sure `app.py` is the app file
   - Rename `requirements-hf.txt` to `requirements.txt`
   - Rename `README_HF.md` to `README.md`

## 🐳 Docker Deployment

### Build and Run Locally

```bash
# Build the Docker image
docker build -t voiceprint .

# Run the container
docker run -p 7860:7860 voiceprint

# Access at http://localhost:7860
```

### Using Docker Compose

```bash
# Start the service
docker-compose up

# Stop the service
docker-compose down
```

## 🖥️ Local Development

### Quick Start

```bash
cd backend
./setup.sh
./start.sh
```

### Manual Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-hf.txt

# Download models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Run the Gradio app
python app.py
```

## 🌐 Other Deployment Options

### Railway

1. Fork this repo to your GitHub
2. Create new project on Railway
3. Connect your GitHub repo
4. Railway will auto-detect Dockerfile
5. Add environment variables if needed
6. Deploy!

### Render

1. Create new Web Service
2. Connect your GitHub repo
3. Select "Docker" as environment
4. Deploy with default settings

### AWS EC2

```bash
# SSH into your EC2 instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Install Docker
sudo apt-get update
sudo apt-get install docker.io docker-compose -y

# Clone repo
git clone https://github.com/Ab-Romia/Style-Echo-AI-Humanizer
cd Style-Echo-AI-Humanizer

# Build and run
sudo docker build -t voiceprint .
sudo docker run -d -p 80:7860 voiceprint

# Access at http://your-instance-ip
```

## 📊 Resource Requirements

### Minimum Requirements

- **RAM**: 4 GB
- **CPU**: 2 cores
- **Storage**: 5 GB
- **Python**: 3.10+

### Recommended for Production

- **RAM**: 8 GB
- **CPU**: 4 cores
- **Storage**: 10 GB
- **GPU**: Optional (speeds up embeddings)

## 🔧 Configuration

### Environment Variables

Create a `.env` file with:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=7860
DEBUG=False

# Model Configuration
SENTENCE_TRANSFORMER_MODEL=sentence-transformers/all-MiniLM-L6-v2
SPACY_MODEL=en_core_web_sm

# Optional: for advanced features
OPENAI_API_KEY=your_key_here
```

### Hugging Face Secrets

In your Space settings, add secrets:
- `OPENAI_API_KEY` (if using GPT-4 paraphrasing)

## 🐛 Troubleshooting

### "Model not found" error

```bash
# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Out of memory errors

- Reduce batch size in configuration
- Use smaller embedding models
- Increase container memory limit

### Slow startup on Hugging Face

- This is normal! First build downloads ~2GB of models
- Subsequent starts will be much faster
- Consider using persistent storage

## 📈 Scaling

### For High Traffic

1. **Use Redis for caching:**
   ```bash
   pip install redis
   # Add Redis connection in config
   ```

2. **Enable PostgreSQL:**
   ```bash
   # Update DATABASE_URL in .env
   # Profiles will persist across restarts
   ```

3. **Add load balancing:**
   - Deploy multiple instances
   - Use Nginx or Cloudflare for load balancing

### Optimize Models

```python
# Use smaller models for faster inference
SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fast
# vs
SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Accurate but slower
```

## 🔐 Security Considerations

1. **Rate Limiting**: Add rate limits to prevent abuse
2. **Input Validation**: Already implemented in schemas
3. **API Keys**: Use secrets for any external APIs
4. **CORS**: Configure allowed origins in production

## 📞 Support

- **Issues**: https://github.com/Ab-Romia/Style-Echo-AI-Humanizer/issues
- **Discussions**: GitHub Discussions
- **Email**: aabouroumia@gmail.com

## 🎉 Success Checklist

- [ ] Files uploaded to Hugging Face Space
- [ ] Space is building (check logs)
- [ ] Space is running (green status)
- [ ] Can create a profile successfully
- [ ] Can humanize text
- [ ] Analysis feature works
- [ ] No error messages in logs

---

**Happy deploying! 🚀**
