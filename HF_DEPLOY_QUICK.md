# Quick Hugging Face Spaces Deployment

## For Username: Ab-Romia

### ⚡ Super Quick Deploy (5 minutes)

1. **Create Space**
   - Go to: https://huggingface.co/new-space
   - Name: `voiceprint`
   - SDK: Select **Gradio**
   - Hardware: CPU Basic (free)
   - Visibility: Public
   - Click **Create Space**

2. **Upload Files via Web UI**

   Click "Files" tab, then "Add file" → "Upload files"

   Upload these files (in order):
   ```
   ✓ app.py
   ✓ requirements-hf.txt → RENAME to "requirements.txt"
   ✓ README_HF.md → RENAME to "README.md"
   ✓ backend/ (entire folder)
   ```

3. **Wait for Build**
   - Build starts automatically
   - Takes 5-10 minutes (downloads models)
   - Watch logs in "Build" tab
   - Green "Running" status when ready

4. **Test Your Space**
   - Visit: `https://huggingface.co/spaces/Ab-Romia/voiceprint`
   - Try the example in the interface
   - Share the URL!

---

## 🚀 Alternative: Git Push Method

```bash
# 1. Clone your new Space
git clone https://huggingface.co/spaces/Ab-Romia/voiceprint
cd voiceprint

# 2. Copy files from this repo
cp /path/to/Style-Echo-AI-Humanizer/app.py .
cp /path/to/Style-Echo-AI-Humanizer/requirements-hf.txt requirements.txt
cp /path/to/Style-Echo-AI-Humanizer/README_HF.md README.md
cp -r /path/to/Style-Echo-AI-Humanizer/backend .

# 3. Commit and push
git add .
git commit -m "Initial VoicePrint deployment"
git push

# 4. Wait for build at:
# https://huggingface.co/spaces/Ab-Romia/voiceprint
```

---

## 🐳 Test Locally with Docker First

```bash
# From the project root
docker build -t voiceprint .
docker run -p 7860:7860 voiceprint

# Open browser to: http://localhost:7860
# If it works locally, it will work on HF Spaces!
```

---

## 📋 Files Needed for HF Spaces

| Local File | → | HF Spaces File |
|------------|---|----------------|
| `app.py` | → | `app.py` |
| `requirements-hf.txt` | → | `requirements.txt` |
| `README_HF.md` | → | `README.md` |
| `backend/` | → | `backend/` |

**Important**: Rename files as shown above!

---

## ✅ Verification Checklist

After deployment, test these features:

- [ ] Space loads without errors
- [ ] Can see the 3 tabs (Create Profile, Humanize, Analyze)
- [ ] Can create a profile with sample text
- [ ] Get a profile ID back
- [ ] Can humanize AI text using the profile
- [ ] Can analyze any text
- [ ] See results and metrics

---

## 🐛 Troubleshooting

### Build fails with "Model not found"
✅ **Fixed**: requirements-hf.txt includes model downloads

### "Out of memory" error
✅ **Solution**: Use CPU Basic hardware (should be enough)

### Interface doesn't load
- Check "Build" tab for errors
- Ensure all files uploaded correctly
- Verify requirements.txt exists (not requirements-hf.txt)

### Slow first start
✅ **Normal**: First run downloads 2GB of ML models
✅ **Future**: Subsequent starts are much faster

---

## 📊 Expected Performance

| Metric | Value |
|--------|-------|
| First build time | 5-10 minutes |
| Startup time (cold) | 30-60 seconds |
| Startup time (warm) | 10-20 seconds |
| Profile creation | 5-15 seconds |
| Text humanization | 3-8 seconds |
| Analysis | 1-3 seconds |

---

## 🎯 Your Space URL

Once deployed, access at:
```
https://huggingface.co/spaces/Ab-Romia/voiceprint
```

Share this URL to let others use VoicePrint!

---

## 💡 Pro Tips

1. **Test locally first** with `python app.py`
2. **Use the helper script**: `./prepare_hf_space.sh`
3. **Check HF logs** if something fails
4. **Start with CPU Basic** (free tier works fine)
5. **Monitor Space analytics** in Settings

---

## 📞 Need Help?

- **Full Guide**: See DEPLOY.md
- **Testing**: Run test_gradio.py
- **Issues**: GitHub Issues
- **Docs**: README.md

---

**Ready to deploy? Let's go! 🚀**
