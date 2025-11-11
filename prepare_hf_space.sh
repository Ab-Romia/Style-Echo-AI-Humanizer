#!/bin/bash

# Prepare VoicePrint for Hugging Face Spaces deployment
# This script creates a clean directory ready to push to HF Spaces

echo "🚀 Preparing VoicePrint for Hugging Face Spaces"
echo "================================================"
echo ""

# Create deployment directory
DEPLOY_DIR="voiceprint_hf_space"

if [ -d "$DEPLOY_DIR" ]; then
    echo "⚠️  Directory $DEPLOY_DIR already exists."
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$DEPLOY_DIR"
    else
        echo "❌ Aborting"
        exit 1
    fi
fi

echo "📁 Creating deployment directory..."
mkdir -p "$DEPLOY_DIR"

# Copy necessary files
echo "📋 Copying files..."

# Main application files
cp app.py "$DEPLOY_DIR/"
cp README_HF.md "$DEPLOY_DIR/README.md"
cp requirements-hf.txt "$DEPLOY_DIR/requirements.txt"

# Copy backend directory
cp -r backend "$DEPLOY_DIR/"

# Copy license
cp LICENSE "$DEPLOY_DIR/" 2>/dev/null || echo "  ⚠️  LICENSE not found, skipping"

echo ""
echo "✅ Deployment directory prepared: $DEPLOY_DIR"
echo ""
echo "📤 Next steps to deploy to Hugging Face Spaces:"
echo ""
echo "1. Create a new Space on Hugging Face:"
echo "   https://huggingface.co/new-space"
echo "   - Name: voiceprint (or your choice)"
echo "   - SDK: Gradio"
echo "   - Hardware: CPU Basic (free)"
echo ""
echo "2. Clone your new Space:"
echo "   git clone https://huggingface.co/spaces/Ab-Romia/voiceprint"
echo "   cd voiceprint"
echo ""
echo "3. Copy files:"
echo "   cp -r ../$DEPLOY_DIR/* ."
echo ""
echo "4. Commit and push:"
echo "   git add ."
echo "   git commit -m \"Initial VoicePrint deployment\""
echo "   git push"
echo ""
echo "5. Wait for build (5-10 minutes)"
echo ""
echo "🎉 Your Space will be live at:"
echo "   https://huggingface.co/spaces/Ab-Romia/voiceprint"
echo ""
echo "================================================"
