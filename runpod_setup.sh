#!/bin/bash
# 🚀 RunPod A40 Setup Script for Thai TTS Training
# Copy-paste this entire script into RunPod terminal

echo "🇹🇭 Setting up Thai TTS Training on RunPod A40"
echo "================================================"

# Update system and install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate peft datasets huggingface_hub
pip install librosa soundfile torchaudio-augmentations

# Clone the repository (you'll need to upload your files or use git)
echo "📁 Setting up project files..."
# If you have a git repo: git clone YOUR_REPO_URL
# Otherwise, you'll upload the files manually

# Create directories
mkdir -p thai_training_data
mkdir -p thai_higgs_model

echo "✅ Setup complete!"
echo "📋 Next steps:"
echo "1. Upload your project files to this pod"
echo "2. Run: python runpod_thai_training.py"
echo "3. Wait 1-2 hours for training to complete"
echo "4. Download your trained Thai TTS model!"
