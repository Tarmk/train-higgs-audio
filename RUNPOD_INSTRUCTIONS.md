# 🚀 RunPod A40 Setup Instructions for Thai TTS

## Step-by-Step RunPod Setup

### 1. 🌐 Go to RunPod
- Visit: https://runpod.io
- Login to your account

### 2. 🚀 Deploy Pod
- Click **"Deploy"** or **"+ Deploy"**

### 3. 📋 Template Selection
- Choose: **"PyTorch"** or **"RunPod PyTorch"** template
- Or: **"Jupyter PyTorch"** if you want Jupyter

### 4. 🎯 GPU Selection
- **Filter by GPU**: Select **"A40"** 
- **VRAM**: 48GB
- **Price**: ~$0.50/hour
- Click on an available A40 pod

### 5. 💾 Storage Configuration
- **Container Disk**: 50GB (minimum)
- **Volume Disk**: 100GB+ (for models)

### 6. ⚙️ Pod Settings
- **Expose HTTP Ports**: ✅ Check this
- **Start Jupyter**: ✅ Enable (if using Jupyter template)

### 7. 🚀 Deploy
- Click **"Deploy On-Demand"**
- Wait 5-10 minutes for pod to start

---

## 📁 File Upload Instructions

### 8. 📤 Upload Your Files
Once pod is running, upload these files:

**Required Files:**
```
- runpod_thai_training.py
- trainer/ (entire directory)
- higgs_training_data_mini/ (if you have it)
- requirements_train.txt
- boson_multimodal/ (entire directory)
```

**How to Upload:**
- **Method 1**: Use Jupyter file browser (drag & drop)
- **Method 2**: Use `wget` or `git clone` if files are online
- **Method 3**: Use RunPod's file sync feature

---

## 🎯 Training Execution

### 9. 🖥️ Open Terminal
- **If using Jupyter**: New → Terminal
- **If using SSH**: Connect via SSH

### 10. 📦 Install Dependencies
```bash
pip install --upgrade pip
pip install peft transformers accelerate
pip install librosa soundfile
```

### 11. 🚀 Start Training
```bash
python runpod_thai_training.py
```

### 12. ⏳ Wait for Training
- **A40 Training Time**: 1-2 hours
- **Expected Cost**: $0.50-1.00
- **Progress**: Watch the logs for training progress

---

## 📊 What to Expect

### Training Output:
```
🇹🇭 RunPod A40 Thai TTS Training
==================================================
✅ GPU: NVIDIA A40
💾 VRAM: 48.0GB
🎯 A40 detected - using optimized settings
📁 Found existing mini training data
✅ Created 10 text files, 10 audio files
📥 Downloading Higgs Audio models...
🚀 Starting optimized training for a40...
📊 Settings: batch_size=2, lora_rank=16
⏳ Training will take 1-2 hours on A40...
```

### Success Message:
```
🎉 Training completed!
📁 Thai model saved to: ./thai_higgs_model
🧪 Testing Thai TTS...
✅ Thai TTS generation successful!
🌟 SUCCESS! You now have Thai TTS capability!
```

---

## 📥 Download Your Model

### 13. 💾 Download Trained Model
- **Folder**: `thai_higgs_model/`
- **Size**: ~2-3GB
- **Method**: Use Jupyter file browser or `scp`

### 14. 🧪 Test Locally
```bash
python examples/generation.py \
  --model_path ./thai_higgs_model \
  --audio_tokenizer bosonai/higgs-audio-v2-tokenizer \
  --transcript your_thai_text.txt \
  --out_path thai_speech.wav
```

---

## 💰 Cost Breakdown
- **A40 Rental**: $0.50/hour
- **Training Time**: 1-2 hours
- **Total Cost**: $0.50-1.00
- **Storage**: Minimal (included)

## 🎯 Final Result
**You'll have a Thai-speaking Higgs Audio model that can:**
- Convert any Thai text to natural speech
- Use different voices (with reference audio)
- Generate high-quality 24kHz audio
- Work with your existing TTS pipeline

**Ready to start? Go to RunPod and follow steps 1-7! 🚀**
