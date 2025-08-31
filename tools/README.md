# 🛠️ Tools Directory

This directory contains essential tools for the Higgs Audio project.

## 📁 Contents

### Core Thai Processing
- **`prepare_gigaspeech2_thai.py`** - Thai GigaSpeech2 dataset preparation with transcript integration
  - Loads Thai transcripts from TSV files
  - Streams audio data with transcript matching
  - Saves paired audio-text files for training

### Utilities  
- **`mp4towav.sh`** - Audio format conversion utility (MP4 to WAV)

## 🚀 Quick Start

For Thai processing, use the consolidated script in the project root:

```bash
# Test setup
python run_thai_complete.py test

# Prepare data
python run_thai_complete.py prepare-data --output_dir ./thai_data --max_samples 1000

# Full pipeline
python run_thai_complete.py full-pipeline --model_path your-model --audio_tokenizer_path your-tokenizer
```

## 🧹 Cleanup Notes

The following files have been removed as they were unused or redundant:
- `hf_login.py` - Authentication now built into main scripts
- `setup_env.ps1/sh` - Environment setup scripts  
- `test_audio_data.py` - Old test script with hardcoded paths
- `convert_jsonl_to_dataset.py` - JSONL conversion (not Thai-related)
- `validate_dataset.py` - Generic validation (not Thai-specific)

All Thai processing functionality is now consolidated in `run_thai_complete.py` at the project root.
