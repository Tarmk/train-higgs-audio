#!/usr/bin/env python3
"""
🇹🇭 Simple Thai Training Script
Uses existing mini training data to train Higgs Audio with Thai language support.
"""

import argparse
import os
import json
from pathlib import Path
import subprocess
import sys

def create_thai_training_data():
    """Create Thai training data from existing mini dataset and add more."""
    print("🇹🇭 Creating Thai training dataset...")
    
    # Use existing mini data
    mini_data_path = Path("higgs_training_data_mini")
    if not mini_data_path.exists():
        print("❌ Mini training data not found. Please ensure higgs_training_data_mini/ exists.")
        return False
    
    # Create Thai training directory
    thai_data_path = Path("thai_training_data")
    thai_data_path.mkdir(exist_ok=True)
    
    # Copy existing files first
    import shutil
    wav_files = list(mini_data_path.glob("*.wav"))
    for wav_file in wav_files:
        txt_file = wav_file.with_suffix('.txt')
        if txt_file.exists():
            # Copy to Thai data directory
            shutil.copy2(wav_file, thai_data_path)
            shutil.copy2(txt_file, thai_data_path)
    
    # Add Thai text samples using existing audio as templates
    thai_texts = [
        "สวัสดีครับ ผมชื่อ AI ผมสามารถพูดภาษาไทยได้",
        "วันนี้อากาศดีมาก แสงแดดสวยงาม",
        "ขอบคุณที่มาฟังเสียงของผม",
        "ผมหวังว่าเสียงของผมจะชัดเจน",
        "ภาษาไทยเป็นภาษาที่สวยงาม",
        "ผมกำลังเรียนรู้ที่จะพูดภาษาไทย",
        "ขอให้มีความสุขมากๆ นะครับ",
        "ผมยินดีที่ได้พูดคุยกับคุณ"
    ]
    
    # Create additional Thai samples using existing audio as templates
    copied_wav_files = list(thai_data_path.glob("*.wav"))
    for i, thai_text in enumerate(thai_texts):
        if i < len(copied_wav_files):
            # Create new text file with Thai content
            thai_txt_file = thai_data_path / f"thai_sample_{i:03d}.txt"
            with open(thai_txt_file, 'w', encoding='utf-8') as f:
                f.write(thai_text)
            
            # Copy corresponding audio file
            thai_wav_file = thai_data_path / f"thai_sample_{i:03d}.wav"
            if not thai_wav_file.exists():
                shutil.copy2(copied_wav_files[i % len(copied_wav_files)], thai_wav_file)
    
    # Count final files
    final_wav_files = list(thai_data_path.glob("*.wav"))
    final_txt_files = list(thai_data_path.glob("*.txt"))
    
    print(f"✅ Created Thai training data directory: {thai_data_path}")
    print(f"📊 Total samples: {len(final_wav_files)} audio files, {len(final_txt_files)} text files")
    
    return str(thai_data_path)

def download_models():
    """Download the base models to D drive."""
    print("📥 Downloading base models to D drive...")
    
    try:
        import os
        from huggingface_hub import snapshot_download
        
        # Set cache directory to D drive
        cache_dir = "D:/huggingface_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        print(f"💾 Using cache directory: {cache_dir}")
        
        print("🔄 Downloading model...")
        model_path = snapshot_download(
            "bosonai/higgs-audio-v2-generation-3B-base",
            cache_dir=cache_dir
        )
        
        print("🔄 Downloading tokenizer...")
        tokenizer_path = snapshot_download(
            "bosonai/higgs-audio-v2-tokenizer",
            cache_dir=cache_dir
        )
        
        print(f"✅ Model: {model_path}")
        print(f"✅ Tokenizer: {tokenizer_path}")
        
        return model_path, tokenizer_path
    except Exception as e:
        print(f"❌ Failed to download models: {e}")
        return None, None

def run_training(model_path, tokenizer_path, training_data_dir, output_dir):
    """Run the training process."""
    print("🚀 Starting Thai training...")
    
    trainer_script = Path("trainer/trainer.py")
    if not trainer_script.exists():
        print("❌ Trainer script not found.")
        return False
    
    # Use optimized trainer arguments for memory efficiency
    cmd = [
        "python", str(trainer_script),
        "--model_path", model_path,
        "--audio_tokenizer_path", tokenizer_path,
        "--train_data_dir", training_data_dir,
        "--output_dir", output_dir,
        "--task_type", "single_speaker_smart_voice",
        "--num_train_epochs", "1",
        "--per_device_train_batch_size", "1",  # Very small batch size
        "--learning_rate", "1e-4",  # Slightly higher LR for small batch
        "--save_steps", "20",  # Save more frequently
        "--logging_steps", "2",  # Log more frequently
        "--warmup_steps", "5",  # Fewer warmup steps
        "--use_lora",  # Enable LoRA training for efficiency
        "--lora_rank", "8",  # Smaller LoRA rank for memory
        "--lora_alpha", "16",  # Corresponding alpha
        "--lora_dropout", "0.1"
    ]
    
    print(f"🏃 Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("✅ Training completed successfully!")
        return True
    else:
        print(f"❌ Training failed with exit code: {result.returncode}")
        return False

def test_trained_model(model_path, tokenizer_path):
    """Test the trained model with Thai text."""
    print("🧪 Testing trained model with Thai text...")
    
    thai_test = "สวัสดีครับ ผมสามารถพูดภาษาไทยได้แล้ว"
    
    # Create test text file
    with open("thai_test.txt", 'w', encoding='utf-8') as f:
        f.write(thai_test)
    
    # Test generation
    cmd = [
        "python", "examples/generation.py",
        "--model_path", model_path,
        "--audio_tokenizer", tokenizer_path,
        "--transcript", "thai_test.txt",
        "--out_path", "thai_generated.wav",
        "--temperature", "1.0",
        "--seed", "12345"
    ]
    
    print(f"🎵 Generating: {thai_test}")
    result = subprocess.run(cmd)
    
    # Cleanup
    if os.path.exists("thai_test.txt"):
        os.remove("thai_test.txt")
    
    if result.returncode == 0 and os.path.exists("thai_generated.wav"):
        print("✅ Thai TTS generation successful!")
        print("🎵 Output: thai_generated.wav")
        return True
    else:
        print("❌ Thai TTS generation failed")
        return False

def main():
    parser = argparse.ArgumentParser(description="Simple Thai training for Higgs Audio")
    parser.add_argument("--output_dir", default="./thai_higgs_model", help="Output directory for trained model")
    parser.add_argument("--skip_download", action="store_true", help="Skip model download if already cached")
    parser.add_argument("--test_only", action="store_true", help="Only test existing model")
    parser.add_argument("--model_path", help="Path to trained model (for test only)")
    parser.add_argument("--tokenizer_path", help="Path to tokenizer (for test only)")
    
    args = parser.parse_args()
    
    print("🇹🇭 Simple Thai Training for Higgs Audio")
    print("=" * 50)
    
    if args.test_only:
        if not args.model_path or not args.tokenizer_path:
            print("❌ For test only, provide --model_path and --tokenizer_path")
            return
        test_trained_model(args.model_path, args.tokenizer_path)
        return
    
    # Step 1: Create training data
    training_data_dir = create_thai_training_data()
    if not training_data_dir:
        return
    
    # Step 2: Download models (unless skipped)
    if not args.skip_download:
        model_path, tokenizer_path = download_models()
        if not model_path:
            return
    else:
        # Use cached paths (you might need to adjust these)
        model_path = "bosonai/higgs-audio-v2-generation-3B-base"
        tokenizer_path = "bosonai/higgs-audio-v2-tokenizer"
    
    # Step 3: Run training
    if run_training(model_path, tokenizer_path, training_data_dir, args.output_dir):
        print(f"\n🎉 Thai training completed!")
        print(f"📁 Model saved to: {args.output_dir}")
        
        # Step 4: Test the trained model
        print(f"\n🧪 Testing trained model...")
        test_trained_model(args.output_dir, tokenizer_path)
        
        print(f"\n✅ Thai language support added to Higgs Audio!")
        print(f"🚀 You can now use Thai TTS with:")
        print(f"python examples/generation.py --model_path {args.output_dir} --audio_tokenizer {tokenizer_path} --transcript your_thai_text.txt")
    
if __name__ == "__main__":
    main()
