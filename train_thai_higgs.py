#!/usr/bin/env python3
"""
🇹🇭 Thai Language Training Pipeline for Higgs Audio
This script trains Higgs Audio model with Thai GigaSpeech2 data to add Thai TTS capability.
"""

import argparse
import os
import sys
from pathlib import Path
import subprocess

def download_base_model():
    """Download the base Higgs Audio model for fine-tuning."""
    print("📥 Downloading base Higgs Audio model...")
    
    try:
        from huggingface_hub import snapshot_download
        
        # Download the base model and tokenizer
        model_path = snapshot_download("bosonai/higgs-audio-v2-generation-3B-base")
        tokenizer_path = snapshot_download("bosonai/higgs-audio-v2-tokenizer")
        
        print(f"✅ Base model downloaded to: {model_path}")
        print(f"✅ Audio tokenizer downloaded to: {tokenizer_path}")
        
        return model_path, tokenizer_path
    except Exception as e:
        print(f"❌ Failed to download models: {e}")
        return None, None

def prepare_thai_training_data(output_dir: str, max_samples: int = 1000):
    """Prepare Thai training data using our existing pipeline."""
    print(f"🇹🇭 Preparing Thai training data ({max_samples} samples)...")
    
    # Use our existing data preparation script
    cmd = [
        "python", "run_thai_complete.py", "prepare-data",
        "--output_dir", output_dir,
        "--max_samples", str(max_samples),
        "--target_size_gb", "2.0"  # Limit to 2GB for manageable training
    ]
    
    print(f"🚀 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Thai data preparation completed!")
        
        # Count prepared files
        data_path = Path(output_dir)
        wav_files = list(data_path.glob("*.wav"))
        txt_files = list(data_path.glob("*.txt"))
        
        print(f"📊 Prepared {len(wav_files)} audio files and {len(txt_files)} text files")
        return len(wav_files) > 0
    else:
        print(f"❌ Data preparation failed:")
        print(result.stderr)
        return False

def convert_to_training_format(data_dir: str, output_jsonl: str):
    """Convert prepared data to the training format expected by Higgs Audio."""
    print("🔄 Converting data to training format...")
    
    data_path = Path(data_dir)
    wav_files = sorted(data_path.glob("*.wav"))
    
    training_samples = []
    
    for wav_file in wav_files:
        # Find corresponding text file
        txt_file = wav_file.with_suffix('.txt')
        if txt_file.exists():
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            # Create training sample in the expected format
            sample = {
                "messages": [
                    {"role": "assistant", "content": text}
                ],
                "audios": [str(wav_file.absolute())]
            }
            training_samples.append(sample)
    
    # Write JSONL file
    import json
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for sample in training_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✅ Created training file: {output_jsonl}")
    print(f"📊 Total training samples: {len(training_samples)}")
    return len(training_samples) > 0

def run_thai_training(
    model_path: str,
    tokenizer_path: str,
    training_data: str,
    output_dir: str,
    epochs: int = 1,
    batch_size: int = 1,
    learning_rate: float = 5e-5
):
    """Run the actual training process."""
    print("🚀 Starting Thai language training...")
    print("=" * 60)
    
    # Prepare training command using the existing trainer
    cmd = [
        "python", "trainer/trainer.py",
        "--model_path", model_path,
        "--audio_tokenizer_path", tokenizer_path,
        "--data_file", training_data,
        "--output_dir", output_dir,
        "--num_train_epochs", str(epochs),
        "--per_device_train_batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--save_steps", "100",
        "--logging_steps", "10",
        "--warmup_steps", "50",
        "--bf16", "True",
        "--gradient_checkpointing", "True",
        "--dataloader_num_workers", "4",
        "--remove_unused_columns", "False"
    ]
    
    print(f"🏃 Training command:")
    print(f"   {' '.join(cmd)}")
    print()
    
    # Run training
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("🎉 Thai training completed successfully!")
        print(f"📁 Model saved to: {output_dir}")
        return True
    else:
        print(f"❌ Training failed with exit code: {result.returncode}")
        return False

def test_thai_tts(model_path: str, tokenizer_path: str, test_text: str = "สวัสดีครับ วันนี้อากาศดีมาก"):
    """Test the trained model with Thai text."""
    print("🧪 Testing Thai TTS with trained model...")
    
    # Create a temporary text file
    with open("temp_thai_test.txt", 'w', encoding='utf-8') as f:
        f.write(test_text)
    
    # Use the generation script
    cmd = [
        "python", "examples/generation.py",
        "--model_path", model_path,
        "--audio_tokenizer", tokenizer_path,
        "--transcript", "temp_thai_test.txt",
        "--out_path", "thai_tts_test.wav",
        "--temperature", "1.0",
        "--seed", "12345"
    ]
    
    print(f"🎵 Generating Thai speech: {test_text}")
    result = subprocess.run(cmd)
    
    # Cleanup
    if os.path.exists("temp_thai_test.txt"):
        os.remove("temp_thai_test.txt")
    
    if result.returncode == 0 and os.path.exists("thai_tts_test.wav"):
        print("✅ Thai TTS generation successful!")
        print("📁 Output: thai_tts_test.wav")
        return True
    else:
        print("❌ Thai TTS generation failed")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Train Higgs Audio with Thai language data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick training with 500 samples
  python train_thai_higgs.py --max_samples 500 --epochs 1

  # Full training with 5000 samples
  python train_thai_higgs.py --max_samples 5000 --epochs 3 --batch_size 2

  # Test only (if you have trained model)
  python train_thai_higgs.py --test_only --model_path ./thai_model --tokenizer_path ./tokenizer
        """
    )
    
    parser.add_argument("--max_samples", type=int, default=1000,
                       help="Maximum number of Thai samples to prepare for training")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./thai_higgs_model",
                       help="Directory to save the trained model")
    parser.add_argument("--data_dir", type=str, default="./thai_training_data",
                       help="Directory for prepared training data")
    parser.add_argument("--test_only", action="store_true",
                       help="Only test existing trained model")
    parser.add_argument("--model_path", type=str,
                       help="Path to existing model (for test only)")
    parser.add_argument("--tokenizer_path", type=str,
                       help="Path to tokenizer (for test only)")
    
    args = parser.parse_args()
    
    print("🇹🇭 Thai Language Training Pipeline for Higgs Audio")
    print("=" * 70)
    
    if args.test_only:
        if not args.model_path or not args.tokenizer_path:
            print("❌ For test only mode, provide --model_path and --tokenizer_path")
            return
        
        test_thai_tts(args.model_path, args.tokenizer_path)
        return
    
    # Step 1: Download base model
    print("Step 1: Download base model")
    base_model_path, base_tokenizer_path = download_base_model()
    if not base_model_path:
        print("❌ Failed to download base model")
        return
    
    # Step 2: Prepare Thai training data
    print(f"\nStep 2: Prepare Thai training data")
    if not prepare_thai_training_data(args.data_dir, args.max_samples):
        print("❌ Failed to prepare training data")
        return
    
    # Step 3: Convert to training format
    print(f"\nStep 3: Convert to training format")
    training_jsonl = f"{args.data_dir}/training_data.jsonl"
    if not convert_to_training_format(args.data_dir, training_jsonl):
        print("❌ Failed to convert data to training format")
        return
    
    # Step 4: Run training
    print(f"\nStep 4: Train model with Thai data")
    if not run_thai_training(
        base_model_path,
        base_tokenizer_path,
        training_jsonl,
        args.output_dir,
        args.epochs,
        args.batch_size,
        args.learning_rate
    ):
        print("❌ Training failed")
        return
    
    # Step 5: Test trained model
    print(f"\nStep 5: Test Thai TTS")
    test_thai_tts(args.output_dir, base_tokenizer_path)
    
    print("\n🎉 Thai language training pipeline completed!")
    print(f"📁 Your Thai-enabled Higgs Audio model is saved in: {args.output_dir}")
    print("\n🚀 You can now use it for Thai TTS:")
    print(f"python examples/generation.py --model_path {args.output_dir} --audio_tokenizer {base_tokenizer_path} --transcript your_thai_text.txt")

if __name__ == "__main__":
    main()
