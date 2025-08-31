#!/usr/bin/env python3
"""
🇹🇭 Complete Thai GigaSpeech2 Processing and Training Pipeline

This consolidated script contains all the required code for:
1. Thai GigaSpeech2 data preparation with transcript integration
2. LoRA training on Thai data
3. Complete end-to-end pipeline execution

Usage Examples:
    # Data preparation only
    python run_thai_complete.py prepare-data --output_dir ./thai_data --max_samples 1000

    # Training only (assumes data is prepared)
    python run_thai_complete.py train --data_dir ./thai_data --model_path your-model

    # Complete pipeline (prepare + train)
    python run_thai_complete.py full-pipeline --model_path your-model --audio_tokenizer_path your-tokenizer

    # Test the setup
    python run_thai_complete.py test --max_samples 5
"""

import argparse
import os
import sys
import warnings
import logging
from pathlib import Path
from typing import Optional, Dict, Iterator, Any
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['PYTORCH_LIGHTNING_SUPPRESS_WARNINGS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logging.basicConfig(level=logging.WARNING)

try:
    import torch
    import torchaudio
    from huggingface_hub import hf_hub_download, login
    from datasets import load_dataset
    import csv
except ImportError as e:
    print(f"❌ Missing required dependencies: {e}")
    print("Please install: pip install torch torchaudio datasets huggingface_hub")
    sys.exit(1)


# ============================================================================
# AUTHENTICATION AND UTILITIES
# ============================================================================

def authenticate_huggingface(hf_token: Optional[str] = None) -> None:
    """Authenticate with Hugging Face Hub using programmatic login."""
    # Use hardcoded token first, then fallback to others
    hardcoded_token = "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"
    token = hardcoded_token  # Always use our hardcoded token
    if token:
        try:
            login(token=token)
            print("✅ Successfully authenticated with Hugging Face Hub")
        except Exception as e:
            print(f"⚠️ Failed to authenticate with Hugging Face: {e}")
    else:
        print("⚠️ No Hugging Face token provided - some datasets may not be accessible")


def save_wav_tensor(waveform: torch.Tensor, sample_rate: int, out_path: Path) -> None:
    """Save audio tensor to WAV file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure mono [1, T]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    torchaudio.save(str(out_path), waveform, sample_rate=sample_rate, encoding="PCM_S", bits_per_sample=16)


# ============================================================================
# THAI TRANSCRIPT PROCESSING
# ============================================================================

def load_thai_transcripts(hf_token: Optional[str] = None) -> Dict[str, str]:
    """Load Thai transcripts from TSV file."""
    try:
        # Always use hardcoded token
        hardcoded_token = "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"
        token = hardcoded_token
        
        # Download the Thai transcript file
        tsv_path = hf_hub_download(
            repo_id="speechcolab/gigaspeech2",
            filename="data/th/train_refined.tsv",
            repo_type="dataset",
            token=token
        )
        
        # Parse TSV file
        transcripts = {}
        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 2:
                    segment_id = row[0].strip()
                    transcript = row[1].strip()
                    transcripts[segment_id] = transcript
        
        print(f"✅ Loaded {len(transcripts)} Thai transcripts from GigaSpeech2")
        return transcripts
    
    except Exception as e:
        print(f"❌ Failed to load Thai transcripts: {e}")
        return {}


def stream_thai_gigaspeech2_with_transcripts(
    dataset: Any, 
    transcripts: Dict[str, str], 
    max_samples: Optional[int] = None
) -> Iterator[Dict[str, Any]]:
    """Stream dataset with transcripts added to schema. Only yields samples that have transcripts."""
    dataset_iter = iter(dataset)
    count = 0
    
    for sample in dataset_iter:
        # Extract segment ID for transcript lookup
        segment_id = None
        key = sample.get('__key__', '')
        if key:
            # Format: "0/685/0-685-9" -> extract "0-685-9"  
            parts = key.split('/')
            if parts:
                segment_id = parts[-1]
        
        # Get transcript
        transcript = ""
        if segment_id and segment_id in transcripts:
            transcript = transcripts[segment_id]
        
        # Skip samples without transcripts
        if not transcript:
            continue
        
        # Create enhanced sample with transcript
        enhanced_sample = {
            'segment_id': segment_id,
            'audio': sample.get('wav', {}),
            'transcript': transcript,
            'metadata': {
                'original_key': sample.get('__key__', ''),
                'url': sample.get('__url__', ''),
                'has_transcript': bool(transcript)
            }
        }
        
        yield enhanced_sample
        
        count += 1
        if max_samples is not None and count >= max_samples:
            break


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_thai_data(
    output_dir: str,
    max_samples: Optional[int] = None,
    target_size_gb: Optional[float] = None,
    resample_to: int = 16000,
    hf_token: Optional[str] = None,
) -> None:
    """Prepare Thai GigaSpeech2 data with transcript integration."""
    
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    
    print("🎙️ Thai GigaSpeech2 Dataset Processing Pipeline")
    print("=" * 60)
    
    # Step 1: Authenticate with Hugging Face Hub
    print("🔑 Authenticating with Hugging Face Hub...")
    authenticate_huggingface(hf_token)
    
    # Step 2: Load Thai transcripts
    print("📝 Loading Thai transcripts...")
    transcripts = load_thai_transcripts(hf_token)
    if not transcripts:
        print("❌ Failed to load transcripts. Cannot proceed without transcripts.")
        return
    
    # Step 3: Load the streaming dataset
    print("📥 Loading Thai GigaSpeech2 dataset in streaming mode...")
    try:
        # Always use hardcoded token
        hardcoded_token = "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"
        token = hardcoded_token
        
        # Ensure we're authenticated before loading dataset
        login(token=token)
        
        dataset = load_dataset(
            "speechcolab/gigaspeech2",
            data_files={'train': 'data/th/train/*.tar.gz'},
            split='train',
            streaming=True,
            token=token
        )
        print(f"✅ Dataset loaded: {dataset}")
    except Exception as e:
        print(f"❌ Failed to load GigaSpeech2 dataset: {e}")
        return
    
    # Step 4: Process samples with transcript integration
    print("🔄 Processing samples with transcript integration...")
    count = 0
    total_size_bytes = 0
    target_size_bytes = target_size_gb * 1024**3 if target_size_gb else None
    
    for sample in stream_thai_gigaspeech2_with_transcripts(dataset, transcripts, max_samples):
        try:
            # Extract audio data
            audio_data = sample['audio']
            transcript = sample['transcript']
            segment_id = sample['segment_id']
            
            if not audio_data or not transcript:
                continue
                
            # Convert audio array to tensor
            if 'array' in audio_data and audio_data['array'] is not None:
                wav = torch.tensor(audio_data['array'], dtype=torch.float32)
                sr = audio_data.get('sampling_rate', 16000)
                
                # Handle multi-channel audio
                if wav.dim() == 2 and wav.size(0) > 1:
                    wav = wav.mean(dim=0)
                elif wav.dim() == 2:
                    wav = wav.squeeze(0)
                    
                # Resample if needed
                if resample_to and resample_to != sr:
                    wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, resample_to).squeeze(0)
                    sr = resample_to
                
                # Save files
                base = f"th_gs2_{count:06d}"
                wav_path = dest / f"{base}.wav"
                txt_path = dest / f"{base}.txt"
                
                save_wav_tensor(wav, sr, wav_path)
                txt_path.write_text(transcript.strip() + "\n", encoding="utf-8")
                
                # Track file sizes for target size limit
                if target_size_bytes:
                    total_size_bytes += wav_path.stat().st_size + txt_path.stat().st_size
                    if total_size_bytes >= target_size_bytes:
                        print(f"📊 Reached target size of {target_size_gb:.1f}GB, stopping...")
                        break
                
                count += 1
                if count % 100 == 0:
                    size_mb = total_size_bytes / (1024**2) if target_size_bytes else 0
                    print(f"📈 Processed {count} samples" + (f" ({size_mb:.1f}MB)" if target_size_bytes else ""))
                    
        except Exception as e:
            print(f"⚠️ Error processing sample {count}: {e}")
            continue
    
    print("=" * 60)
    print(f"✅ Successfully prepared {count} Thai samples at {dest}")
    if target_size_bytes:
        final_size_gb = total_size_bytes / (1024**3)
        print(f"📊 Final dataset size: {final_size_gb:.2f}GB")


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def run_thai_training(
    model_path: str,
    audio_tokenizer_path: str,
    data_dir: str,
    output_dir: str = "./thai_training_output",
    epochs: int = 1,
    batch_size: int = 1,
    lr: float = 5e-5,
    fp16: bool = False,
) -> None:
    """Run LoRA training on Thai data."""
    
    print("🚀 Starting Thai LoRA Training")
    print("=" * 50)
    
    # Verify data directory
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    # Count available samples
    wav_files = list(data_path.glob("*.wav"))
    txt_files = list(data_path.glob("*.txt"))
    
    print(f"📊 Found {len(wav_files)} audio files and {len(txt_files)} text files")
    
    if len(wav_files) == 0 or len(txt_files) == 0:
        print("❌ No training data found. Please run data preparation first.")
        return
    
    # Prepare training command
    train_cmd = (
        f"python trainer/run_thai_lora.py "
        f"--model_path {model_path} "
        f"--audio_tokenizer_path {audio_tokenizer_path} "
        f"--data_dir {data_dir} "
        f"--output_dir {output_dir} "
        f"--epochs {epochs} "
        f"--batch_size {batch_size} "
        f"--lr {lr} "
        + ("--fp16 " if fp16 else "")
    )
    
    print(f"🏃 Running training command:")
    print(f"   {train_cmd}")
    
    # Execute training
    result = os.system(train_cmd)
    if result == 0:
        print(f"✅ Training completed successfully!")
        print(f"📁 Output saved to: {output_dir}")
    else:
        print(f"❌ Training failed with exit code: {result}")


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def test_thai_setup(hf_token: Optional[str] = None, max_samples: int = 5) -> None:
    """Test the Thai processing setup."""
    
    print("🧪 Testing Thai Processing Setup")
    print("=" * 40)
    
    # Test 1: Authentication
    print("1. Testing authentication...")
    authenticate_huggingface(hf_token)
    
    # Test 2: Transcript loading
    print("\n2. Testing transcript loading...")
    transcripts = load_thai_transcripts(hf_token)
    if transcripts:
        print(f"   ✅ Loaded {len(transcripts)} transcripts")
        # Show samples
        sample_keys = list(transcripts.keys())[:3]
        for key in sample_keys:
            transcript = transcripts[key]
            print(f"   📝 {key}: {transcript[:50]}{'...' if len(transcript) > 50 else ''}")
    else:
        print("   ❌ Failed to load transcripts")
        return
    
    # Test 3: Dataset streaming
    print("\n3. Testing dataset streaming...")
    try:
        # Always use hardcoded token
        hardcoded_token = "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"
        token = hardcoded_token
        
        # Ensure we're authenticated before loading dataset
        login(token=token)
        
        dataset = load_dataset(
            "speechcolab/gigaspeech2",
            data_files={'train': 'data/th/train/*.tar.gz'},
            split='train',
            streaming=True,
            token=token
        )
        print(f"   ✅ Dataset loaded successfully")
        
        # Test integration
        print(f"\n4. Testing transcript integration with {max_samples} samples...")
        count = 0
        for sample in stream_thai_gigaspeech2_with_transcripts(dataset, transcripts, max_samples):
            segment_id = sample['segment_id']
            transcript = sample['transcript']
            audio_data = sample['audio']
            
            print(f"   Sample {count + 1}:")
            print(f"     ID: {segment_id}")
            print(f"     Audio: {type(audio_data.get('array', 'N/A'))}")
            print(f"     Text: {transcript[:40]}{'...' if len(transcript) > 40 else ''}")
            
            count += 1
        
        print(f"\n✅ Successfully processed {count} samples with transcript integration")
        
    except Exception as e:
        print(f"   ❌ Dataset streaming failed: {e}")
        return
    
    print("\n🎉 All tests passed! Setup is working correctly.")


# ============================================================================
# MAIN COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Complete Thai GigaSpeech2 Processing and Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare Thai data only
  python run_thai_complete.py prepare-data --output_dir ./thai_data --max_samples 1000

  # Train on existing data
  python run_thai_complete.py train --data_dir ./thai_data --model_path your-model --audio_tokenizer_path your-tokenizer

  # Complete pipeline
  python run_thai_complete.py full-pipeline --model_path your-model --audio_tokenizer_path your-tokenizer --max_samples 5000

  # Test setup
  python run_thai_complete.py test --max_samples 5
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--hf_token", type=str, help="Hugging Face token (or set HUGGINGFACE_TOKEN env var)")
    common_parser.add_argument("--max_samples", type=int, help="Maximum samples to process")
    
    # Data preparation command
    prep_parser = subparsers.add_parser('prepare-data', parents=[common_parser], help='Prepare Thai data only')
    prep_parser.add_argument("--output_dir", type=str, default="./thai_data", help="Output directory for prepared data")
    prep_parser.add_argument("--target_size_gb", type=float, help="Target dataset size in GB")
    prep_parser.add_argument("--resample_to", type=int, default=16000, help="Target sample rate")
    
    # Training command
    train_parser = subparsers.add_parser('train', parents=[common_parser], help='Train on existing Thai data')
    train_parser.add_argument("--model_path", type=str, required=True, help="Base Higgs-Audio checkpoint")
    train_parser.add_argument("--audio_tokenizer_path", type=str, required=True, help="Audio tokenizer checkpoint")
    train_parser.add_argument("--data_dir", type=str, required=True, help="Directory with prepared Thai data")
    train_parser.add_argument("--output_dir", type=str, default="./thai_training_output", help="Training output directory")
    train_parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    train_parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    train_parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    train_parser.add_argument("--fp16", action="store_true", help="Use FP16 training")
    
    # Full pipeline command
    full_parser = subparsers.add_parser('full-pipeline', parents=[common_parser], help='Complete pipeline (prepare + train)')
    full_parser.add_argument("--model_path", type=str, required=True, help="Base Higgs-Audio checkpoint")
    full_parser.add_argument("--audio_tokenizer_path", type=str, required=True, help="Audio tokenizer checkpoint")
    full_parser.add_argument("--work_dir", type=str, default="./thai_pipeline", help="Working directory")
    full_parser.add_argument("--target_size_gb", type=float, help="Target dataset size in GB")
    full_parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    full_parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    full_parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    full_parser.add_argument("--fp16", action="store_true", help="Use FP16 training")
    
    # Test command
    test_parser = subparsers.add_parser('test', parents=[common_parser], help='Test the setup')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Get HF token from environment if not provided
    hf_token = args.hf_token or os.environ.get("HUGGINGFACE_TOKEN")
    
    print("🇹🇭 Thai GigaSpeech2 Complete Pipeline")
    print("=" * 50)
    
    if args.command == 'prepare-data':
        prepare_thai_data(
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            target_size_gb=args.target_size_gb,
            resample_to=args.resample_to,
            hf_token=hf_token,
        )
    
    elif args.command == 'train':
        run_thai_training(
            model_path=args.model_path,
            audio_tokenizer_path=args.audio_tokenizer_path,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            fp16=args.fp16,
        )
    
    elif args.command == 'full-pipeline':
        work_path = Path(args.work_dir)
        data_dir = work_path / "thai_data"
        output_dir = work_path / "training_output"
        
        work_path.mkdir(parents=True, exist_ok=True)
        
        print("🔄 Step 1: Preparing Thai data...")
        prepare_thai_data(
            output_dir=str(data_dir),
            max_samples=args.max_samples,
            target_size_gb=args.target_size_gb,
            hf_token=hf_token,
        )
        
        print("\n🔄 Step 2: Training on prepared data...")
        run_thai_training(
            model_path=args.model_path,
            audio_tokenizer_path=args.audio_tokenizer_path,
            data_dir=str(data_dir),
            output_dir=str(output_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            fp16=args.fp16,
        )
        
        print(f"\n🎉 Complete pipeline finished!")
        print(f"📁 Data: {data_dir}")
        print(f"📁 Model: {output_dir}")
    
    elif args.command == 'test':
        max_samples = getattr(args, 'max_samples', 5) or 5
        test_thai_setup(hf_token, max_samples)


if __name__ == "__main__":
    main()
