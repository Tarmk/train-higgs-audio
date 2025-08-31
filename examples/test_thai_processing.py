#!/usr/bin/env python3
"""
Test script for Thai GigaSpeech2 processing using the notebook approach.

This script demonstrates the enhanced Thai language processing pipeline
that integrates streaming audio data with transcripts from GigaSpeech2.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.prepare_gigaspeech2_thai import (
    authenticate_huggingface,
    load_thai_transcripts,
    stream_thai_gigaspeech2_with_transcripts,
    prepare_gigaspeech2_thai_notebook_style,
)
from datasets import load_dataset


def test_authentication(hf_token: str = None):
    """Test Hugging Face authentication."""
    print("🔑 Testing Hugging Face Authentication")
    print("-" * 40)
    authenticate_huggingface(hf_token)
    print()


def test_transcript_loading(hf_token: str = None):
    """Test loading Thai transcripts."""
    print("📝 Testing Thai Transcript Loading")
    print("-" * 40)
    transcripts = load_thai_transcripts(hf_token)
    
    if transcripts:
        print(f"✅ Loaded {len(transcripts)} transcripts")
        # Show a sample of transcripts
        sample_keys = list(transcripts.keys())[:3]
        for key in sample_keys:
            transcript = transcripts[key]
            print(f"  {key}: {transcript[:50]}{'...' if len(transcript) > 50 else ''}")
    else:
        print("❌ Failed to load transcripts")
    print()
    return transcripts


def test_streaming_dataset(hf_token: str = None, max_samples: int = 5):
    """Test streaming dataset loading."""
    print("📥 Testing Dataset Streaming")
    print("-" * 40)
    
    try:
        dataset = load_dataset(
            "speechcolab/gigaspeech2",
            data_files={'train': 'data/th/train/*.tar.gz'},
            split='train',
            streaming=True,
            token=hf_token
        )
        print(f"✅ Dataset loaded: {dataset}")
        
        # Test basic iteration
        print("Testing basic dataset iteration...")
        count = 0
        for sample in dataset:
            key = sample.get('__key__', 'Unknown')
            audio_shape = sample.get('wav', {}).get('array', [])
            audio_shape = getattr(audio_shape, 'shape', 'No shape') if hasattr(audio_shape, 'shape') else len(audio_shape) if audio_shape else 0
            print(f"  Sample {count + 1}: Key={key}, Audio shape={audio_shape}")
            count += 1
            if count >= max_samples:
                break
                
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return None
        
    print()
    return dataset


def test_transcript_integration(hf_token: str = None, max_samples: int = 5):
    """Test the complete transcript integration pipeline."""
    print("🔄 Testing Transcript Integration")
    print("-" * 40)
    
    # Load transcripts
    transcripts = load_thai_transcripts(hf_token)
    if not transcripts:
        print("❌ Cannot test integration without transcripts")
        return
    
    # Load dataset
    try:
        dataset = load_dataset(
            "speechcolab/gigaspeech2",
            data_files={'train': 'data/th/train/*.tar.gz'},
            split='train',
            streaming=True,
            token=hf_token
        )
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    # Test integrated streaming
    print(f"Testing integrated streaming with {max_samples} samples...")
    count = 0
    for sample in stream_thai_gigaspeech2_with_transcripts(dataset, transcripts, max_samples):
        segment_id = sample['segment_id']
        transcript = sample['transcript']
        audio_data = sample['audio']
        audio_shape = audio_data.get('array', [])
        audio_shape = getattr(audio_shape, 'shape', 'No shape') if hasattr(audio_shape, 'shape') else len(audio_shape) if audio_shape else 0
        
        print(f"  Sample {count + 1}:")
        print(f"    Segment ID: {segment_id}")
        print(f"    Audio shape: {audio_shape}")
        print(f"    Transcript: {transcript[:60]}{'...' if len(transcript) > 60 else ''}")
        print()
        
        count += 1
        
    print(f"✅ Successfully processed {count} samples with transcript integration")
    print()


def test_full_processing_pipeline(output_dir: str, hf_token: str = None, max_samples: int = 10):
    """Test the complete processing pipeline."""
    print("🎙️ Testing Full Processing Pipeline")
    print("-" * 40)
    
    try:
        prepare_gigaspeech2_thai_notebook_style(
            output_dir=output_dir,
            max_samples=max_samples,
            hf_token=hf_token,
        )
        
        # Verify output files
        output_path = Path(output_dir)
        wav_files = list(output_path.glob("*.wav"))
        txt_files = list(output_path.glob("*.txt"))
        
        print(f"✅ Processing completed!")
        print(f"   Generated {len(wav_files)} audio files")
        print(f"   Generated {len(txt_files)} transcript files")
        
        # Show sample files
        if wav_files and txt_files:
            sample_wav = wav_files[0]
            sample_txt = txt_files[0]
            print(f"   Sample files: {sample_wav.name}, {sample_txt.name}")
            
            # Show transcript content
            transcript_content = sample_txt.read_text(encoding='utf-8').strip()
            print(f"   Sample transcript: {transcript_content}")
            
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        
    print()


def main():
    parser = argparse.ArgumentParser(description="Test Thai GigaSpeech2 processing pipeline")
    parser.add_argument("--hf_token", type=str, help="Hugging Face token")
    parser.add_argument("--output_dir", type=str, default="./test_thai_output", help="Output directory for test files")
    parser.add_argument("--max_samples", type=int, default=5, help="Maximum samples to test")
    parser.add_argument("--test", type=str, choices=["auth", "transcripts", "streaming", "integration", "full", "all"], 
                       default="all", help="Which test to run")
    
    args = parser.parse_args()
    
    # Get token from environment if not provided
    hf_token = args.hf_token or os.environ.get("HUGGINGFACE_TOKEN")
    
    print("🧪 Thai GigaSpeech2 Processing Test Suite")
    print("=" * 50)
    print()
    
    if args.test in ["auth", "all"]:
        test_authentication(hf_token)
    
    if args.test in ["transcripts", "all"]:
        test_transcript_loading(hf_token)
    
    if args.test in ["streaming", "all"]:
        test_streaming_dataset(hf_token, args.max_samples)
    
    if args.test in ["integration", "all"]:
        test_transcript_integration(hf_token, args.max_samples)
    
    if args.test in ["full", "all"]:
        test_full_processing_pipeline(args.output_dir, hf_token, args.max_samples)
    
    print("🎉 Test suite completed!")


if __name__ == "__main__":
    main()
