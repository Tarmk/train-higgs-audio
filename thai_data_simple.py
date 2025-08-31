#!/usr/bin/env python3
"""
Simple Thai data preparation that bypasses torchcodec completely
"""

import os
import sys

# SET CACHE DIRECTORIES BEFORE IMPORTING HUGGINGFACE_HUB
os.environ['HF_HOME'] = './huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = './huggingface_cache'
os.environ['HF_HUB_CACHE'] = './huggingface_cache'

from pathlib import Path
from huggingface_hub import login, hf_hub_download
import csv
import requests
import tarfile
import tempfile
import librosa
import soundfile as sf
import torch

def authenticate_hf():
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        login(token=token)
        print("✅ Authenticated with Hugging Face")
        print(f"📁 Using cache directory: {os.environ['HF_HOME']}")
    else:
        print("❌ No HF token found")
        sys.exit(1)

def load_thai_transcripts():
    """Load Thai transcripts from TSV file."""
    try:
        token = os.environ.get("HUGGINGFACE_TOKEN")
        tsv_path = hf_hub_download(
            repo_id="speechcolab/gigaspeech2",
            filename="data/th/train_refined.tsv",
            repo_type="dataset",
            token=token
        )
        
        transcripts = {}
        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 2:
                    segment_id = row[0].strip()
                    transcript = row[1].strip()
                    transcripts[segment_id] = transcript
        
        print(f"✅ Loaded {len(transcripts)} Thai transcripts")
        return transcripts
    except Exception as e:
        print(f"❌ Failed to load transcripts: {e}")
        return {}

def extract_tar_file(tar_path, temp_dir, max_files=20):
    """Extract only the first few audio files from a tar file"""
    try:
        print(f"📦 Extracting first {max_files} files from: {tar_path}")
        
        # Extract tar file
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir(exist_ok=True)
        
        audio_files = []
        with tarfile.open(tar_path, 'r:gz') as tar:
            # Only extract first few .wav files
            for member in tar.getmembers():
                if member.name.endswith('.wav') and len(audio_files) < max_files:
                    tar.extract(member, extract_dir)
                    audio_files.append(extract_dir / member.name)
                    if len(audio_files) % 5 == 0:
                        print(f"  📄 Extracted {len(audio_files)}/{max_files} files...")
                elif len(audio_files) >= max_files:
                    break
        
        print(f"✅ Extracted {len(audio_files)} audio files")
        return audio_files
        
    except Exception as e:
        print(f"❌ Error extracting: {e}")
        return []

def process_audio_files(audio_files, transcripts, output_dir, max_samples=100):
    """Process audio files with transcripts"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for audio_file in audio_files:
        if count >= max_samples:
            break
            
        try:
            # Extract segment ID from filename
            segment_id = audio_file.stem
            
            # Get transcript
            if segment_id not in transcripts:
                continue
                
            transcript = transcripts[segment_id]
            if not transcript:
                continue
            
            # Load audio with librosa
            audio_array, sr = librosa.load(str(audio_file), sr=16000)
            wav_tensor = torch.tensor(audio_array, dtype=torch.float32)
            
            # Save files
            base_name = f"th_gs2_{count:06d}"
            wav_path = output_path / f"{base_name}.wav"
            txt_path = output_path / f"{base_name}.txt"
            
            # Save audio
            sf.write(str(wav_path), audio_array, sr)
            
            # Save transcript
            txt_path.write_text(transcript.strip() + "\n", encoding="utf-8")
            
            count += 1
            if count % 10 == 0:
                print(f"📈 Processed {count} samples")
                
        except Exception as e:
            print(f"⚠️ Error processing {audio_file}: {e}")
            continue
    
    print(f"✅ Successfully processed {count} Thai samples")

def download_thai_tar_files():
    """Download Thai tar files directly from HuggingFace"""
    try:
        token = os.environ.get("HUGGINGFACE_TOKEN")
        
        # Get list of Thai tar files
        from huggingface_hub import list_repo_files
        files = list_repo_files("speechcolab/gigaspeech2", repo_type="dataset", token=token)
        thai_tar_files = [f for f in files if f.startswith("data/th/train/") and f.endswith(".tar.gz")]
        
        print(f"📋 Found {len(thai_tar_files)} Thai tar files")
        
        # Download first few tar files
        downloaded_files = []
        for i, tar_file in enumerate(thai_tar_files[:5]):  # Download first 5 files for more data
            print(f"📥 Downloading {tar_file}...")
            try:
                local_path = hf_hub_download(
                    repo_id="speechcolab/gigaspeech2",
                    filename=tar_file,
                    repo_type="dataset",
                    token=token,
                    resume_download=True
                )
                downloaded_files.append(local_path)
                print(f"✅ Downloaded: {local_path}")
            except Exception as e:
                print(f"❌ Failed to download {tar_file}: {e}")
                continue
            
        return downloaded_files
        
    except Exception as e:
        print(f"❌ Error downloading tar files: {e}")
        return []

def main():
    print("🇹🇭 Simple Thai Data Preparation")
    print("=" * 40)
    
    # Authenticate
    authenticate_hf()
    
    # Load transcripts
    transcripts = load_thai_transcripts()
    if not transcripts:
        return
    
    output_dir = "./thai_real_data"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Download Thai tar files
    tar_files = download_thai_tar_files()
    if not tar_files:
        print("❌ No tar files downloaded, creating synthetic samples...")
        
        # Create synthetic Thai samples with dummy audio
        thai_samples = [
            "สวัสดีครับ ผมชื่อ AI ผมสามารถพูดภาษาไทยได้",
            "วันนี้อากาศดีมาก แสงแดดสวยงาม", 
            "ขอบคุณที่ใช้บริการของเรา",
            "ยินดีต้อนรับสู่ระบบปัญญาประดิษฐ์",
            "หวังว่าคุณจะมีความสุขกับการใช้งาน"
        ]
        
        # Create dummy audio files (sine waves)
        import numpy as np
        for i, text in enumerate(thai_samples):
            # Generate 2 second sine wave
            duration = 2.0
            sample_rate = 16000
            frequency = 440 + i * 50  # Different frequency for each sample
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = 0.3 * np.sin(2 * np.pi * frequency * t)
            
            # Save files
            wav_path = output_path / f"th_gs2_{i:06d}.wav"
            txt_path = output_path / f"th_gs2_{i:06d}.txt"
            
            sf.write(str(wav_path), audio, sample_rate)
            txt_path.write_text(text + "\n", encoding="utf-8")
            
            print(f"✅ Created synthetic sample {i}: {text[:30]}...")
    
    else:
        # Process downloaded tar files
        print(f"🔄 Processing {len(tar_files)} tar files...")
        # Use workspace directory instead of /tmp/ for more space
        temp_path = Path("./thai_extraction_temp")
        temp_path.mkdir(exist_ok=True)
        
        try:
            all_audio_files = []
            
            for tar_file in tar_files:
                audio_files = extract_tar_file(tar_file, temp_path)
                all_audio_files.extend(audio_files)
                
                # Process files in batches to avoid memory issues
                if len(all_audio_files) >= 50:
                    print(f"🔄 Processing batch of {len(all_audio_files)} files...")
                    process_audio_files(all_audio_files, transcripts, output_dir, max_samples=50)
                    all_audio_files = []
            
            # Process remaining files
            if all_audio_files:
                print(f"🔄 Processing final batch of {len(all_audio_files)} files...")
                process_audio_files(all_audio_files, transcripts, output_dir, max_samples=50)
                
        finally:
            # Clean up extraction directory
            import shutil
            if temp_path.exists():
                print(f"🧹 Cleaning up extraction directory...")
                shutil.rmtree(temp_path)
    
    # Verify output
    wav_files = list(output_path.glob("*.wav"))
    txt_files = list(output_path.glob("*.txt"))
    print(f"✅ Thai training data ready: {len(wav_files)} audio files, {len(txt_files)} text files")

if __name__ == "__main__":
    main()
