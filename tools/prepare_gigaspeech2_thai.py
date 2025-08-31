import argparse
import os
from pathlib import Path
from typing import Optional, Dict, Iterator, Any
import warnings
import logging

import torchaudio
import torch
import csv
from typing import List
from huggingface_hub import hf_hub_download, login
from datasets import load_dataset

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['PYTORCH_LIGHTNING_SUPPRESS_WARNINGS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logging.basicConfig(level=logging.WARNING)


def authenticate_huggingface(hf_token: Optional[str] = None) -> None:
    """Authenticate with Hugging Face Hub using programmatic login."""
    token = hf_token or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        try:
            login(token=token)
            print("✅ Successfully authenticated with Hugging Face Hub")
        except Exception as e:
            print(f"⚠️ Failed to authenticate with Hugging Face: {e}")
    else:
        print("⚠️ No Hugging Face token provided - some datasets may not be accessible")


def load_thai_transcripts(hf_token: Optional[str] = None) -> Dict[str, str]:
    """Load Thai transcripts from TSV file, following the notebook approach."""
    try:
        # Download the Thai transcript file
        tsv_path = hf_hub_download(
            repo_id="speechcolab/gigaspeech2",
            filename="data/th/train_refined.tsv",
            repo_type="dataset",
            token=hf_token
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
    """
    Stream dataset with transcripts added to schema, following the notebook approach.
    Only yields samples that have transcripts.
    """
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


def save_wav_tensor(waveform: torch.Tensor, sample_rate: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure mono [1, T]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    torchaudio.save(str(out_path), waveform, sample_rate=sample_rate, encoding="PCM_S", bits_per_sample=16)


def prepare_gigaspeech2_thai_notebook_style(
    output_dir: str,
    split: str = "train_refined",
    max_samples: Optional[int] = None,
    resample_to: Optional[int] = None,
    hf_token: Optional[str] = None,
    target_size_gb: Optional[float] = None,
) -> None:
    """
    Process Thai GigaSpeech2 dataset using the notebook approach with streaming and transcript integration.
    
    This function follows the approach demonstrated in the dataset.ipynb notebook:
    1. Authenticate with Hugging Face Hub
    2. Load Thai transcripts from TSV file
    3. Stream the audio dataset with transcript matching
    4. Save matched audio-transcript pairs in Higgs-Audio format
    
    Produces:
      output_dir/
        th_gs2_000000.wav
        th_gs2_000000.txt
        ...
    """
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
        dataset = load_dataset(
            "speechcolab/gigaspeech2",
            data_files={'train': 'data/th/train/*.tar.gz'},
            split='train',
            streaming=True,
            token=hf_token
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


def prepare_gigaspeech2_thai(
    output_dir: str,
    split: str = "train_refined",
    max_samples: Optional[int] = None,
    resample_to: Optional[int] = None,
    hf_token: Optional[str] = None,
    streaming: bool = False,
    target_size_gb: Optional[float] = None,
) -> None:
    """
    Download and convert the Thai portion of GigaSpeech2 into Higgs-Audio training format.
    
    This function now uses the notebook-style approach as the primary method,
    with fallbacks to Common Voice if needed.
    """
    # Try the notebook-style approach first (preferred for GigaSpeech2)
    try:
        prepare_gigaspeech2_thai_notebook_style(
            output_dir=output_dir,
            split=split,
            max_samples=max_samples,
            resample_to=resample_to,
            hf_token=hf_token,
            target_size_gb=target_size_gb,
        )
        return
    except Exception as e:
        print(f"⚠️ Notebook-style processing failed: {e}")
        print("🔄 Falling back to Common Voice approach...")
    
    # Fallback to original approach with Common Voice
    from datasets import load_dataset, Audio

    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)

    # Prefer Common Voice Thai (small subset) to avoid heavy GigaSpeech2 and torchcodec issues
    last_err = None
    ds = None
    for cv_repo in [
        "mozilla-foundation/common_voice_17_0",
        "mozilla-foundation/common_voice_16_1",
        "mozilla-foundation/common_voice_15_0",
    ]:
        try:
            split_expr = "train" if max_samples is None else f"train[:{max_samples}]"
            ds = load_dataset(cv_repo, "th", split=split_expr, streaming=False)
            print(f"Loaded dataset: {cv_repo} (th), split={split_expr}")
            break
        except Exception as e:
            last_err = e
            ds = None
    
    if ds is None:
        raise last_err if last_err else RuntimeError("Failed to load Thai dataset from Common Voice.")

    # Process Common Voice data
    count = 0
    for example in ds:
        text = example.get("text") or example.get("transcript") or example.get("sentence")
        audio = example["audio"]
        if text is None or audio is None:
            continue

        # Handle audio data
        wav = None
        sr = 16000
        if isinstance(audio, dict) and "array" in audio:
            wav = torch.tensor(audio["array"], dtype=torch.float32)
            sr = int(audio.get("sampling_rate", sr))
            
        if wav is None:
            continue

        if resample_to and resample_to != sr:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, resample_to).squeeze(0)
            sr = resample_to

        base = f"th_cv_{count:06d}"
        wav_path = dest / f"{base}.wav"
        txt_path = dest / f"{base}.txt"

        save_wav_tensor(wav, sr, wav_path)
        txt_path.write_text(text.strip() + "\n", encoding="utf-8")

        count += 1
        if max_samples is not None and count >= max_samples:
            break

    print(f"Prepared {count} Thai samples at {dest} (Common Voice fallback)")


def prepare_common_voice_thai_via_hub(
    output_dir: str,
    max_samples: Optional[int] = None,
    resample_to: Optional[int] = 16000,
    hf_token: Optional[str] = None,
) -> int:
    """Download Common Voice Thai clips directly from the Hub without using datasets' Audio.

    This avoids torchcodec requirements on Windows by fetching TSV metadata and MP3 clips
    and decoding locally with torchaudio.
    """
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)

    repo_id = "mozilla-foundation/common_voice_17_0"
    # Download train.tsv
    tsv_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename="th/train.tsv",
        token=hf_token,
    )

    # Read rows and pick first N with sentence and path
    rows: List[dict] = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sentence = row.get("sentence") or row.get("text")
            clip_rel = row.get("path") or row.get("filename")
            if sentence and clip_rel:
                rows.append({"sentence": sentence.strip(), "clip_rel": clip_rel.strip()})
            if max_samples is not None and len(rows) >= max_samples:
                break

    count = 0
    for i, item in enumerate(rows):
        clip_rel = item["clip_rel"]
        sentence = item["sentence"]
        try:
            clip_local = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=f"th/clips/{clip_rel}",
                token=hf_token,
            )
        except Exception:
            continue

        try:
            waveform, sr = torchaudio.load(clip_local)
            if waveform.dim() == 2 and waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.squeeze(0)
            if resample_to and int(sr) != int(resample_to):
                waveform = torchaudio.functional.resample(waveform.unsqueeze(0), int(sr), int(resample_to)).squeeze(0)
                sr = int(resample_to)
            base = f"th_cv_{i:06d}"
            wav_path = dest / f"{base}.wav"
            txt_path = dest / f"{base}.txt"
            save_wav_tensor(waveform, int(sr), wav_path)
            txt_path.write_text(sentence + "\n", encoding="utf-8")
            count += 1
        except Exception:
            continue

    return count


def main():
    parser = argparse.ArgumentParser(description="Prepare GigaSpeech2 Thai for Higgs-Audio training using notebook approach")
    parser.add_argument("--output_dir", type=str, required=True, help="Destination directory for prepared data")
    parser.add_argument("--split", type=str, default="train_refined", choices=["train", "train_refined", "train_raw", "dev", "test"], help="Dataset split")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples to convert (streamed)")
    parser.add_argument("--target_size_gb", type=float, default=None, help="Approximate target size in GB for prepared data (stop when reached)")
    parser.add_argument("--resample_to", type=int, default=16000, help="Target sample rate for saved wav files")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token (or set HUGGINGFACE_TOKEN env var)")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode (may not be supported by all configs)")

    args = parser.parse_args()
    
    print("🎙️ Thai GigaSpeech2 Processing - Notebook Style")
    print("=" * 50)
    
    # Use the enhanced notebook-style approach with fallbacks
    prepare_gigaspeech2_thai(
        output_dir=args.output_dir,
        split=args.split,
        max_samples=args.max_samples,
        resample_to=args.resample_to,
        hf_token=args.hf_token,
        streaming=args.streaming,
        target_size_gb=args.target_size_gb,
    )


if __name__ == "__main__":
    main()

