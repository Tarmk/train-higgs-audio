import argparse
import os
from pathlib import Path
from typing import Optional

import torchaudio
import torch
import csv
from typing import List
from huggingface_hub import hf_hub_download


def save_wav_tensor(waveform: torch.Tensor, sample_rate: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure mono [1, T]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    torchaudio.save(str(out_path), waveform, sample_rate=sample_rate, encoding="PCM_S", bits_per_sample=16)


def prepare_gigaspeech2_thai(
    output_dir: str,
    split: str = "train_refined",
    max_samples: Optional[int] = None,
    resample_to: Optional[int] = None,
    hf_token: Optional[str] = None,
    streaming: bool = False,
) -> None:
    """
    Download and convert the Thai portion of GigaSpeech2 into Higgs-Audio training format.

    Produces:
      output_dir/
        metadata.json              (optional; we rely on directory scan so it's omitted for simplicity)
        th_gs2_000000.wav
        th_gs2_000000.txt
        ...
    """
    from datasets import load_dataset, Audio

    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)

    # Stream to avoid downloading full dataset. Requires access accepted at
    # https://huggingface.co/datasets/speechcolab/gigaspeech2
    auth = hf_token if hf_token else os.environ.get("HUGGINGFACE_TOKEN", None)
    
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
    
    # If CV fails entirely, try GigaSpeech2 as a fallback
    if ds is None:
        def _try_load(split_opt: str, data_dir_opt: Optional[str]):
            return load_dataset(
                "speechcolab/gigaspeech2",
                split=split_opt,
                streaming=streaming,
                data_dir=data_dir_opt,
            )
        for data_dir_opt in ["data/th", "th", None]:
            for split_opt in [split, "train"]:
                try:
                    ds = _try_load(split_opt, data_dir_opt)
                    split = split_opt
                    break
                except Exception as e:
                    last_err = e
                    ds = None
            if ds is not None:
                break
        if ds is None:
            raise last_err if last_err else RuntimeError("Failed to load Thai dataset from Common Voice or GigaSpeech2.")

    # Ensure audio is handled; prefer not to decode to avoid torchcodec requirement
    ds = ds.cast_column("audio", Audio(sampling_rate=None, decode=False))
    # For streaming datasets, disable decoding entirely
    if hasattr(ds, "decode"):
        try:
            ds = ds.decode(False)
        except Exception:
            pass

    count = 0
    for example in ds:
        text = example.get("text") or example.get("transcript") or example.get("sentence")
        audio = example["audio"]
        if text is None or audio is None:
            continue

        # Decode audio locally using torchaudio (from file path); skip if not available
        wav = None
        sr = 16000
        if isinstance(audio, dict):
            audio_path = audio.get("path") or audio.get("filename") or audio.get("file")
            if audio_path is not None and isinstance(audio_path, str) and os.path.exists(audio_path):
                try:
                    waveform, sr_loaded = torchaudio.load(audio_path)
                    if waveform.dim() == 2 and waveform.size(0) > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    wav = waveform.squeeze(0)
                    sr = int(sr_loaded)
                except Exception:
                    pass
        if wav is None and isinstance(audio, dict) and "array" in audio:
            # If decoded array is present, use it
            wav = torch.tensor(audio["array"], dtype=torch.float32)
            sr = int(audio.get("sampling_rate", sr))
        if wav is None:
            # Can't handle this sample; skip
            continue

        if resample_to and resample_to != sr:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, resample_to).squeeze(0)
            sr = resample_to

        base = f"th_gs2_{count:06d}"
        wav_path = dest / f"{base}.wav"
        txt_path = dest / f"{base}.txt"

        save_wav_tensor(wav, sr, wav_path)
        txt_path.write_text(text.strip() + "\n", encoding="utf-8")

        count += 1
        if max_samples is not None and count >= max_samples:
            break

    print(f"Prepared {count} Thai samples at {dest}")


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
    parser = argparse.ArgumentParser(description="Prepare GigaSpeech2 Thai for Higgs-Audio training")
    parser.add_argument("--output_dir", type=str, required=True, help="Destination directory for prepared data")
    parser.add_argument("--split", type=str, default="train_refined", choices=["train", "train_refined", "train_raw", "dev", "test"], help="Dataset split")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples to convert (streamed)")
    parser.add_argument("--resample_to", type=int, default=16000, help="Target sample rate for saved wav files")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token (or set HUGGINGFACE_TOKEN env var)")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode (may not be supported by all configs)")

    args = parser.parse_args()
    # Try direct Common Voice download first to avoid torchcodec issues on some platforms
    prepared = prepare_common_voice_thai_via_hub(
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        resample_to=args.resample_to,
        hf_token=args.hf_token,
    )
    if prepared and prepared > 0:
        print(f"Prepared {prepared} Thai samples at {args.output_dir} (Common Voice)")
        return

    # Fallback to datasets-based pipeline
    prepare_gigaspeech2_thai(
        output_dir=args.output_dir,
        split=args.split,
        max_samples=args.max_samples,
        resample_to=args.resample_to,
        hf_token=args.hf_token,
        streaming=args.streaming,
    )


if __name__ == "__main__":
    main()

