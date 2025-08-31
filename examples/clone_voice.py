"""Simple wrapper to clone a voice and synthesize target text using HiggsAudio.

This script forwards your inputs to examples/generation.py with the right flags.

Usage examples:

1) Direct files (any folder):
   python examples/clone_voice.py \
       --text "Hello world" \
       --ref-audio-file "examples/Vivan Test/Untitled video - Made with Clipchamp.m4a" \
       --ref-text-file "examples/Vivan Test/vivan.txt" \
       --out "examples/Vivan Test/output.wav"

2) Named prompt directory (like voice_prompts):
   python examples/clone_voice.py \
       --text-file "examples/transcript/single_speaker/en_dl.txt" \
       --ref-name vivaan \
       --ref-dir "examples/Vivan Test" \
       --out "examples/Vivan Test/output.wav"

       python examples/clone_voice.py --text "Hello, my name is Vivaan" --ref-audio-file "examples/Vivan Test/Untitled video - Made with Clipchamp.m4a" --ref-text-file "examples/Vivan Test/vivan.txt" --out "examples/Vivan Test/output.wav"
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Voice clone wrapper for HiggsAudio")

    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", type=str, help="Target text to synthesize in the cloned voice")
    text_group.add_argument("--text-file", type=str, help="Path to a text file containing the target text")

    ref_group = parser.add_mutually_exclusive_group(required=True)
    ref_group.add_argument("--ref-name", type=str, help="Name of the reference (stem of .wav/.txt) inside --ref-dir")
    ref_group.add_argument("--ref-audio-file", type=str, help="Path to a reference audio file (e.g., WAV/M4A)")

    parser.add_argument("--ref-text-file", type=str, default=None, help="Reference transcript file. If omitted with --ref-audio-file, the script will not infer and generation.py will prompt for explicit path.")
    parser.add_argument("--ref-dir", type=str, default=None, help="Directory that contains named reference prompts when using --ref-name")

    parser.add_argument("--model-path", type=str, default="bosonai/higgs-audio-v2-generation-3B-base", help="Model path or HF repo id")
    parser.add_argument("--audio-tokenizer", type=str, default="bosonai/higgs-audio-v2-tokenizer", help="Audio tokenizer path or HF repo id")
    parser.add_argument("--out", type=str, default="generation.wav", help="Output WAV path")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("--ras-win-len", type=int, default=0, help="RAS window length; set 0 to disable RAS for deterministic runs")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Work around Windows symlink permission issue with huggingface_hub
    if os.name == "nt" and os.environ.get("HF_HUB_DISABLE_SYMLINKS") is None:
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

    # Resolve transcript argument: pass either inline or as a file path
    if args.text is not None:
        transcript_arg = args.text
    else:
        transcript_arg = os.path.abspath(args.text_file)

    def convert_to_wav_if_needed(input_path: str) -> str:
        ext = os.path.splitext(input_path)[1].lower()
        if ext in {".wav", ".flac"}:
            return input_path
        # Try to convert using ffmpeg if available
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            return input_path  # Let downstream raise a clear error; we'll also print a hint below
        # Convert to mono 24kHz WAV
        out_dir = tempfile.gettempdir()
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        out_path = os.path.join(out_dir, f"{base_name}_converted.wav")
        cmd = [ffmpeg_path, "-y", "-i", input_path, "-ar", "24000", "-ac", "1", out_path]
        conv = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if conv.returncode != 0:
            print("ffmpeg conversion failed; stderr:", conv.stderr.decode(errors="ignore"), file=sys.stderr)
            return input_path
        return out_path

    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "generation.py"),
        "--model_path",
        args.model_path,
        "--audio_tokenizer",
        args.audio_tokenizer,
        "--transcript",
        transcript_arg,
        "--out_path",
        os.path.abspath(args.out),
        "--seed",
        str(args.seed),
        "--ras_win_len",
        str(args.ras_win_len),
    ]

    if args.ref_audio_file is not None:
        ref_audio_path = os.path.abspath(args.ref_audio_file)
        converted_path = convert_to_wav_if_needed(ref_audio_path)
        if converted_path == ref_audio_path:
            # If still non-wav, give a helpful note
            ext = os.path.splitext(ref_audio_path)[1].lower()
            if ext not in {".wav", ".flac"} and shutil.which("ffmpeg") is None:
                print("Note: Non-WAV reference detected and ffmpeg not found. If this run fails to read audio, install ffmpeg or convert the file to WAV.", file=sys.stderr)
        cmd.extend(["--ref_audio_file", converted_path])
        if args.ref_text_file is not None:
            cmd.extend(["--ref_text_file", os.path.abspath(args.ref_text_file)])
    else:
        # Using named prompt in a directory
        if args.ref_dir is None:
            print("error: --ref-dir is required when using --ref-name", file=sys.stderr)
            sys.exit(2)
        ref_dir_abs = os.path.abspath(args.ref_dir)
        name_wav = os.path.join(ref_dir_abs, f"{args.ref_name}.wav")
        name_m4a = os.path.join(ref_dir_abs, f"{args.ref_name}.m4a")
        if not os.path.exists(name_wav) and os.path.exists(name_m4a):
            # Auto-convert m4a to wav in place if ffmpeg exists
            ffmpeg_path = shutil.which("ffmpeg")
            if ffmpeg_path is not None:
                conv_cmd = [ffmpeg_path, "-y", "-i", name_m4a, "-ar", "24000", "-ac", "1", name_wav]
                subprocess.run(conv_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                print("Note: Found .m4a without .wav in the prompt dir. Install ffmpeg or convert to WAV manually.", file=sys.stderr)
        cmd.extend(["--ref_audio", args.ref_name, "--ref_audio_dir", ref_dir_abs])

    # Execute the generation
    print("Running:")
    print(" ".join(shlex.quote(x) for x in cmd))
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        sys.exit(completed.returncode)


if __name__ == "__main__":
    main()


