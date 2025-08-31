#!/usr/bin/env python3
"""
🇹🇭 Thai Text-to-Speech (TTS) Generator
Simple script to generate Thai speech from text using Higgs Audio model.
"""

import argparse
import os
import sys
from pathlib import Path

def generate_thai_speech(
    text: str,
    model_path: str,
    audio_tokenizer_path: str,
    voice: str = "en_woman",  # Default voice
    output_file: str = "thai_output.wav",
    temperature: float = 1.0,
    seed: int = 12345
):
    """Generate Thai speech from text."""
    
    print("🇹🇭 Thai Text-to-Speech Generator")
    print("=" * 50)
    print(f"📝 Text: {text}")
    print(f"🎤 Voice: {voice}")
    print(f"📁 Output: {output_file}")
    print("=" * 50)
    
    # Create temporary text file
    temp_text_file = "temp_thai_text.txt"
    with open(temp_text_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    # Build the generation command
    cmd = [
        "python", "examples/generation.py",
        "--model_path", model_path,
        "--audio_tokenizer", audio_tokenizer_path,
        "--transcript", temp_text_file,
        "--ref_audio", voice,
        "--out_path", output_file,
        "--temperature", str(temperature),
        "--seed", str(seed),
        "--scene_prompt", "empty"  # No scene description for simple TTS
    ]
    
    print("🚀 Generating speech...")
    print(f"Command: {' '.join(cmd)}")
    
    # Execute the command
    result = os.system(' '.join(cmd))
    
    # Clean up temp file
    if os.path.exists(temp_text_file):
        os.remove(temp_text_file)
    
    if result == 0:
        print(f"✅ Success! Thai speech saved to: {output_file}")
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            print(f"📊 File size: {file_size:.2f} MB")
    else:
        print(f"❌ Generation failed with exit code: {result}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate Thai speech from text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default voice
  python thai_tts.py --text "สวัสดีครับ วันนี้อากาศดีมาก" --model_path ./model --audio_tokenizer_path ./tokenizer

  # Use specific voice
  python thai_tts.py --text "สวัสดีค่ะ" --voice belinda --model_path ./model --audio_tokenizer_path ./tokenizer

  # From text file
  python thai_tts.py --text_file thai_text_example.txt --model_path ./model --audio_tokenizer_path ./tokenizer

Available voices: belinda, en_woman, en_man, chadwick, mabel, huo, zh_man_sichuan
        """
    )
    
    # Text input options
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", type=str, help="Thai text to convert to speech")
    text_group.add_argument("--text_file", type=str, help="File containing Thai text to convert")
    
    # Model paths (required)
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to the Higgs Audio model checkpoint")
    parser.add_argument("--audio_tokenizer_path", type=str, required=True,
                       help="Path to the audio tokenizer checkpoint")
    
    # Voice and output options
    parser.add_argument("--voice", type=str, default="en_woman",
                       help="Voice to use (default: en_woman)")
    parser.add_argument("--output", type=str, default="thai_output.wav",
                       help="Output audio file path (default: thai_output.wav)")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Temperature for generation (default: 1.0)")
    parser.add_argument("--seed", type=int, default=12345,
                       help="Random seed for reproducible generation (default: 12345)")
    
    args = parser.parse_args()
    
    # Get text from argument or file
    if args.text:
        text = args.text
    else:
        if not os.path.exists(args.text_file):
            print(f"❌ Text file not found: {args.text_file}")
            return
        with open(args.text_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
    
    # Validate model paths
    if not os.path.exists(args.model_path):
        print(f"❌ Model path not found: {args.model_path}")
        print("💡 You need to download or specify the correct path to your Higgs Audio model")
        return
    
    if not os.path.exists(args.audio_tokenizer_path):
        print(f"❌ Audio tokenizer path not found: {args.audio_tokenizer_path}")
        print("💡 You need to download or specify the correct path to your audio tokenizer")
        return
    
    # Generate speech
    generate_thai_speech(
        text=text,
        model_path=args.model_path,
        audio_tokenizer_path=args.audio_tokenizer_path,
        voice=args.voice,
        output_file=args.output,
        temperature=args.temperature,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
