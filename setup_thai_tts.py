#!/usr/bin/env python3
"""
🇹🇭 Thai TTS Setup Script
Downloads the required models and tests Thai text-to-speech functionality.
"""

import os
import sys
from pathlib import Path

def setup_models():
    """Download and setup the Higgs Audio models."""
    print("🚀 Setting up Higgs Audio models for Thai TTS...")
    print("=" * 60)
    
    # Model paths from Hugging Face
    model_path = "bosonai/higgs-audio-v2-generation-3B-base"
    tokenizer_path = "bosonai/higgs-audio-v2-tokenizer"
    
    print(f"📥 Model: {model_path}")
    print(f"🎯 Tokenizer: {tokenizer_path}")
    print()
    
    # Test if we can load the models
    try:
        from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
        from boson_multimodal.data_types import ChatMLSample, Message
        
        print("✅ Boson multimodal imports successful")
        
        # Try to initialize the engine (this will download models if needed)
        print("🔄 Initializing Higgs Audio engine...")
        print("⏳ This may take a few minutes to download models (~6GB)...")
        
        device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
        print(f"🖥️  Using device: {device}")
        
        engine = HiggsAudioServeEngine(
            model_name_or_path=model_path,
            audio_tokenizer_name_or_path=tokenizer_path,
            device=device
        )
        
        print("✅ Models loaded successfully!")
        return True, model_path, tokenizer_path
        
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        print("\n💡 Make sure you have:")
        print("1. Installed all dependencies: pip install -r requirements.txt")
        print("2. Internet connection for downloading models")
        print("3. Sufficient disk space (~6GB)")
        return False, None, None

def test_thai_generation():
    """Test Thai text generation."""
    print("\n🧪 Testing Thai text-to-speech generation...")
    print("=" * 50)
    
    try:
        from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
        from boson_multimodal.data_types import ChatMLSample, Message
        import torch
        import torchaudio
        
        model_path = "bosonai/higgs-audio-v2-generation-3B-base"
        tokenizer_path = "bosonai/higgs-audio-v2-tokenizer"
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        engine = HiggsAudioServeEngine(
            model_name_or_path=model_path,
            audio_tokenizer_name_or_path=tokenizer_path,
            device=device
        )
        
        # Thai test text
        thai_text = "สวัสดีครับ ผมชื่อ AI ผมสามารถพูดภาษาไทยได้"
        
        system_prompt = (
            "Generate audio following instruction.\n\n"
            "<|scene_desc_start|>\n"
            "Audio is recorded from a quiet room.\n"
            "<|scene_desc_end|>"
        )
        
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=thai_text),
        ]
        
        sample = ChatMLSample(messages=messages)
        
        print(f"📝 Generating audio for: {thai_text}")
        print("⏳ This may take a moment...")
        
        response = engine.generate(
            chat_ml_sample=sample,
            max_new_tokens=2048,
            temperature=1.0,
            seed=12345
        )
        
        if response.audio is not None:
            output_path = "thai_test_output.wav"
            torchaudio.save(output_path, torch.tensor(response.audio).unsqueeze(0), 24000)
            print(f"✅ Success! Audio saved to: {output_path}")
            
            # File info
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"📊 File size: {file_size:.2f} MB")
            print(f"🎵 Sample rate: 24kHz")
            
            return True
        else:
            print("❌ No audio generated")
            return False
            
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False

def main():
    print("🇹🇭 Thai Text-to-Speech Setup & Test")
    print("=" * 60)
    
    # Check if required packages are installed
    try:
        import torch
        import torchaudio
        from huggingface_hub import login
        print("✅ Basic dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Please install: pip install -r requirements.txt")
        return
    
    # Setup models
    success, model_path, tokenizer_path = setup_models()
    
    if not success:
        print("\n❌ Setup failed. Please check the errors above.")
        return
    
    # Test generation
    if test_thai_generation():
        print("\n🎉 Setup complete! You can now use Thai TTS.")
        print("\n📋 Usage examples:")
        print(f"python thai_tts.py --text 'สวัสดีครับ' --model_path {model_path} --audio_tokenizer_path {tokenizer_path}")
        print("python thai_tts.py --text_file thai_text_example.txt --model_path bosonai/higgs-audio-v2-generation-3B-base --audio_tokenizer_path bosonai/higgs-audio-v2-tokenizer")
    else:
        print("\n❌ Test generation failed.")

if __name__ == "__main__":
    main()
