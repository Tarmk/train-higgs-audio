#!/usr/bin/env python3
"""
🧪 Test Thai Training Pipeline
Quick test to verify the Thai training pipeline is working correctly.
"""

import os
import sys
from pathlib import Path

def test_data_preparation():
    """Test the Thai data preparation step."""
    print("🧪 Testing Thai data preparation...")
    
    # Test with very small sample
    cmd = "python run_thai_complete.py prepare-data --output_dir ./test_thai_data --max_samples 5"
    result = os.system(cmd)
    
    if result == 0:
        # Check if files were created
        test_path = Path("./test_thai_data")
        wav_files = list(test_path.glob("*.wav"))
        txt_files = list(test_path.glob("*.txt"))
        
        if len(wav_files) > 0 and len(txt_files) > 0:
            print(f"✅ Data preparation test passed! Created {len(wav_files)} audio files")
            return True
        else:
            print("❌ No files were created")
            return False
    else:
        print("❌ Data preparation command failed")
        return False

def test_training_format_conversion():
    """Test converting data to training format."""
    print("🧪 Testing training format conversion...")
    
    try:
        import json
        from pathlib import Path
        
        data_path = Path("./test_thai_data")
        wav_files = list(data_path.glob("*.wav"))
        
        if not wav_files:
            print("❌ No test data available. Run data preparation first.")
            return False
        
        # Convert one sample
        wav_file = wav_files[0]
        txt_file = wav_file.with_suffix('.txt')
        
        if txt_file.exists():
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            sample = {
                "messages": [{"role": "assistant", "content": text}],
                "audios": [str(wav_file.absolute())]
            }
            
            # Write test JSONL
            with open("test_training.jsonl", 'w', encoding='utf-8') as f:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            print("✅ Training format conversion test passed!")
            print(f"📝 Sample text: {text[:50]}...")
            return True
        else:
            print("❌ Text file not found")
            return False
            
    except Exception as e:
        print(f"❌ Training format conversion failed: {e}")
        return False

def test_model_download():
    """Test downloading the base model."""
    print("🧪 Testing model download...")
    
    try:
        from huggingface_hub import snapshot_download
        
        print("📥 Downloading tokenizer (smaller, faster test)...")
        tokenizer_path = snapshot_download("bosonai/higgs-audio-v2-tokenizer")
        
        if os.path.exists(tokenizer_path):
            print(f"✅ Model download test passed! Path: {tokenizer_path}")
            return True
        else:
            print("❌ Download path doesn't exist")
            return False
            
    except Exception as e:
        print(f"❌ Model download failed: {e}")
        return False

def cleanup_test_files():
    """Clean up test files."""
    print("🧹 Cleaning up test files...")
    
    import shutil
    
    # Remove test directories and files
    for path in ["./test_thai_data", "test_training.jsonl"]:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            print(f"🗑️  Removed: {path}")

def main():
    print("🇹🇭 Thai Training Pipeline Test Suite")
    print("=" * 50)
    
    tests = [
        ("Thai Data Preparation", test_data_preparation),
        ("Training Format Conversion", test_training_format_conversion),
        ("Model Download", test_model_download),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📈 {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! Your Thai training pipeline is ready.")
        print("\n🚀 Next steps:")
        print("1. Run: python train_thai_higgs.py --max_samples 500 --epochs 1")
        print("2. Wait for training to complete")
        print("3. Test Thai TTS with your trained model")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    # Cleanup
    cleanup_test_files()

if __name__ == "__main__":
    main()
