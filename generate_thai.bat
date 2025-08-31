@echo off
echo 🇹🇭 Thai Text-to-Speech Generator
echo ================================

REM Example usage - you need to update the model paths
REM python thai_tts.py --text "สวัสดีครับ วันนี้อากาศดีมาก" --model_path YOUR_MODEL_PATH --audio_tokenizer_path YOUR_TOKENIZER_PATH

echo.
echo To use this script, you need:
echo 1. Download the Higgs Audio model checkpoint
echo 2. Download the audio tokenizer checkpoint  
echo 3. Update the paths in this script
echo.
echo Example command:
echo python thai_tts.py --text "สวัสดีครับ" --model_path ./model --audio_tokenizer_path ./tokenizer
echo.
echo Available voices: belinda, en_woman, en_man, chadwick, mabel, huo
echo.
pause
