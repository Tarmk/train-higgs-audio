import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run LoRA training on Thai GigaSpeech2 data")
    parser.add_argument("--model_path", type=str, required=True, help="Base Higgs-Audio checkpoint dir")
    parser.add_argument("--audio_tokenizer_path", type=str, required=True, help="Audio tokenizer checkpoint dir")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory produced by tools/prepare_gigaspeech2_thai.py")
    parser.add_argument("--output_dir", type=str, default="./output/higgs_th_lora", help="Training output dir")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    # Delegate to trainer.trainer with appropriate flags
    # We use single_speaker_smart_voice task which maps text->audio
    cmd = (
        f"python -m trainer.trainer "
        f"--model_path {args.model_path} "
        f"--audio_tokenizer_path {args.audio_tokenizer_path} "
        f"--train_data_dir {args.data_dir} "
        f"--output_dir {args.output_dir} "
        f"--num_train_epochs {args.epochs} "
        f"--per_device_train_batch_size {args.batch_size} "
        f"--learning_rate {args.lr} "
        f"--use_lora "
        f"--task_type single_speaker_smart_voice "
        f"--report_to none "
        + ("--fp16 " if args.fp16 else "")
    )
    print("Launching:", cmd)
    os.system(cmd)


if __name__ == "__main__":
    main()

