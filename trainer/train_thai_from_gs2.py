import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare GigaSpeech2 Thai and run LoRA training")
    parser.add_argument("--model_path", type=str, required=True, help="Base Higgs-Audio checkpoint or HF repo id")
    parser.add_argument("--audio_tokenizer_path", type=str, required=True, help="Audio tokenizer checkpoint dir or HF repo id")
    parser.add_argument("--work_dir", type=str, default="./thai_run", help="Working directory")
    parser.add_argument("--split", type=str, default="train_refined", choices=["train_refined", "train_raw"], help="GigaSpeech2 Thai split")
    parser.add_argument("--max_samples", type=int, default=5000, help="Max samples to prepare for a quick run")
    parser.add_argument("--resample_to", type=int, default=16000)
    parser.add_argument("--hf_token", type=str, default=None, help="HF token or set HUGGINGFACE_TOKEN env var")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    work = Path(args.work_dir)
    data_dir = work / "higgs_training_data_th"
    out_dir = work / "output"
    work.mkdir(parents=True, exist_ok=True)

    # Step 1: Prepare Thai data
    prep_cmd = (
        f"python tools/prepare_gigaspeech2_thai.py "
        f"--output_dir {data_dir} "
        f"--split {args.split} "
        f"--max_samples {args.max_samples} "
        f"--resample_to {args.resample_to} "
        + (f"--hf_token {args.hf_token} " if args.hf_token else "")
    )
    print("Preparing data:", prep_cmd)
    if os.system(prep_cmd) != 0:
        raise SystemExit("Data preparation failed")

    # Step 2: Run LoRA training
    train_cmd = (
        f"python trainer/run_thai_lora.py "
        f"--model_path {args.model_path} "
        f"--audio_tokenizer_path {args.audio_tokenizer_path} "
        f"--data_dir {data_dir} "
        f"--output_dir {out_dir} "
        f"--epochs {args.epochs} "
        f"--batch_size {args.batch_size} "
        f"--lr {args.lr} "
        + ("--fp16 " if args.fp16 else "")
    )
    print("Training:", train_cmd)
    if os.system(train_cmd) != 0:
        raise SystemExit("Training failed")

    print(f"Done. Outputs at {out_dir}")


if __name__ == "__main__":
    main()

