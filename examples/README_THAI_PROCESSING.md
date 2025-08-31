# 🇹🇭 Thai GigaSpeech2 Processing Pipeline

This document describes the enhanced Thai language processing pipeline for the Higgs Audio project, based on the approach demonstrated in `dataset.ipynb`.

## 🌟 Overview

The Thai processing pipeline provides:

- **🎯 Streaming Processing**: Efficient handling of large Thai GigaSpeech2 dataset
- **📝 Transcript Integration**: Automatic matching of Thai audio with transcripts
- **🔐 Secure Authentication**: Programmatic Hugging Face Hub authentication
- **📊 Size Management**: Configurable dataset size limits
- **🔄 Fallback Support**: Automatic fallback to Common Voice Thai if needed

## 🚀 Quick Start

### Prerequisites

1. **Hugging Face Account**: You need access to the GigaSpeech2 dataset
2. **Authentication**: Set up your HF token (see [Authentication](#authentication))
3. **Dependencies**: Install required packages (see [Installation](#installation))

### Basic Usage

```bash
# Process Thai data with default settings
python tools/prepare_gigaspeech2_thai.py \
    --output_dir ./thai_data \
    --max_samples 1000 \
    --hf_token YOUR_HF_TOKEN

# Process with size limit
python tools/prepare_gigaspeech2_thai.py \
    --output_dir ./thai_data \
    --target_size_gb 2.0 \
    --hf_token YOUR_HF_TOKEN

# Run full training pipeline
python trainer/train_thai_from_gs2.py \
    --model_path "your-model-path" \
    --audio_tokenizer_path "your-tokenizer-path" \
    --max_samples 5000 \
    --target_size_gb 1.5 \
    --hf_token YOUR_HF_TOKEN
```

## 📋 Installation

Install the required dependencies:

```bash
pip install datasets huggingface_hub torchaudio torch
```

## 🔐 Authentication

### Method 1: Environment Variable (Recommended)
```bash
export HUGGINGFACE_TOKEN="your_token_here"
```

### Method 2: Command Line Parameter
```bash
python tools/prepare_gigaspeech2_thai.py --hf_token "your_token_here" ...
```

### Method 3: Programmatic Login
The pipeline uses `huggingface_hub.login()` for secure authentication as recommended in the project memory.

## 🔧 Configuration Options

### Data Preparation (`prepare_gigaspeech2_thai.py`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--output_dir` | str | Required | Destination directory for processed data |
| `--split` | str | `train_refined` | Dataset split to use |
| `--max_samples` | int | `None` | Maximum number of samples to process |
| `--target_size_gb` | float | `None` | Stop when dataset reaches this size |
| `--resample_to` | int | `16000` | Target sample rate for audio files |
| `--hf_token` | str | `None` | Hugging Face authentication token |
| `--streaming` | flag | `False` | Enable streaming mode |

### Training Pipeline (`train_thai_from_gs2.py`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model_path` | str | Required | Base Higgs-Audio checkpoint |
| `--audio_tokenizer_path` | str | Required | Audio tokenizer checkpoint |
| `--work_dir` | str | `./thai_run` | Working directory |
| `--max_samples` | int | `5000` | Max samples for quick run |
| `--target_size_gb` | float | `None` | Target dataset size in GB |
| `--epochs` | int | `1` | Number of training epochs |
| `--batch_size` | int | `1` | Training batch size |
| `--lr` | float | `5e-5` | Learning rate |

## 🔄 Processing Pipeline

### Step 1: Authentication
```python
from tools.prepare_gigaspeech2_thai import authenticate_huggingface
authenticate_huggingface(hf_token)
```

### Step 2: Load Transcripts
```python
from tools.prepare_gigaspeech2_thai import load_thai_transcripts
transcripts = load_thai_transcripts(hf_token)
print(f"Loaded {len(transcripts)} transcripts")
```

### Step 3: Stream Dataset
```python
from datasets import load_dataset
dataset = load_dataset(
    "speechcolab/gigaspeech2",
    data_files={'train': 'data/th/train/*.tar.gz'},
    split='train',
    streaming=True,
    token=hf_token
)
```

### Step 4: Process with Transcript Integration
```python
from tools.prepare_gigaspeech2_thai import stream_thai_gigaspeech2_with_transcripts

for sample in stream_thai_gigaspeech2_with_transcripts(dataset, transcripts, max_samples=100):
    segment_id = sample['segment_id']
    audio_data = sample['audio']
    transcript = sample['transcript']
    # Process audio and transcript...
```

## 🧪 Testing

Use the test script to verify your setup:

```bash
# Test authentication only
python examples/test_thai_processing.py --test auth --hf_token YOUR_TOKEN

# Test transcript loading
python examples/test_thai_processing.py --test transcripts --hf_token YOUR_TOKEN

# Test streaming dataset
python examples/test_thai_processing.py --test streaming --hf_token YOUR_TOKEN

# Test full integration
python examples/test_thai_processing.py --test integration --hf_token YOUR_TOKEN

# Test complete pipeline
python examples/test_thai_processing.py --test full --output_dir ./test_output --hf_token YOUR_TOKEN

# Run all tests
python examples/test_thai_processing.py --test all --hf_token YOUR_TOKEN
```

## 📊 Output Format

The pipeline generates paired audio and transcript files:

```
output_dir/
├── th_gs2_000000.wav    # Audio file (16kHz, mono)
├── th_gs2_000000.txt    # Transcript file (UTF-8)
├── th_gs2_000001.wav
├── th_gs2_000001.txt
└── ...
```

### File Naming Convention
- **Audio**: `th_gs2_XXXXXX.wav` (Thai GigaSpeech2)
- **Transcripts**: `th_gs2_XXXXXX.txt`
- **Fallback**: `th_cv_XXXXXX.wav/txt` (Common Voice)

## 🔍 Data Schema

Each processed sample contains:

```python
{
    'segment_id': str,           # Unique segment identifier
    'audio': {
        'array': np.ndarray,     # Audio waveform
        'sampling_rate': int     # Sample rate
    },
    'transcript': str,           # Thai text transcript
    'metadata': {
        'original_key': str,     # Original dataset key
        'url': str,              # Source URL
        'has_transcript': bool   # Transcript availability
    }
}
```

## ⚠️ Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Ensure your HF token has access to GigaSpeech2
   - Check token permissions on Hugging Face Hub

2. **Dataset Access Denied**
   - Accept the dataset terms at: https://huggingface.co/datasets/speechcolab/gigaspeech2
   - Verify your account has the necessary permissions

3. **Memory Issues**
   - Use `--streaming` flag for large datasets
   - Reduce `--max_samples` or set `--target_size_gb`

4. **Transcript Mismatches**
   - The pipeline only processes samples with matching transcripts
   - Check transcript loading success in logs

### Fallback Behavior

The pipeline includes automatic fallbacks:

1. **Primary**: Thai GigaSpeech2 with transcript integration
2. **Fallback**: Common Voice Thai dataset
3. **Error Handling**: Graceful degradation with informative messages

## 🎯 Performance Tips

1. **Use Streaming**: Always enable `--streaming` for large datasets
2. **Set Size Limits**: Use `--target_size_gb` to control dataset size
3. **Batch Processing**: Process in chunks for better memory management
4. **Monitor Progress**: Watch for progress indicators in logs

## 📈 Example Usage Scenarios

### Quick Test (Small Dataset)
```bash
python tools/prepare_gigaspeech2_thai.py \
    --output_dir ./thai_test \
    --max_samples 100 \
    --hf_token $HUGGINGFACE_TOKEN
```

### Production Training (Large Dataset)
```bash
python trainer/train_thai_from_gs2.py \
    --model_path "higgs-audio-base" \
    --audio_tokenizer_path "higgs-tokenizer" \
    --target_size_gb 10.0 \
    --epochs 3 \
    --batch_size 4 \
    --hf_token $HUGGINGFACE_TOKEN
```

### Custom Processing
```bash
python tools/prepare_gigaspeech2_thai.py \
    --output_dir ./custom_thai \
    --split train_refined \
    --resample_to 22050 \
    --target_size_gb 5.0 \
    --streaming \
    --hf_token $HUGGINGFACE_TOKEN
```

## 🤝 Contributing

When contributing to the Thai processing pipeline:

1. Follow the notebook-style approach for consistency
2. Ensure proper error handling and fallbacks
3. Add progress indicators for long-running operations
4. Test with various dataset sizes and configurations
5. Update documentation for any new features

## 📚 References

- [Dataset Notebook](../dataset%20(1).ipynb) - Original processing approach
- [GigaSpeech2 Dataset](https://huggingface.co/datasets/speechcolab/gigaspeech2)
- [Common Voice Thai](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0)
- [Hugging Face Hub Authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)
