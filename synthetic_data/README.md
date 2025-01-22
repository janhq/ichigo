# Synthetic Data Pipeline - Sound Instruct

## 1. Introduction

The Synthetic Data Pipeline is designed to process large raw-text dataset into trainable sound. This pipeline leverages:
- First, [WhisperSpeech](https://github.com/collabora/WhisperSpeech) to convert text into speech.
- Second, [Encodec](https://github.com/facebookresearch/encodec) to convert into trainable sound tokens.

However, as WhisperSpeech and Encodec do not support batch processing, this pipeline is designed to process data in parallel across multiple GPUs in multiple processes.

## 2. Features

- Multi-GPU and multi-process for each GPU
- Text-to-speech conversion using WhisperSpeech
- Audio tokenization using EncodecModel
- CSV output for easy integration and recovery
- Configurable batch processing
- Progress tracking and logging

## 3. Requirements

### 3.1 Hardware

- Multiple CUDA-capable GPUs (tested with 4 GPUs)
- Sufficient RAM to handle large datasets
- Fast storage for input/output operations

### 3.2 Dependencies

- torch
- pyarrow
- datasets
- WhisperSpeech
- encodec
- Dependencies listed in `requirements.txt`

## 4. Installation

1. Move to the directory:
   ```
   cd synthetic-data
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## 5. Project Structure

```
synthetic-data-pipeline/
├── synthetic_data_pipeline.py
├── audio_tokenizer.py
├── tts_processor.py
├── requirements.txt
├── README.md
└── <save_dir>/
    ├── audio_tokens_*.csv
    └── failed_indices_*.json
```

## 6. Configuration

Edit the `synthetic_generation_cfg.yaml` file to configure the following parameters:

```yaml
# Dataset configuration
dataset:
  name: # Dataset name from Hugging Face
  split: # Dataset split to use
  remaining_indices_file: # File to store remaining indices (List[int])

# Processing configuration
processing:
  devices: # List of GPUs to use
  num_procs_per_device: # Number of processes per GPU
  save_dir: # Directory to save processed data
  save_batch_size: # Batch size for saving processed data for each process
  max_retries: # Maximum number of retries for processing a sample
  sample_rate: # Sample rate for audio data
  speaker: # Speaker class to use for speech generation
  format: # Format for saving processed data

test_mode: # Whether to run the test
test:
  num_samples: # Number of samples to process
  devices: # List of GPUs to use
  num_procs_per_device: # Number of processes per GPU
  save_dir: # Directory to save processed data
  save_batch_size: # Batch size for saving processed data for each process
  max_retries: # Maximum number of retries for processing a sample
  sample_rate: # Sample rate for audio data
  speaker: # Speaker class to use for speech generation

# Logging configuration
logging:
  log_file: # Log file name
  console_level: # Console log level
  file_level: # File log level

upload_to_s3: # Whether to upload processed data to S3
s3:
  save_dir: # Directory to save processed data
  bucket_name: # S3 bucket name
  s3_folder: # S3 folder name
```

### 6.1 Speaker

There are currently 3 supported speakers for text-to-speech conversion:
- default_speaker
- speaker_5304

### 6.2 Format

There are currently 2 supported speakers for text-to-speech conversion:
- parquet
- csv

## 7. Usage

### 7.1 Running the Pipeline

To run the full pipeline:

```python
python synthetic_data_pipeline.py --config_file synthetic_generation_cfg.yaml
```

### 7.2 Test Mode

To run the pipeline in test mode with a smaller dataset. modify the `test_mode` parameter in the configuration file:

```yaml
...
test_mode: true
...
```

## 8. Pipeline Components

### 8.1 Text-to-Speech Processor

The `TTSProcessor` class uses WhisperSpeech to convert text prompts into audio waveforms. It's initialized with a specific device (GPU) and can generate audio from text inputs.

### 8.2 Audio Tokenizer

The `AudioTokenizer` class uses WhisperSpeech and Ichigo Tokenizer to process audio waveforms into tokenized representations. It can also decode audio tokens back into waveforms if needed.

## 9. Data Flow

```mermaid
graph TD
    A[Load Dataset] --> B[Split into Chunks]
    B --> C[Process Chunk]
    
    subgraph PROC ["Process x N"]
        C --> D[Text to Speech Conversion]
        D --> E[Audio Tokenization]
        E --> F[Save to CSV]
        C --> G[Error Handling]
        G -->|Retry| D
        G -->|Max Retries Exceeded| H[Log Failed Indices]
    end
    
    F --> I[Combine Results]
    H --> J[Aggregate Failed Indices]

    classDef PROC padding-left:23em;
    class PROC PROC;
```

## 10. Acknowledgements

This pipeline is built on top of the [WhisperSpeech](https://github.com/collabora/WhisperSpeech) and [Encodec](https://github.com/facebookresearch/encodec) libraries. We would like to thank the developers of these libraries for their contributions to the field of audio processing.
