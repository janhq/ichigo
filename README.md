# Ichigo Package

## Installation

```bash
pip install ichigo
```

## ASR

### Batch Processing

```python
# Quick one-liner for transcription
from ichigo.asr import transcribe, get_stoks
results = transcribe("path/to/file/or/folder")
tokens = get_stoks("path/to/file")

# Or with more control using the model class
from ichigo.asr import IchigoASR
model = IchigoASR(config="config-name")
results = model.transcribe(
    "path/to/file/or/folder",
    output_path="./output_folder",
    extensions=(".wav", ".mp3", ".flac", ".m4a")
)
stoks = model.get_stoks("path/to/file")
```

### API

```bash
# Start the API server
uvicorn ichigo.asr.server:app --host 0.0.0.0 --port 8000

# Use with curl for transcription
curl -X POST "http://localhost:8000/transcribe/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/audio.wav"

# Get semantic tokens
curl -X POST "http://localhost:8000/get_stoks/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/audio.wav"
```

You can also access the API documentation at `http://localhost:8000/docs`