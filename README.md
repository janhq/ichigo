<div align="center">

# :strawberry: Ichigo: A simple speech package optimised for local inference
<a href='https://homebrew.ltd/blog/llama3-just-got-ears'><img src='https://img.shields.io/badge/Project-Blog-Green'></a>
<a href='https://ichigo.homebrew.ltd/'><img src='https://img.shields.io/badge/Project-Demo-violet'></a>
<a href='https://arxiv.org/pdf/2410.15316'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href='https://huggingface.co/homebrewltd'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>
<a href='https://huggingface.co/homebrewltd'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-green'></a>
<a href='https://colab.research.google.com/drive/18IiwN0AzBZaox5o0iidXqWD1xKq11XbZ?usp=sharing'><img src='https://colab.research.google.com/assets/colab-badge.svg'></a>

[**About**](#About) | [**Installation**](#Installation) | [**Ichigo-ASR**](#Ichigo-ASR) | [**Ichigo-LLM**](#Ichigo-LLM)

<img src="assets/ichigo.jpeg" width="400"/>
</div>

## About

This package does 3 things: 
1) Automatic Speech Recognition: [**Ichigo-ASR**](#ichigo-asr)
2) Text to Speech: Coming Soon
3) Speech Language Model: [**Ichigo-LLM**](#ichigo-llm) (experimental)

It contains only inference code, and caters to most local inference use cases around these three tasks.

## Installation

To get started, simply install the package.

```bash
pip install ichigo
```

## Ichigo-ASR

Ichigo-ASR is a compact (22M parameters), open-source speech tokenizer for the `Whisper-medium model`, designed to enhance performance on multilingual with minimal impact on its original English capabilities. Unlike models that output continuous embeddings, Ịchigo-ASR compresses speech into discrete tokens, making it more compatible with large language models (LLMs) for immediate speech understanding. This speech tokenizer has been trained on over ~400 hours of English data and ~1000 hours of Vietnamese data.

### Batch Processing

The ichigo package can handle batch processing of audio files using a single line of code, with additional parameters for available for more control.

1. For single files

```python
# Quick one-liner for transcription
from ichigo.asr import transcribe
results = transcribe("path/to/your/file")
# Expected output: "{filename: transcription}"
```
A transcription.txt will also stored in the same folder as "path/to/your/file"

2. For multiple files (folder)

```python
# Quick one-liner for transcription
from ichigo.asr import transcribe
results = transcribe("path/to/your/folder")
# Expected output: "{filename1: transcription1, filename2: transcription2, ... filenameN: transcriptionN,}"
```
A subfolder will be created in `path/to/your/folder` and transcriptions will be stored as `filenameN.txt` in the subfolder.

### API

For integration with frontend, a python fastAPI is also available. This api also does batch processing. Streaming is currently not supported.

1. Start the server

```bash
# Uvicorn
cd api && uvicorn asr:app --host 0.0.0.0 --port 8000

# or Docker 
docker compose up -d
```

2. curl
```bash
# S2T
curl "http://localhost:8000/v1/audio/transcriptions" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.wav" -F "model=ichigo"

# S2R
curl "http://localhost:8000/s2r" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.wav"

# R2T
curl "http://localhost:8000/r2t" -X POST \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  --data '{"tokens":"<|sound_start|><|sound_1012|><|sound_1508|><|sound_1508|><|sound_0636|><|sound_1090|><|sound_0567|><|sound_0901|><|sound_0901|><|sound_1192|><|sound_1820|><|sound_0547|><|sound_1999|><|sound_0157|><|sound_0157|><|sound_1454|><|sound_1223|><|sound_1223|><|sound_1223|><|sound_1223|><|sound_1808|><|sound_1808|><|sound_1573|><|sound_0065|><|sound_1508|><|sound_1508|><|sound_1268|><|sound_0568|><|sound_1745|><|sound_1508|><|sound_0084|><|sound_1768|><|sound_0192|><|sound_1048|><|sound_0826|><|sound_0192|><|sound_0517|><|sound_0192|><|sound_0826|><|sound_0971|><|sound_1845|><|sound_1694|><|sound_1048|><|sound_0192|><|sound_1048|><|sound_1268|><|sound_end|>"}'
```

You can also access the API documentation at `http://localhost:8000/docs`

## Ichigo-LLM

:strawberry: Ichigo-LLM is an open, ongoing research experiment to extend a text-based LLM to have native "listening" ability. Think of it as an open data, open weight, on device Siri.

It uses an [early fusion](https://medium.com/@raj.pulapakura/multimodal-models-and-fusion-a-complete-guide-225ca91f6861#:~:text=3.3.,-Early%20Fusion&text=Early%20fusion%20refers%20to%20combining,fused%20representation%20through%20the%20model.) technique inspired by [Meta's Chameleon paper](https://arxiv.org/abs/2405.09818).

We ~~build~~ train in public:
- [Ichigo v0.3 Checkpoint Writeup](https://homebrew.ltd/blog/llama-learns-to-talk)
- [Ichigo v0.2 Checkpoint Writeup](https://homebrew.ltd/blog/llama3-just-got-ears)
- [Ichigo v0.1 Checkpoint Writeup](https://homebrew.ltd/blog/can-llama-3-listen)

## Ichigo-TTS

Coming Soon
  
## Join Us

:strawberry: Ichigo-LLM and 🍰 Ichigo-ASR is an open research project. We're looking for collaborators, and will likely move towards crowdsourcing speech datasets in the future.

## References

```bibtex
@article{dao2024ichigo,
  title={Ichigo: Mixed-Modal Early-Fusion Realtime Voice Assistant},
  author={Dao, Alan and Vu, Dinh Bach and Ha, Huy Hoang},
  journal={arXiv preprint arXiv:2410.15316},
  year={2024}
}

@misc{chameleonteam2024chameleonmixedmodalearlyfusionfoundation,
      title={Chameleon: Mixed-Modal Early-Fusion Foundation Models}, 
      author={Chameleon Team},
      year={2024},
      eprint={2405.09818},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      journal={arXiv preprint}
}

@misc{WhisperSpeech,
      title={WhisperSpeech: An Open Source Text-to-Speech System Built by Inverting Whisper}, 
      author={Collabora and LAION},
      year={2024},
      url={https://github.com/collabora/WhisperSpeech},
      note={GitHub repository}
}
```

## Acknowledgement

- [torchtune](https://github.com/pytorch/torchtune): The codebase we built upon
- [WhisperSpeech](https://github.com/collabora/WhisperSpeech): Text-to-speech model for synthetic audio generation
- [llama3](https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6): the Family of Models that we based on that has the amazing language capabilities
