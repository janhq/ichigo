import logging
import os
from io import BytesIO
from typing import Optional, Tuple

import torch
import torchaudio
import whisper
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LibriSpeechASR(torch.utils.data.Dataset):
    def __init__(
        self,
        model_name,
        split: str = "test-clean",
        cache_dir: Optional[str] = None,
    ):
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache")
        try:
            self.dataset = torchaudio.datasets.LIBRISPEECH(
                root=self.cache_dir,
                url=split,
                download=True,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load LibriSpeech dataset: {e}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, str]:
        audio, sr, text, _, _, _ = self.dataset[item]
        if sr != 16000:
            raise ValueError(f"Expected sample rate 16000, got {sr}")
        if self.model_name in ["medium", "large-v3"]:
            audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
            n_mels = 80 if self.model_name == "medium" else 128
            mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels)
            return mel, text, sr
        else:
            audio = audio.to(self.device)
            return audio, text, sr


class Earnings22ASR(torch.utils.data.Dataset):
    def __init__(
        self,
        model_name,
    ):
        try:
            self.dataset = load_dataset(
                "anton-l/earnings22_baseline_5_gram",
                split="test",
                trust_remote_code=True,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Earnings22 dataset: {e}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, str]:
        example = self.dataset[item]
        audio = torch.tensor(example["audio"]["array"]).float()
        sr = example["audio"]["sampling_rate"]
        if self.model_name in ["medium", "large-v3"]:
            audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
            n_mels = 80 if self.model_name == "medium" else 128
            mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels)
            return mel, example["sentence"], sr
        else:
            audio = audio.to(self.device)
            return audio.unsqueeze(0), example["sentence"], sr


class LargeScaleASR(torch.utils.data.Dataset):
    def __init__(
        self,
        model_name,
        subset: str = "medium",
        split: str = "test",
    ):
        try:
            self.dataset = load_dataset(
                "/home/jovyan/aws-s3-data/aws-s3/tuanlda78202",
                subset,
                split=split,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load LargeScaleASR dataset: {e}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, str]:
        example = self.dataset[item]
        audio, sr = torchaudio.load(BytesIO(example["wav"]["bytes"]))

        if self.model_name in ["medium", "large-v3"]:
            audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
            n_mels = 80 if self.model_name == "medium-whisper" else 128
            mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels)
            return mel, example["text"], sr
        else:
            audio = audio.to(self.device)
            return audio, example["text"], sr
