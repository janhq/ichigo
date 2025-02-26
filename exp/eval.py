import argparse
import datetime
import logging
import os
from io import BytesIO
from typing import List, Optional, Tuple

import jiwer
import pandas as pd
import torch
import torchaudio
import whisper
from datasets import load_dataset
from tqdm import tqdm
from whisper.normalizers import EnglishTextNormalizer

from ichigo.asr import IchigoASR

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


def evaluate_wer(
    dataset: torch.utils.data.Dataset,
    model_name: str = "ichigo-asr-2501-en",
    dataset_name: str = "test-clean",
    chunk_size: int = 1000000,
) -> Tuple[str, float]:
    """
    Evaluate Word Error Rate on the dataset.

    Args:
        dataset: Dataset instance
        model_name: Name of the model to evaluate
        dataset_name: Name of the dataset split
        chunk_size: Audio chunk size for processing

    Returns:
        Tuple of experiment name and WER score
    """
    exp_name = f"{model_name}-{dataset_name}"
    logger.info(f"Starting evaluation: {exp_name}")

    try:
        if model_name.startswith(("ichigo-asr", "whispervq")):
            asr = IchigoASR(model_name)
        else:
            asr = whisper.load_model(model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_name}: {e}")

    predictions: List[str] = []
    gt: List[str] = []
    durations: List[float] = []
    srs: List[float] = []
    wers: List[float] = []
    normalizer = EnglishTextNormalizer()

    for idx in tqdm(range(len(dataset)), desc="Processing audio"):
        try:
            audio, text, sr = dataset[idx]
            duration = audio.shape[1] / sr
            srs.append(sr)

            if isinstance(asr, IchigoASR):
                result = asr.transcribe_tensor(audio, chunk=chunk_size)
                predictions.append(result)
                norm_pred = normalizer(result)
            else:
                result = asr.decode(
                    audio,
                    whisper.DecodingOptions(language="en", without_timestamps=True),
                )
                predictions.append(result.text)
                norm_pred = normalizer(result.text)

            norm_gt = normalizer(text)
            wer = jiwer.wer(norm_gt, norm_pred)
            print(
                f"\n-----WER ID-{idx}: {wer:.2f}\nPred: {norm_pred}\nGT: {norm_gt}-----\n"
            )

            wers.append(wer)
            gt.append(text)
            durations.append(duration)

        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            continue

    print(f"Avg WER: {sum(wers)/len(wers)}")

    data = pd.DataFrame(
        {
            "wer": wers,
            "length": durations,
            "sr": srs,
            "preds": predictions,
            "gt": gt,
            "preds_clean": [normalizer(text) for text in predictions],
            "gt_clean": [normalizer(text) for text in gt],
        }
    )

    os.makedirs("outputs", exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("outputs", f"{exp_name}_{current_time}.csv")
    data.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")

    data = data.dropna()
    wer = jiwer.wer(list(data["gt_clean"]), list(data["preds_clean"]))

    return exp_name, wer


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ASR models on LibriSpeech dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["test-clean", "test-other", "earnings22", "ls-asr"],
        default="ls-asr",
        help="Dataset to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "ichigo-asr-2501-en",
            "ichigo-asr-2502-medium-en",
            "ichigo-asr-2502-large-en",
            "whispervq-2405-en",
            "medium",
            "large-v3",
        ],
        default="ichigo-asr-2502-medium-en",
        help="Model name to evaluate",
    )

    args = parser.parse_args()

    try:
        if args.dataset == "earnings22":
            dataset = Earnings22ASR(model_name=args.model)
        elif args.dataset == "ls-asr":
            dataset = LargeScaleASR(
                model_name=args.model, subset="medium", split="test"
            )
        else:
            dataset = LibriSpeechASR(model_name=args.model, split=args.dataset)

        exp_name, wer = evaluate_wer(
            dataset, model_name=args.model, dataset_name=args.dataset
        )
        logger.info(f"{exp_name}: {wer*100:.2f}%")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
