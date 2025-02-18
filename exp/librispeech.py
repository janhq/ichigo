import argparse
import logging
import os
from typing import List, Optional, Tuple

import jiwer
import pandas as pd
import torch
import torchaudio
import whisper
from tqdm import tqdm
from whisper.normalizers import EnglishTextNormalizer

from ichigo.asr import IchigoASR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LibriSpeechASR(torch.utils.data.Dataset):
    """LibriSpeech dataset wrapper for ASR evaluation."""

    def __init__(
        self,
        is_whisper=False,
        split: str = "test-clean",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the LibriSpeech dataset.

        Args:
            split: Dataset split to use ("test-clean", "test-other", etc.)
            cache_dir: Directory to cache the dataset. Defaults to ~/.cache
        """
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
        self.is_whisper = is_whisper

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, str]:
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        if sample_rate != 16000:
            raise ValueError(f"Expected sample rate 16000, got {sample_rate}")
        if self.is_whisper:
            audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
            mel = whisper.log_mel_spectrogram(audio)
            return (mel, text)
        else:
            audio = audio.to(self.device)
            return audio, text


def evaluate_wer(
    dataset: LibriSpeechASR,
    model_name: str = "ichigo-asr-2501-en",
    dataset_name: str = "test-clean",
    chunk_size: int = 1000000,
) -> Tuple[str, float]:
    """
    Evaluate Word Error Rate on the dataset.

    Args:
        dataset: LibriSpeech dataset instance
        model_name: Name of the model to evaluate
        dataset_name: Name of the dataset split
        chunk_size: Audio chunk size for processing

    Returns:
        Tuple of experiment name and WER score
    """
    exp_name = f"{model_name}_{dataset_name}"
    logger.info(f"Starting evaluation: {exp_name}")

    try:
        if model_name.startswith(("ichigo-asr", "whispervq")):
            asr = IchigoASR(model_name)
        elif model_name.startswith("medium-whisper"):
            asr = whisper.load_model("medium")
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_name}: {e}")

    predictions: List[str] = []
    gt: List[str] = []
    durations: List[float] = []

    for idx in tqdm(range(len(dataset)), desc="Processing audio"):
        try:
            audio, text = dataset[idx]
            duration = audio.shape[1] / 16000

            if isinstance(asr, IchigoASR):
                result = asr.transcribe_tensor(audio, chunk=chunk_size)
                predictions.append(result)
            else:
                result = asr.decode(
                    audio,
                    whisper.DecodingOptions(language="en", without_timestamps=True),
                )
                predictions.append(result.text)

            gt.append(text)
            durations.append(duration)
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            continue

    normalizer = EnglishTextNormalizer()

    data = pd.DataFrame(
        {
            "duration": durations,
            "predictions": predictions,
            "gt": gt,
            "predictions_clean": [normalizer(text) for text in predictions],
            "gt_clean": [normalizer(text) for text in gt],
        }
    )

    wer = jiwer.wer(list(data["gt_clean"]), list(data["predictions_clean"]))

    # Save results
    os.makedirs(dataset_name, exist_ok=True)
    output_path = os.path.join(dataset_name, f"{exp_name}.csv")
    data.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")

    return exp_name, wer


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ASR models on LibriSpeech dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="test-clean",
        help="Dataset split to use (default: test-clean)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ichigo-asr-2501-en",
        help="Model name to evaluate (default: ichigo-asr-2501-en)",
    )

    parser.add_argument(
        "--is_whisper",
        type=bool,
        default=False,
    )

    args = parser.parse_args()

    try:
        dataset = LibriSpeechASR(is_whisper=args.is_whisper, split=args.dataset)
        exp_name, wer = evaluate_wer(
            dataset, model_name=args.model, dataset_name=args.dataset
        )
        logger.info(f"{exp_name}: {wer*100:.2f}%")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
