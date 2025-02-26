import argparse
import datetime
import logging
import os
from typing import List, Tuple

import jiwer
import pandas as pd
import torch
import whisper
from data import Earnings22ASR, LargeScaleASR, LibriSpeechASR
from termcolor import colored
from tqdm import tqdm
from whisper.normalizers import EnglishTextNormalizer

import wandb
from ichigo.asr import IchigoASR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_wer(
    dataset: torch.utils.data.Dataset,
    model_name: str = "ichigo-asr-2501-en",
    dataset_name: str = "test-clean",
    chunk_size: int = 1000000,
) -> Tuple[str, float]:
    """
    Evaluate Word Error Rate on the dataset.

    Args:
        dataset: Dataset instanceƒ
        model_name: Name of the model to evaluate
        dataset_name: Name of the dataset split
        chunk_size: Audio chunk size for processing

    Returns:
        Tuple of experiment name and WER score
    """
    exp_name = f"{model_name}-{dataset_name}"
    logger.info(f"Starting evaluation: {exp_name}")

    wandb.init(
        project="ichigo-eval",
        name=exp_name,
        config={
            "model_name": model_name,
            "dataset_name": dataset_name,
        },
    )

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
            boundaries = "-" * 50
            logger.info(
                f"\n{boundaries}\n"
                f"WER #{idx}: {colored(f'{wer:.2f}', 'red')}\n"
                f"Pred: {colored(norm_pred, 'cyan')}\n"
                f"GT: {colored(norm_gt, 'green')}\n"
                f"{boundaries}"
            )

            wandb.log({"sample_wer": wer, "audio_duration": duration})

            wers.append(wer)
            gt.append(text)
            durations.append(duration)

        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            continue

    avg_wer = sum(wers) / len(wers)
    logger.info(colored(f"Avg WER: {avg_wer:.2f}", "yellow", attrs=["bold"]))

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

    # Wandb Table
    table = wandb.Table(columns=["sample_id", "wer", "duration", "pred", "gt"])
    for i in range(len(wers)):
        table.add_data(
            i,
            wers[i],
            durations[i],
            data["preds_clean"].iloc[i],
            data["gt_clean"].iloc[i],
        )

    wandb.log(
        {
            "avg_wer": avg_wer,
            "samples_table": table,
        }
    )

    wandb.finish()

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
