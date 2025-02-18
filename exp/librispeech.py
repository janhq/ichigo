import json
import os

import jiwer
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from whisper.normalizers import EnglishTextNormalizer

from ichigo.asr import IchigoASR


class LibriSpeechIchigo(torch.utils.data.Dataset):
    def __init__(self, split="test-clean"):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=os.path.expanduser("~/.cache"),
            url=split,
            download=True,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        audio = audio.to(self.device)
        assert sample_rate == 16000
        return audio, text


def evaluate_config(dataset, time_limit, chunk_sec, overlap_sec, output_dir):
    # Create experiment name
    exp_name = f"chunk{chunk_sec}_overlap{overlap_sec}"
    print(f"\nEvaluating configuration: {exp_name}")

    # Initialize ASR model
    asr = IchigoASR("ichigo-asr-2501-en")
    normalizer = EnglishTextNormalizer()

    predictions = []
    gt = []
    durations = []

    # Process dataset
    for idx in tqdm(range(len(dataset))):
        audio, text = dataset[idx]
        duration = audio.shape[1] / 16000

        if duration >= time_limit:
            result = asr.transcribe_tensor(audio, chunk=chunk_sec, overlap=overlap_sec)
            predictions.append(result)
            gt.append(text)
            durations.append(duration)

    # Create DataFrame
    data = pd.DataFrame(
        {
            "duration": durations,
            "predictions": predictions,
            "gt": gt,
            "predictions_clean": [normalizer(text) for text in predictions],
            "gt_clean": [normalizer(text) for text in gt],
        }
    )

    # Calculate WER
    wer = jiwer.wer(list(data["gt_clean"]), list(data["predictions_clean"]))

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    # Save DataFrame
    data.to_csv(os.path.join(output_dir, f"{exp_name}.csv"), index=False)

    return exp_name, wer


def main():
    # Configurations to test
    chunk_sizes = [2, 5, 10, 20]
    overlap_sizes = [0, 0.1, 0.2, 0.5, 1.0]
    time_limit = 0
    dataset_name = "test-clean"

    # Setup output directory
    output_dir = "librispeech_" + dataset_name
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    dataset = LibriSpeechIchigo(dataset_name)

    # Store WER results
    wer_results = {}

    # Evaluate all configurations
    for chunk_sec in chunk_sizes:
        for overlap_sec in overlap_sizes:
            if overlap_sec >= chunk_sec:
                continue

            exp_name, wer = evaluate_config(
                dataset, time_limit, chunk_sec, overlap_sec, output_dir
            )
            wer_results[exp_name] = float(wer)

    # Save WER results
    with open(os.path.join(output_dir, "wer_results.json"), "w") as f:
        json.dump(wer_results, f, indent=4)

    print("\nEvaluation complete! Results saved in:", output_dir)
    print("\nWER Results:")
    for exp_name, wer in wer_results.items():
        print(f"{exp_name}: {wer*100:.2f}%")


if __name__ == "__main__":
    main()
