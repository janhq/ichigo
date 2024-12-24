"""Orchestrates the pipeline to convert text to audio and tokenize the audio
using multiple processes and GPU devices."""

import os
import importlib
import json
import warnings
import time
from multiprocessing import Process, Value

import fire
import torch
import pyarrow as pa
from datasets import Dataset, load_dataset

from writer import Writer, save_batch
from utils import (
    configure_logging,
    upload_folder_to_s3,
    load_config,
    save_failed_indices,
    create_non_overlapping_chunks,
)

warnings.filterwarnings("ignore")


@torch.no_grad()
def process_and_save_audio(
    subset: Dataset,
    device: str,
    process_id: int,
    processed_count: Value,
    save_dir: str,
    save_batch_size: int,
    max_retries: int,
    format,
    tokenizer_cls,
):
    """Process the text and save the audio tokens to a file.

    Args:
        subset (Dataset): The subset of the dataset to process.
        device (str): The device to use for processing.
        process_id (int): The ID of the process.
        processed_count (Value): The shared value to store
        the number of processed items.
        save_dir (str): The directory to save the file.
        save_batch_size (int): The batch size to save to the file.
        sample_rate (int): The sample rate for the audio.
        max_retries (int): The maximum number of retries for processing an item.
        speaker (str): The speaker to use for the TTS.
        format (str): The format of the audio file.
    """
    logger.debug("Process %s will process %s examples.", process_id, len(subset))
    batch_audio = subset["audio"]
    batch_index = subset["text"]

    audio_tokenizer = tokenizer_cls(device=device)

    # Create a writer for this process
    schema = pa.schema(
        [
            pa.field("text", pa.string()),
            pa.field("tokens", pa.list_(pa.int64())),
        ]
    )

    file_path = os.path.join(save_dir, f"audio_tokens_{process_id}")
    writer = Writer(file_path, schema, format)
    logger.debug("Process %s will save to %s.", process_id, file_path)

    failed_indices = []
    saved_failed_indice_path = os.path.join(
        save_dir, f"failed_indices_{process_id}.json"
    )
    logger.debug(
        "Process %s will save failed indices to %s.",
        process_id,
        saved_failed_indice_path,
    )

    batch = []
    for audio, index in zip(batch_audio, batch_index):
        logger.debug("Process %s processing item sample %s.", process_id, index)
        for attempt in range(max_retries):
            try:
                array = audio["array"]
                sampling_rate = audio["sampling_rate"]

                tensor_audio = torch.from_numpy(array).float().unsqueeze(0)
                audio_tokens = audio_tokenizer.encode(
                    (tensor_audio, sampling_rate)
                )

                batch.append(
                    {
                        "text": index,
                        "tokens": audio_tokens,
                    }
                )

                if len(batch) >= save_batch_size:
                    save_batch(batch, writer)
                    batch = []
                    save_failed_indices(failed_indices, saved_failed_indice_path)
                    failed_indices = []

                with processed_count.get_lock():
                    processed_count.value += 1
                break
            except Exception as e:
                logger.warning(
                    "Attempt %s failed for index %s: %s", attempt + 1, index, str(e)
                )
                if attempt == max_retries - 1:
                    logger.error("All attempts failed for index %s", index)
                    failed_indices.append(index)

    # Save any remaining items in the batch
    if batch:
        logger.debug("Saving progress.")
        save_batch(batch, writer)
    if failed_indices:
        logger.info("Saving failed samples.")
        save_failed_indices(failed_indices, saved_failed_indice_path)

    writer.close()


def run_pipeline(
    dataset: Dataset,
    config: dict,
):
    """Run the pipeline to convert text to audio and tokenize the audio.

    Args:
        dataset (Dataset): The dataset to process.
        devices (List): The list of devices to use for processing.
        num_procs_per_device (int): The number of processes to run on each device.
        save_dir (str): The directory to save the files.
        save_batch_size (int): The batch size to save to the files.
        sample_rate (int): The sample rate for the audio.
        max_retries (int): The maximum number of retries for processing an item."""
    print(config)
    # Unpack the configuration
    (
        devices,
        num_procs_per_device,
        save_dir,
        save_batch_size,
        max_retries,
        format,
        tokenizer_cls,
    ) = (
        config[key]
        for key in [
            "devices",
            "num_procs_per_device",
            "save_dir",
            "save_batch_size",
            "max_retries",
            "format",
            "tokenizer",
        ]
    )

    tokenizer_cls = getattr(importlib.import_module("audio_tokenizer"), tokenizer_cls)
    logger.info("Using tokenizer: %s", tokenizer_cls)

    # Create the save directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)
    num_workers = len(devices) * num_procs_per_device
    logger.info("Dataset size: %s", len(dataset))

    # Split the dataset into non-overlapping chunks
    chunks = create_non_overlapping_chunks(dataset, num_workers)

    processed_count = Value("i", 0)  # Value to store the number of items processed

    # Start the worker processes
    worker_processes = []
    for i, chunk in enumerate(chunks):
        device = devices[i % len(devices)]
        p = Process(
            target=process_and_save_audio,
            args=(
                chunk,
                device,
                i,
                processed_count,
                save_dir,
                save_batch_size,
                max_retries,
                format,
                tokenizer_cls
            ),
        )
        p.start()
        worker_processes.append(p)

    while any(p.is_alive() for p in worker_processes):
        # Log the progress every minute
        logger.info("Processed: %s", processed_count.value)
        time.sleep(60)

    # Wait for the worker processes to finish
    for p in worker_processes:
        p.join()

    logger.info("All worker processes have finished.")

    # Log the final counts
    logger.info("Final processed count: %s", processed_count.value)


def main(
    config_path: str = "./configs/synthetic_generation_cfg.yaml",
    test_mode: bool = False,
    name: str = None,
    remaining_indices_file: str = None,
    save_dir: str = None,
):
    """Run the pipeline to convert text to audio and tokenize the audio.

    Args:
        config_path (str): The path to the configuration file."""
    test_mode = False
    config = load_config(config_path)

    # Override config values if provided
    if name:
        config["dataset"]["name"] = name
    if remaining_indices_file:
        config["dataset"]["remaining_indices_file"] = remaining_indices_file
    if save_dir:
        config["processing"]["save_dir"] = save_dir

    global logger
    logger = configure_logging(config)

    dataset = load_dataset(
        config["dataset"]["name"],
        split=config["dataset"]["split"],
        num_proc=config["dataset"]["num_proc"],
    )

    # Check test mode
    if test_mode:
        logger.info("Running in test mode.")
        pipeline_config = config["test"]
        dataset = dataset.select(range(config["test"]["num_samples"]))
    else:
        pipeline_config = config["processing"]
        # Check remaining_indices_file and prepare dataset
        if config["dataset"]["remaining_indices_file"]:
            with open(config["dataset"]["remaining_indices_file"], "r") as f:
                remaining_indices = json.load(f)
            dataset = dataset.select(remaining_indices)
            logger.info(
                "Process %d samples sub-sampling from %s",
                len(dataset),
                config["dataset"]["name"],
            )
        else:
            logger.info("Process FULL samples from %s", config["dataset"]["name"])

    run_pipeline(dataset, pipeline_config)

    if config["upload_to_s3"]:
        logger.info("Uploading files to S3.")
        upload_folder_to_s3(
            config["s3"]["save_dir"],
            config["s3"]["bucket_name"],
            config["s3"]["s3_folder"],
            config["s3"]["num_processes"],
        )


if __name__ == "__main__":
    fire.Fire(main)
