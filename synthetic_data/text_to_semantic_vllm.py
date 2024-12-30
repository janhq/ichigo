"""Orchestrates the pipeline to convert text to audio and generate with vllm."""

import os
import importlib
import json
import warnings
import time
from multiprocessing import Process, Value, Manager

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
from vllm import LLM, SamplingParams

warnings.filterwarnings("ignore")

def initialize_vllm_model(
    model_path: str,
    gpu_utilization: float = 0.95,
    max_model_len: int = 4096,
) -> LLM:
    """Initializes the vLLM model with optimized parameters."""
    return LLM(
        model=model_path,
        gpu_memory_utilization=gpu_utilization,
        max_model_len=max_model_len,
    )

def process_and_save_audio_vllm(
    subset: Dataset,
    process_id: int,
    processed_count: Value,
    failed_indices: list,
    save_dir: str,
    save_batch_size: int,
    max_retries: int,
    format,
    model_path: str,
    sampling_params_dict: dict,
):
    """Process the text and generate with vllm and save the audio tokens to a file.

    Args:
        subset (Dataset): The subset of the dataset to process.
        process_id (int): The ID of the process.
        processed_count (Value): The shared value to store the number of processed items.
        save_dir (str): The directory to save the file.
        save_batch_size (int): The batch size to save to the file.
        max_retries (int): The maximum number of retries for processing an item.
        format (str): The format of the audio file.
        llm_model (LLM): The vLLM model for generation.
        sampling_params (SamplingParams): The sampling parameters for generation.
    """
    # Assuming each process will handle a single GPU
    device_id = process_id % torch.cuda.device_count()
    torch.cuda.set_device(device_id)

    # Initialize vLLM model on the specific GPU
    llm_model = initialize_vllm_model(model_path)
    sampling_params = SamplingParams(**sampling_params_dict)

    logger.debug("Process %s will process %s examples on GPU %d.", process_id, len(subset), device_id)
    batch_audio = subset["audio"]
    batch_index = subset["transcript"]

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

    saved_failed_indice_path = os.path.join(
        save_dir, f"failed_indices_{process_id}.json"
    )
    logger.debug(
        "Process %s will save failed indices to %s.",
        process_id,
        saved_failed_indice_path,
    )

    batch = []
    prompts = []
    for audio, index in zip(batch_audio, batch_index):
        logger.debug("Process %s processing item sample %s on GPU %d.", process_id, index, device_id)
        for attempt in range(max_retries):
            try:
                array = audio["array"]
                sampling_rate = audio["sampling_rate"]

                tensor_audio = torch.from_numpy(array).float().unsqueeze(0)

                # Assuming audio_tokenizer.prepare_prompts is a method to prepare prompts for vLLM
                prepared_prompt = audio_tokenizer.prepare_prompts(tensor_audio, sampling_rate)

                prompts.append(prepared_prompt)

                if len(prompts) >= save_batch_size:
                    # Use vLLM for batch generation
                    outputs = llm_model.generate(prompts, sampling_params)
                    for output in outputs:
                        # Assuming audio_tokenizer.postprocess_token is a method to postprocess vLLM output
                        audio_tokens = audio_tokenizer.postprocess_token(output.outputs[0].token_ids)
                        batch.append(
                            {
                                "text": index,
                                "tokens": audio_tokens,
                            }
                        )
                    save_batch(batch, writer)
                    batch = []
                    prompts = []
                    save_failed_indices(failed_indices, saved_failed_indice_path)

                with processed_count.get_lock():
                    processed_count.value += 1
                break
            except Exception as e:
                logger.warning(
                    "Attempt %s failed for index %s on GPU %d: %s", attempt + 1, index, device_id, str(e)
                )
                if attempt == max_retries - 1:
                    logger.error("All attempts failed for index %s on GPU %d", index, device_id)
                    failed_indices.append(index)

    # Save any remaining items in the batch
    if prompts:
        outputs = llm_model.generate(prompts, sampling_params)
        for output in outputs:
            audio_tokens = audio_tokenizer.postprocess_token(output.outputs[0].token_ids)
            batch.append(
                {
                    "text": index,
                    "tokens": audio_tokens,
                }
            )
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
        model_path,
        num_procs_per_device,
        save_dir,
        save_batch_size,
        max_retries,
        format,
        tokenizer_cls,
        sampling_params_dict,
    ) = (
        config[key]
        for key in [
            "model_path",
            "num_procs_per_device",
            "save_dir",
            "save_batch_size",
            "max_retries",
            "format",
            "tokenizer",
            "sampling_params",
        ]
    )

    tokenizer_cls = getattr(importlib.import_module("audio_tokenizer"), tokenizer_cls)
    logger.info("Using tokenizer: %s", tokenizer_cls)

    # Create the save directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)
    num_workers = 8 * num_procs_per_device  # Assuming 8 GPUs
    logger.info("Dataset size: %s", len(dataset))

    # Split the dataset into non-overlapping chunks
    chunks = create_non_overlapping_chunks(dataset, num_workers)

    processed_count = Value("i", 0)  # Value to store the number of items processed
    with Manager() as manager:
        failed_indices = manager.list()  # Shared list to store failed indices

        # Start the worker processes
        worker_processes = []
        for i, chunk in enumerate(chunks):
            p = Process(
                target=process_and_save_audio_vllm,
                args=(
                    chunk,
                    i,
                    processed_count,
                    failed_indices,
                    save_dir,
                    save_batch_size,
                    max_retries,
                    format,
                    model_path,
                    sampling_params_dict,
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