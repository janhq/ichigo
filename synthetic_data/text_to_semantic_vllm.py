"""Orchestrates the pipeline to generate text with vllm."""

import os
import json
import warnings
import time
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from multiprocessing import Value, Manager

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
@ray.remote(num_gpus=1, num_cpus=1)
class T2SActor:
    def __init__(self, model_path):
        import os
        import torch

        # Get the GPU IDs assigned to this actor by Ray
        gpu_ids = ray.get_gpu_ids()
        # Set CUDA_VISIBLE_DEVICES to limit the GPUs visible to this process
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        # Set the default CUDA device
        torch.cuda.set_device(0)  # Since only one GPU is visible, it's cuda:0
        # Initialize the LLM model
        self.llm = LLM(model=model_path, device="cuda:0")

        self.sampling_params = SamplingParams(
            max_tokens=1024,
            temperature=0.0,
            stop=["<|sound_end|>"],  
            include_stop_str_in_output=True,
            skip_special_tokens=False,
            
        )

    def generate(self, prompts, indices):
        # Generate text using the LLM instance
        prompts = [f"<|start_header_id|>user<|end_header_id|>\n\n<|reserved_special_token_69|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" for prompt in prompts]
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [(index, output.outputs[0].text) for index, output in zip(indices, outputs)]

def process_and_save_text_vllm(
    subset: Dataset,
    process_id: int,
    processed_count: Value,
    failed_indices: list,
    save_dir: str,
    save_batch_size: int,
    max_retries: int,
    format,
    actor: T2SActor
):
    """Process the text using vllm and save the generated text to a file.

    Args:
        subset (Dataset): The subset of the dataset to process.
        process_id (int): The ID of the process.
        processed_count (Value): The shared value to store the number of processed items.
        save_dir (str): The directory to save the file.
        save_batch_size (int): The batch size to save to the file.
        max_retries (int): The maximum number of retries for processing an item.
        format (str): The format of the output file.
        actor (T2SActor): The Ray actor for vLLM generation.
    """
    logger.debug("Process %s will process %s examples.", process_id, len(subset))
    batch_text = subset["input"]
    batch_index = subset["output"]  # Assuming you have an 'index' column

    # Create a writer for this process
    schema = pa.schema(
        [
            pa.field("answer", pa.string()),  # Assuming index is an integer
            pa.field("generated_text", pa.string()),
        ]
    )

    file_path = os.path.join(save_dir, f"generated_text_{process_id}")
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
    prompts_with_indices = []
    for text, index in zip(batch_text, batch_index):
        logger.debug("Process %s processing item with index %s.", process_id, index)
        for attempt in range(max_retries):
            try:
                # No audio tokenizer needed, 'text' should be your prompt
                prompts_with_indices.append((index, text))

                if len(prompts_with_indices) >= save_batch_size:
                    indices, prompts = zip(*prompts_with_indices)
                    results = ray.get(actor.generate.remote(list(prompts), list(indices)))
                    for index, generated_text in results:
                        batch.append(
                            {
                                "answer": index,
                                "generated_text": generated_text,
                            }
                        )

                    save_batch(batch, writer)
                    batch = []
                    prompts_with_indices = []
                    save_failed_indices(failed_indices, saved_failed_indice_path)

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

    # Save any remaining items
    if prompts_with_indices:
        indices, prompts = zip(*prompts_with_indices)
        results = ray.get(actor.generate.remote(list(prompts), list(indices)))
        for index, generated_text in results:
            batch.append(
                {
                    "answer": index,
                    "generated_text": generated_text,
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
    """Run the pipeline to generate text using vLLM.

    Args:
        dataset (Dataset): The dataset to process.
        config (dict): Configuration dictionary.
    """
    num_cpus = int(config["num_cpus"])
    num_gpus = int(config["num_gpus"])

    # Initialize Ray
    ray.init(num_cpus=num_cpus, num_gpus=num_gpus)

    print(config)
    # Unpack the configuration
    (
        model_path,
        save_dir,
        save_batch_size,
        max_retries,
        format,
    ) = (
        config[key]
        for key in [
            "model_path",
            "save_dir",
            "save_batch_size",
            "max_retries",
            "format",
        ]
    )

    # Create the save directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)
    logger.info("Dataset size: %s", len(dataset))

    # Create a placement group with bundles for T2SActor and data processing
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)] + [{"CPU": 1} for _ in range(num_cpus)]
    pg = placement_group(
        name="llm_pg",
        bundles=bundles,
        strategy="SPREAD"  # Spread tasks across the cluster
    )
    ray.get(pg.ready())

    # Create T2SActor instances
    actors = []
    for _ in range(num_gpus):
        actor = T2SActor.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=_
            )
        ).remote(model_path)
        actors.append(actor)

    # Use Ray's remote functions to process chunks in parallel
    futures = []
    chunk_size = len(dataset) // num_cpus
    for i in range(num_cpus):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_cpus - 1 else len(dataset)
        chunk = dataset.select(range(start, end))

        # Use a remote function to process each chunk, using CPU resources
        future = process_chunk_with_actor.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=num_gpus + i
            )
        ).remote(
            chunk,
            i,
            save_dir,
            save_batch_size,
            max_retries,
            format,
            actors[i % num_gpus],  # Distribute chunks among actors
        )
        futures.append(future)

    # Monitor progress (optional)
    while futures:
        done, futures = ray.wait(futures, timeout=60)
        logger.info("Processed: %s", sum(ray.get(done)))

    # Wait for all tasks to complete
    processed_counts = ray.get(futures)
    total_processed = sum(processed_counts)

    logger.info("All worker processes have finished.")
    logger.info("Final processed count: %s", total_processed)

@ray.remote(num_cpus=1) # Allocate 1 CPU core
def process_chunk_with_actor(
    chunk: Dataset,
    process_id: int,
    save_dir: str,
    save_batch_size: int,
    max_retries: int,
    format: str,
    actor: T2SActor,
) -> int:
    """Process a chunk of the dataset with a given actor."""
    processed_count = 0
    failed_indices = []

    # Call the existing function to process and save data
    process_and_save_text_vllm(
        chunk,
        process_id,
        processed_count,  # Pass as a regular variable
        failed_indices,
        save_dir,
        save_batch_size,
        max_retries,
        format,
        actor,
    )
    
    return processed_count

def main(
    config_path: str = "./configs/synthetic_generation_cfg.yaml",
    test_mode: bool = False,
    name: str = None,
    remaining_indices_file: str = None,
    save_dir: str = None,
):
    """Run the pipeline to generate text using vLLM.

    Args:
        config_path (str): The path to the configuration file.
        test_mode (bool): Whether to run in test mode.
        name (str): The name of the dataset.
        remaining_indices_file (str): Path to a file containing remaining indices to process.
        save_dir (str): The directory to save output files.
    """
    test_mode = True
    # Load the configuration file
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

    # Ensure the dataset has the necessary columns
    # if not all(col in dataset.column_names for col in ["text", "index"]):
    #     raise ValueError("Dataset must contain 'text' and 'index' columns.")

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

    # Add num_cpus and num_gpus to the pipeline configuration
    pipeline_config["num_cpus"] = config["num_cpus"]
    pipeline_config["num_gpus"] = config["num_gpus"]

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