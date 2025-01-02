import jiwer
import os
import sys
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM, SamplingParams
from datasets import load_dataset
import pandas as pd

def main():
    print(sys.executable)

    # Initialize Ray
    ray.init()

    # Define the number of models (and GPUs) you want to use
    num_models = int(os.environ.get("NUM_GPUS", "6"))  # Adjust this based on your available GPUs or set through environment variable

    # Create a placement group with one GPU and one CPU per bundle
    pg = placement_group(
        name="llm_pg",
        bundles=[{"GPU": 1, "CPU": 3} for _ in range(num_models)],
        strategy="STRICT_PACK"
    )
    ray.get(pg.ready())

    # Define the LLMActor class
    @ray.remote(num_gpus=1, num_cpus=3)
    class LLMActor:
        def __init__(self, model_name):
            import os
            import torch

            gpu_ids = ray.get_gpu_ids()
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(int(gpu_id)) for gpu_id in gpu_ids)
            torch.cuda.set_device(0)
            self.llm = LLM(model=model_name, device="cuda:0", gpu_memory_utilization=0.95, max_num_seqs=480)
            self.sampling_params = SamplingParams(
                max_tokens=1024,
                temperature=0.0,
                stop=["<|sound_end|>"],
                include_stop_str_in_output=True,
                skip_special_tokens=False,
            )

        def generate(self, prompts):
            # Generate text for a batch of prompts
            outputs = self.llm.generate(prompts, self.sampling_params)
            return [output.outputs[0].text.replace("<|eot_id|>", "") for output in outputs]

    # Load dataset from Hugging Face
    dataset_name = "homebrewltd/instruction-speech-whispervq-v3-subset-2" 
    drop_column = ["tokens", "prompt", "conversations"]
    dataset = load_dataset(dataset_name, split="train")
    # Drop unnecessary columns
    dataset = dataset.remove_columns(drop_column)
    text_column_name = "text_prompt"  

    # Divide the dataset equally among the actors
    dataset_size = len(dataset)
    chunk_size = dataset_size // num_models
    remainder = dataset_size % num_models
    
    start_indices = []
    end_indices = []
    for i in range(num_models):
        start = i * chunk_size + min(i, remainder)
        end = start + chunk_size + (1 if i < remainder else 0)
        start_indices.append(start)
        end_indices.append(end)

    # Model and prompt setup
    model_name = "homebrewltd/Speechless-llama3.2-v0.1"
    prompt_template = "<|start_header_id|>user<|end_header_id|>\n\n<|reserved_special_token_69|>{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    # Create LLMActor instances
    actors = [
        LLMActor.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=i
            )
        ).remote(model_name)
        for i in range(num_models)
    ]

    # Generate text in parallel
    futures = []
    for i, actor in enumerate(actors):
        # Prepare prompts for the current actor's chunk
        prompts = [
            prompt_template.format(text=text)
            for text in dataset[text_column_name][start_indices[i]:end_indices[i]]
        ]
        # Divide prompts into smaller batches for each actor
        batch_size = 480 # You can adjust this based on your memory constraints
        for j in range(0, len(prompts), batch_size):
            futures.append(actor.generate.remote(prompts[j:j+batch_size]))

    # Retrieve results and combine them
    results = ray.get(futures)
    generated_texts = []
    for result_batch in results:
        generated_texts.extend(result_batch)

    # Add generated text to the dataset
    dataset = dataset.add_column("compressed_prompt", generated_texts)
    
    # Save the updated dataset (optional, you can choose a different format or location)
    dataset.save_to_disk("./instruction-speech-v2-ichigo-tokens/") # You can also save to Huggingface hub by using push_to_hub() method
    # dataset.push_to_hub("jan-hq/ichigo-instruction-dataset")
    print("Dataset updated and saved successfully!")

if __name__ == "__main__":
    main()