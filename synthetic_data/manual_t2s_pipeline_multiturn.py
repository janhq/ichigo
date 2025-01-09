import os
import sys
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM, SamplingParams
from datasets import load_dataset
from tqdm import tqdm

def main():
    print(sys.executable)

    # Initialize Ray
    ray.init()

    # Define the number of models (and GPUs)
    num_models = int(os.environ.get("NUM_GPUS", "4"))
    print(f"Using {num_models} GPUs")

    # Create placement group
    pg = placement_group(
        name="llm_pg",
        bundles=[{"GPU": 1, "CPU": 3} for _ in range(num_models)],
        strategy="STRICT_PACK"
    )
    ray.get(pg.ready())

    @ray.remote(num_gpus=1, num_cpus=3)
    class LLMActor:
        def __init__(self, model_name):
            import os
            import torch

            gpu_ids = ray.get_gpu_ids()
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(int(gpu_id)) for gpu_id in gpu_ids)
            torch.cuda.set_device(0)
            print(f"Initializing LLM on GPU {os.environ['CUDA_VISIBLE_DEVICES']}")
            
            self.llm = LLM(
                model=model_name, 
                device="cuda:0", 
                gpu_memory_utilization=0.95,
                max_num_seqs=480
            )
            self.sampling_params = SamplingParams(
                max_tokens=1024,
                temperature=0.0,
                stop=["<|sound_end|>"],
                include_stop_str_in_output=True,
                skip_special_tokens=False,
            )

        def generate(self, prompts):
            outputs = self.llm.generate(prompts, self.sampling_params)
            return [output.outputs[0].text.replace("<|eot_id|>", "") for output in outputs]

    # Load dataset
    dataset_name = "jan-hq/instruction-data-text-only-multiturn-filtered-for-tokenize-clean"
    dataset = load_dataset(dataset_name, split="train")
    conversations_column_name = "conversations"

    # Calculate distribution
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
        print(f"GPU {i} will process conversations {start} to {end}")

    # Model and prompt setup
    model_name = "homebrewltd/Speechless-llama3.2-v0.1"
    prompt_template = "<|start_header_id|>user<|end_header_id|>\n\n<|reserved_special_token_69|>{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    # Create actors
    actors = [
        LLMActor.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=i
            )
        ).remote(model_name)
        for i in range(num_models)
    ]

    # Process conversations
    futures = []
    for i, actor in enumerate(actors):
        conversations = dataset[conversations_column_name][start_indices[i]:end_indices[i]]
        
        # Prepare prompts for all user messages in conversations
        prompts = []
        prompt_positions = []  # Track which conversation and turn each prompt came from
        
        for conv_idx, conv in enumerate(conversations):
            for turn_idx, turn in enumerate(conv):
                if turn["role"] == "user":
                    prompts.append(prompt_template.format(text=turn["content"]))
                    prompt_positions.append((conv_idx, turn_idx))
        
        # Process prompts in batches
        batch_size = 240
        for j in range(0, len(prompts), batch_size):
            batch_prompts = prompts[j:j + batch_size]
            batch_positions = prompt_positions[j:j + batch_size]
            futures.append((i, batch_positions, actor.generate.remote(batch_prompts)))
            print(f"GPU {i}: Submitted batch of {len(batch_prompts)} prompts")

    # Collect results
    print("Processing batches...")
    results_by_gpu = {i: [] for i in range(num_models)}
    
    with tqdm(total=len(futures)) as pbar:
        for gpu_idx, positions, future in futures:
            results = ray.get(future)
            results_by_gpu[gpu_idx].extend(zip(positions, results))
            pbar.update(1)

    # Reconstruct conversations with processed text
    processed_conversations = []
    for i in range(num_models):
        chunk_conversations = dataset[conversations_column_name][start_indices[i]:end_indices[i]]
        chunk_results = results_by_gpu[i]
        
        # Process each conversation in the chunk
        for conv_idx, conv in enumerate(chunk_conversations):
            processed_conv = []
            result_idx = 0
            
            for turn in conv:
                if turn["role"] == "user":
                    # Find the corresponding result
                    while result_idx < len(chunk_results) and chunk_results[result_idx][0][0] != conv_idx:
                        result_idx += 1
                    
                    if result_idx < len(chunk_results):
                        processed_conv.append({
                            "role": "user",
                            "content": chunk_results[result_idx][1]
                        })
                        result_idx += 1
                    else:
                        processed_conv.append(turn)  # Fallback to original if not found
                else:
                    processed_conv.append(turn)
            
            processed_conversations.append(processed_conv)

    # Update dataset with processed conversations
    dataset = dataset.add_column("conversations_compressed_token", processed_conversations)

    # Save the updated dataset
    dataset.save_to_disk("./english-multiturn-instruction-ichigo-tokens/")
    print("Dataset updated and saved successfully!")

if __name__ == "__main__":
    main()