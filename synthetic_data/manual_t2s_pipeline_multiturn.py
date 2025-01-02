import os
import sys
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM, SamplingParams
from datasets import load_dataset

def main():
    print(sys.executable)

    # Initialize Ray
    ray.init()

    # Define the number of models (and GPUs)
    num_models = int(os.environ.get("NUM_GPUS", "6"))

    # Create a placement group
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

    # Load dataset
    dataset_name = "jan-hq/instruction-data-text-only-multiturn-filtered-for-tokenize-clean"
    # drop_column = ["tokens", "prompt", "text_prompt"]
    dataset = load_dataset(dataset_name, split="train")
    # dataset = dataset.remove_columns(drop_column)
    conversations_column_name = "conversations"

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

    futures = []
    batch_size = 240  # Adjust batch size as needed
    for i, actor in enumerate(actors):
        chunk = dataset[conversations_column_name][start_indices[i]:end_indices[i]]
        print(f"Actor {i}: Chunk size = {len(chunk)}, Start index = {start_indices[i]}, End index = {end_indices[i]}")
        for j in range(0, len(chunk), batch_size):
            # Correct the slicing to handle the last batch
            end_index = min(j + batch_size, len(chunk))  # Take the smaller value
            batch = chunk[j:end_index]  # Use end_index for the slice
            print(f" Batch: start = {j}, end = {end_index}, batch size = {len(batch)}")

            batch_prompts = []
            batch_indices = []

            for k, conv in enumerate(batch):
                for turn_idx, turn in enumerate(conv):
                    if turn["role"] == "user":
                        batch_prompts.append(prompt_template.format(text=turn["content"]))
                        batch_indices.append((i, j, k, turn_idx))

            futures.append((actor.generate.remote(batch_prompts), batch_indices))

    # Retrieve results and prepare for creating new column
    results = ray.get([f[0] for f in futures])
    indices = [f[1] for f in futures]
    flat_indices = []
    for sublist in indices:
        flat_indices.extend(sublist)

    results_by_actor = {i: [] for i in range(num_models)}
    for result_batch, batch_indices in zip(results, indices):
        for result, (actor_idx, _, _, _) in zip(result_batch, batch_indices):
            results_by_actor[actor_idx].append(result)

    flat_indices.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

    # Create new conversations and store them in a list
    new_conversations = []
    result_idx = 0
    for absolute_conv_idx in range(dataset_size):
        original_conversation = dataset[absolute_conv_idx][conversations_column_name]
        new_conversation = []
        for turn_idx, turn in enumerate(original_conversation):
            if turn["role"] == "user":
                # Find the corresponding result using flat_indices
                actor_idx, chunk_idx, conv_idx, stored_turn_idx = flat_indices[result_idx]
                
                # Ensure the indices match the current turn
                if stored_turn_idx == turn_idx and absolute_conv_idx == start_indices[actor_idx] + chunk_idx * batch_size + conv_idx:
                    new_turn = {"role": "user", "content": results_by_actor[actor_idx][result_idx]}
                    result_idx += 1
                else:
                    # This should ideally not happen if indexing is correct
                    print(f"Warning: Index mismatch at {absolute_conv_idx}, {turn_idx}. Using original content.")
                    new_turn = turn
            else:
                new_turn = turn

            new_conversation.append(new_turn)
        new_conversations.append(new_conversation)

    # Add the new conversations to the dataset
    dataset = dataset.add_column("conversations_compressed_token", new_conversations)

    # Save the updated dataset
    dataset.save_to_disk("./english-multiturn-instruction-ichigo-tokens/")
    print("Dataset updated and saved successfully!")

if __name__ == "__main__":
    main()