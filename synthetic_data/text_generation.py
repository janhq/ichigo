from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

# --- Configuration ---
DATASET_NAME = "jan-hq/Ichigo-VTSNLP-instruct-filtered-clean"  
TEXT_COLUMN = "text_prompt"  
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"
NEW_COLUMN_NAME = "synthetic_answer"
HUB_REPO_ID = "Ichigo-VTSNLP-instruct-filtered-synthetic/batch-1"  
BATCH_SIZE = 240

# --- Load Dataset ---
try:
    dataset = load_dataset(DATASET_NAME, split="train")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# --- Initialize Tokenizer and LLM ---
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    llm = LLM(model=MODEL_NAME, gpu_memory_utilization=0.95, max_num_seqs=480, tensor_parallel_size=2)
except Exception as e:
    print(f"Error initializing model or tokenizer: {e}")
    exit()

# --- Sampling Parameters (Qwen2.5-7B-Instruct defaults) ---
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.8,
    repetition_penalty=1.05,
    max_tokens=4096,
)

# --- Prepare Prompts Function ---
def prepare_prompts(batch):
    prompts = []
    for text_prompt in batch:
        messages = [
            {
                "role": "system",
                "content": "Bạn là một trợ lí ảo hữu ích, hãy trả lời các câu hỏi một cách đầy đủ và chính xác nhất.",
            },
            {"role": "user", "content": text_prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)
    return prompts

# --- Generate Synthetic Data and Add to Dataset ---
synthetic_answers = []
try:
    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Generating synthetic data"):
        batch = dataset[i : i + BATCH_SIZE][TEXT_COLUMN]
        prompts = prepare_prompts(batch)

        outputs = llm.generate(prompts, sampling_params)

        for output in outputs:
            synthetic_answers.append(output.outputs[0].text)
except Exception as e:
    print(f"Error during synthetic data generation: {e}")

# --- Add the new column to the dataset ---
data = dataset.add_column(NEW_COLUMN_NAME, synthetic_answers)

# --- Push to Hugging Face Hub ---
try:
    data.save_to_disk(f"./{HUB_REPO_ID}/")
    print(f"Dataset with synthetic answers pushed to: {HUB_REPO_ID}")
except Exception as e:
    print(f"Error pushing dataset to the Hugging Face Hub: {e}")
    print("Make sure you are logged in to the Hugging Face CLI (`huggingface-cli login`) and have write access to the specified repository.")

print("Process completed.")