from datasets import load_dataset, load_from_disk
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from fastprogress import progress_bar
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import pandas as pd
import whisper
import re
import matplotlib.pyplot as plt
from audio_tokenizer import IchigoQuantizer
ds = load_from_disk("/home/jan/BachVD/ichigo/synthetic_data/VTSNLP-instruct-filtered-ichigo-tokens")

def extract_sound_codes(text):
    # Find all patterns matching <|sound_XXXX|> where X is a digit
    text = text.replace("<|sound_start|>", "").replace("<|sound_end|>", "")
    pattern = r'<\|sound_(\d{4})\|>'
    
    # Extract all matches and convert to integers
    codes = [int(match) for match in re.findall(pattern, text)]
    
    return codes
def decompress_sound_tokens(compressed_str: str) -> str:
    """
    Convert compressed sound tokens with duration back to normal sound tokens.
    Example: "<|sound_start|><|duration_02|><|sound_0194|>" -> 
             "<|sound_start|><|sound_0194|><|sound_0194|>"
    
    Args:
        compressed_str (str): Compressed string with duration tokens
        
    Returns:
        str: Decompressed string with repeated sound tokens
    """
    # Split the tokens
    tokens = compressed_str.strip('<>').split('|><|')
    decompressed_tokens = []
    
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        # Handle special tokens
        if token in ['sound_start', 'sound_end']:
            decompressed_tokens.append(token)
            i += 1
            continue
            
        # Check if current token is a duration token
        duration_match = re.match(r'duration_(\d{2})', token)
        if duration_match and i + 1 < len(tokens):
            # Get duration count and next sound token
            count = int(duration_match.group(1))
            sound_token = tokens[i + 1]
            # Repeat the sound token 'count' times
            decompressed_tokens.extend([sound_token] * count)
            i += 2  # Skip both duration and sound token
        else:
            # If not a duration token, must be a single sound token
            decompressed_tokens.append(token)
            i += 1
    
    return '<' + '|><|'.join(decompressed_tokens) + '>'
sound_tokens = ds[3]['compressed_prompt']
print("groud truth: ", ds[3]['input'])
prompt = None
audio_tokenizer = IchigoQuantizer(language="vi", prompt=prompt)
ichigo_model = audio_tokenizer.ichigo_model
sound_tokens = decompress_sound_tokens(sound_tokens)
codes = extract_sound_codes(sound_tokens)
codes = torch.tensor(codes)
dequantize_embed_t2s = ichigo_model.dequantize(codes).to(ichigo_model.whmodel[0].device)
text_t2s = ichigo_model.whmodel[0].decode(dequantize_embed_t2s, ichigo_model.decoding_options)[0].text
print(text_t2s)