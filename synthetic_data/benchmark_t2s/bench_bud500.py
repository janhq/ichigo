from datasets import load_dataset
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from fastprogress import progress_bar
from ichigo.synthetic_data.benchmark_t2s.wer_metrics import *
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import pandas as pd
import whisper
import re
import matplotlib.pyplot as plt
from audio_tokenizer import IchigoQuantizer

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
libspeech_test = load_dataset("linhtran92/viet_bud500", split='test[:1000]')
# prompt = "You are a professional transcriber, fluent in understanding noisy audio recordings. You are tasked with transcribing a recording where the audio quality is very noisy and potentially monotonous. Despite the challenging input, your transcription should be as clear and accurate as possible."
prompt=None
audio_tokenizer = IchigoQuantizer(language="vi", prompt=prompt)
ichigo_model = audio_tokenizer.ichigo_model

tok_t2s = AutoTokenizer.from_pretrained("jan-hq/Ichigo-llama3.2-base-1B-T2S-2560c-epoch-2")
t2s = AutoModelForCausalLM.from_pretrained("jan-hq/Ichigo-llama3.2-base-1B-T2S-2560c-epoch-2").to("cuda")

def whispervq_tokenizer(audio: tuple) -> list[int]:
    wav, sr = audio['array'], audio['sampling_rate']
    print(len(wav))
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    with torch.no_grad():
        # convert numpy to torch
        wav = torch.from_numpy(wav).float().unsqueeze(0)
        print(wav.shape)
        codes = audio_tokenizer.encode(
                (wav, sr)
            )
    return torch.tensor(codes)    

def Text_to_Sementic(prompt: str) -> list[int]:
    text = f"<|start_header_id|>user<|end_header_id|>\n\n<|reserved_special_token_69|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    model_inputs = tok_t2s([text], return_tensors="pt").to("cuda")
    generated_ids = t2s.generate(
        **model_inputs,
        max_new_tokens=1024,
        do_sample=False,
        temperature=0.0,
        stop_strings=["<|sound_end|>"],
        repetition_penalty=1.0,
        tokenizer=tok_t2s,
        
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    sound_tokens = tok_t2s.batch_decode(generated_ids, skip_special_tokens=False)[0]
    sound_tokens = sound_tokens.replace("<|eot_id|>", "")
    sound_tokens = decompress_sound_tokens(sound_tokens)
    codes = extract_sound_codes(sound_tokens)
    # convert to torch from list
    return torch.tensor(codes)
def Text_to_Sementic_instruct(prompt: str):
    text = f"<|start_header_id|>user<|end_header_id|>\n\n<|reserved_special_token_69|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    model_inputs = tok_t2s([text], return_tensors="pt").to("cuda")
    generated_ids = t2s.generate(
        **model_inputs,
        max_new_tokens=2048,
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.0,
        
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    sound_tokens = tok_t2s.batch_decode(generated_ids, skip_special_tokens=False)[0]
    sound_tokens = sound_tokens.replace("<|eot_id|>", "")
    sound_tokens = decompress_sound_tokens(sound_tokens)
    codes = extract_sound_codes(sound_tokens)
    # convert to torch from list
    return torch.tensor(codes)
stats_t2s = WERStats()
for audio, text in zip(libspeech_test['audio'], libspeech_test['transcription']):
    wav = torch.from_numpy(audio['array']).float()
    codes_t2s = Text_to_Sementic(f"{text}.".lower())
    dequantize_embed_t2s = ichigo_model.dequantize(codes_t2s).to(ichigo_model.whmodel[0].device)
    text_t2s = ichigo_model.whmodel[0].decode(dequantize_embed_t2s, ichigo_model.decoding_options)[0].text
    print(text)
    print(text_t2s)
    print("*"*20)
    # text to sementic
    diff = stats_t2s.push_sample(wav, text, text_t2s)
    last_diff = diff.alignments[0][-1]
    # stats_whispervq.push(hallucination = last_diff.type == 'insert' and last_diff.hyp_end_idx - last_diff.hyp_start_idx > 3)
stats_t2s = stats_t2s.df()
# drop column mer, wil, wip
stats_t2s = stats_t2s.drop(columns=['mer', 'wil', 'wip'])
stats_t2s = stats_t2s.rename(columns={'text': 'Ichigo T2S + Whisper decoder'})
stats_t2s.to_csv("stats.csv")
t2s_wer = round(stats_t2s['wer'].mean() * 100 ,2)
print(f'Text to Sementic WER: {t2s_wer}%')


