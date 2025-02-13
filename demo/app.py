import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import gradio as gr
import torch
import torchaudio
import whisper
from transformers import pipeline

from ichigo.asr import transcribe

device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model_name = "medium"
language = "demo"

whisper_model = whisper.load_model(whisper_model_name)
whisper_model.to(device)

phowhisper = pipeline(
    "automatic-speech-recognition",
    model="vinai/PhoWhisper-medium",
    device=device,
)


# ! Ichigo-ASR
def transcribe_ichigo(inputs):
    if inputs is None:
        raise gr.Error(
            "No audio file submitted! Please upload or record an audio file before submitting your request."
        )
    text = transcribe(inputs)
    return text[0]


# ! Whisper Medium
def transcribe_whisper(inputs):
    if inputs is None:
        raise gr.Error(
            "No audio file submitted! Please upload or record an audio file before submitting your request."
        )
    wav1, sr = torchaudio.load(inputs)
    if sr != 16000:
        wav1 = torchaudio.functional.resample(wav1, sr, 16000)
    audio_sample = wav1.squeeze().float().numpy()
    transcribe = whisper_model.transcribe(
        audio_sample,
        task="transcribe",
        fp16=False,
    )
    return transcribe["text"]


# ! PhoWhisper Medium
def transcribe_phowhisper(inputs):
    wav2, sr = torchaudio.load(inputs)
    if sr != 16000:
        wav2 = torchaudio.functional.resample(wav2, sr, 16000)
    audio_sample = wav2.squeeze().float().numpy()
    return phowhisper(audio_sample)["text"]


with gr.Blocks(title="Ichigo-ASR", theme="allenai/gradio-theme") as interface:
    gr.Markdown("# 🍰 Ichigo-ASR")
    gr.Markdown(
        "Ichigo-ASR is a compact (22M parameters), open-source quantizer for the Whisper-medium model, designed to enhance performance on low-resource languages with minimal impact on its original English capabilities. Unlike models that output continuous embeddings, Ichigo-ASR compresses speech into discrete tokens, making it more compatible with large language models (LLMs) for immediate speech understanding. This quantized version of Whisper-medium has been trained on over ~400 hours of English data and ~1000 hours of Vietnamese data."
    )
    gr.Markdown(
        "Ichigo-ASR is a key component of the Ichigo v0.5 family. For more details, please refer to our official [blog post.](https://huggingface.co/homebrewltd/Ichigo-whisper-v0.1)"
    )
    gr.Markdown(
        "NOTE: Different from the original Whisper model, Ichigo-ASR can support audio longer than 30 seconds."
    )

    with gr.Row():
        # Audio input section
        audio_input = gr.Audio(
            sources=["microphone", "upload"], type="filepath", label="Audio Input"
        )

    with gr.Row():
        # Ichigo Model Column
        with gr.Column():
            gr.Markdown("### Ichigo-ASR")
            ichigo_output = gr.TextArea(
                label="Transcription",
                placeholder="Transcription will appear here...",
                lines=8,
            )
            ichigo_btn = gr.Button("Transcribe with Ichigo")
            ichigo_btn.click(
                fn=transcribe_ichigo, inputs=audio_input, outputs=ichigo_output
            )

        # Whisper Model Column
        with gr.Column():
            gr.Markdown(f"### Whisper {whisper_model_name.capitalize()}")
            whisper_output = gr.TextArea(
                label="Transcription",
                placeholder="Transcription will appear here...",
                lines=8,
            )
            whisper_btn = gr.Button("Transcribe with Whisper")
            whisper_btn.click(
                fn=transcribe_whisper, inputs=audio_input, outputs=whisper_output
            )

        # PhoWhisper Model Column
        with gr.Column():
            gr.Markdown("### PhoWhisper Medium")
            phowhisper_output = gr.TextArea(
                label="Transcription",
                placeholder="Transcription will appear here...",
                lines=8,
            )
            phowhisper_btn = gr.Button("Transcribe with PhoWhisper")
            phowhisper_btn.click(
                fn=transcribe_phowhisper, inputs=audio_input, outputs=phowhisper_output
            )

    custom_css = """
        .gradio-container {
            font-family: 'Helvetica Neue', Arial, sans-serif;
        }
        .output-text {
            font-size: 16px;
            padding: 10px;
            border-radius: 4px;
            transition: all 0.3s ease;
        }
        .transcribe-button {
            margin-top: 10px;
        }
    """
    gr.HTML(f"<style>{custom_css}</style>")

if __name__ == "__main__":
    interface.queue()
    interface.launch(server_name="0.0.0.0", server_port=7860, share=True)
