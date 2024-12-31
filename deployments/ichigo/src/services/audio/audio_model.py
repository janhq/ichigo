from dataclasses import dataclass, asdict
from enum import Enum
from pydantic import BaseModel
class AudioFormat(str, Enum):
        WAV = "wav"    # Supported by both backends
        MP3 = "mp3"    # Supported by ffmpeg
        FLAC = "flac"  # Supported by both
        AAC = "aac"    # Supported by ffmpeg
        OGG = "ogg"    # Supported by ffmpeg
        OPUS = "opus"  # Supported by ffmpeg
        PCM = "pcm"    # Raw PCM data

class WhisperRequest(BaseModel):
    data: str
    format: AudioFormat = "wav"
    

class FishSpeechRequest:
    text: str
    normalize: bool = True
    format: str = "wav"
    latency: str = "balanced"
    max_new_tokens: int = 4096
    chunk_length: int = 200
    repetition_penalty: float = 1.5
    streaming: bool = False
    
class AudioModel:
        
    class AudioCompletionRequest(BaseModel):
        messages: list[dict[str, str]]
        input_audio: WhisperRequest
        model: str = "ichigo:8b-gguf-q4km"
        stream: bool = True
        temperature: float = 0.7
        top_p: float = 0.9
        max_tokens: int = 2048
        presence_penalty: float = 0.0
        frequency_penalty: float = 0.0
        stop: list[str] = ["<|eot_id|>"]
        output_audio: bool = True
        
    @dataclass()
    class Response:
        audio: str
        text: str
        messages: list[dict[str, str]]
