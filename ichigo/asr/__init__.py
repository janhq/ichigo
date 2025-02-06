from typing import Union

import torch

from ichigo.asr.transcriber import IchigoASR

_default_model = None

def get_model(**kwargs) -> IchigoASR:
    """Get or create default ASR model instance"""
    global _default_model
    if _default_model is None:
        _default_model = IchigoASR(**kwargs)
    return _default_model


def transcribe(
    input_path: str,
    output_path: str = None,
    extensions: tuple = (".wav", ".mp3", ".flac"),
    **kwargs,
) -> Union[str, dict[str, str]]:
    """Quick transcription function using default model for file or folder"""
    model = get_model(**kwargs)
    return model.transcribe(input_path, output_path, extensions)


def get_stoks(input_path: str, **kwargs) -> torch.Tensor:
    """Get STOKS for a single file using default model"""
    model = get_model(**kwargs)
    return model.get_stoks(input_path)
