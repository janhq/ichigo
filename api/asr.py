from contextlib import asynccontextmanager
from enum import Enum
import threading
from typing import Annotated

import torch
import torchaudio
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel

from ichigo.asr import get_model


MODEL_LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load model to GPU at startup
    get_model()
    yield


app = FastAPI(
    title="Ichigo ASR API",
    description="API for audio transcription using Ichigo ASR",
    version="0.0.1",
    lifespan=lifespan,
)


class TranscriptionsModelName(str, Enum):
    ichigo = "ichigo"


@app.post("/v1/audio/transcriptions")
def _(
    file: Annotated[UploadFile, File()],
    model: Annotated[TranscriptionsModelName, Form()],
):
    """
    Transcribe an audio file uploaded via HTTP POST request

    Args:
        file: Audio file to transcribe
    """
    wav, sr = torchaudio.load(file.file)
    if wav.shape[0] > 1:  # convert multi-channel audio to mono
        wav = wav.mean(0, keepdim=True)

    model_ = get_model()
    MODEL_LOCK.acquire()
    wav = model_.preprocess(wav, sr)

    with torch.no_grad():
        embs, n_frames = model_.s2r(wav)
        dequantize_embed = model_.quantizer(embs, n_frames)
        output = model_.r2t(dequantize_embed)[0].text
    MODEL_LOCK.release()

    return dict(text=output)


@app.post("/s2r")
def _(file: UploadFile = File(...)):
    wav, sr = torchaudio.load(file.file)
    if wav.shape[0] > 1:  # convert multi-channel audio to mono
        wav = wav.mean(0, keepdim=True)

    model = get_model()
    MODEL_LOCK.acquire()
    with torch.no_grad():
        wav = model.preprocess(wav, sr)
        embs, n_frames = model.s2r(wav)
        token_ids = model.quantizer.quantize(embs, n_frames).squeeze(0).tolist()
    MODEL_LOCK.release()

    output = "".join(f"<|sound_{tok:04d}|>" for tok in token_ids)
    output = f"<|sound_start|>{output}<|sound_end|>"

    return dict(tokens=output)


class R2TRequest(BaseModel):
    tokens: str


@app.post("/r2t")
def _(req: R2TRequest):
    """tokens will have format <|sound_start|><|sound_0000|><|sound_end|>"""
    token_ids = [int(x) for x in req.tokens.split("|><|sound_")[1:-1]]
    token_ids = torch.tensor(token_ids).unsqueeze(0)

    model = get_model()
    MODEL_LOCK.acquire()
    with torch.no_grad():
        token_ids = token_ids.to(model.device)
        embeds = model.quantizer.dequantize(token_ids)
        output = model.r2t(embeds)[0].text
    MODEL_LOCK.release()

    return dict(text=output)
