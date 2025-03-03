import gc
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from typing import Annotated

import torch
import torchaudio
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
from torch import Tensor, nn

from ichigo.asr.transcriber import IchigoASR

S2R_MODEL = None
R2T_MODEL = None
MODEL_LOCK = threading.Lock()

OLD_QUANTIZER_NAME = "ichigo-asr-2501-en-vi"
NEW_QUANTIZER_NAME = "whispervq-2405-en"
OLD_QUANTIZER = None
NEW_QUANTIZER = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAST_DEPLOYED_TIME = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load model to GPU at startup
    global S2R_MODEL, R2T_MODEL, OLD_QUANTIZER, NEW_QUANTIZER, LAST_DEPLOYED_TIME

    m = IchigoASR(NEW_QUANTIZER_NAME)
    NEW_QUANTIZER = m.quantizer
    S2R_MODEL = m.s2r
    R2T_MODEL = m.r2t
    del m

    # convert model to FP16. keep LayerNorm in FP32
    for model in [S2R_MODEL, R2T_MODEL]:
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                m.half()

    m = IchigoASR(OLD_QUANTIZER_NAME)
    OLD_QUANTIZER = m.quantizer
    del m

    gc.collect()
    torch.cuda.empty_cache()

    LAST_DEPLOYED_TIME = datetime.now()
    yield


SAMPLE_RATE = 16_000
_RESAMPLERS = dict()


def preprocess_audio(x: Tensor, fs: int) -> Tensor:
    x = x.mean(0, keepdim=True)  # convert multi-channel audio to mono
    if fs == SAMPLE_RATE:
        return x

    # cache resampler
    if fs not in _RESAMPLERS:
        _RESAMPLERS[fs] = torchaudio.transforms.Resample(fs, SAMPLE_RATE)
    return _RESAMPLERS[fs](x)


def _find_split_point(wav: torch.Tensor, start_idx: int, end_idx: int) -> int:
    """Find the best point to split audio by looking for silence or low amplitude.
    Args:
        wav: Audio tensor [1, T]
        start_idx: Start index of search region
        end_idx: End index of search region
    Returns:
        Index of best splitting point
    """
    segment = wav.abs().squeeze(0)[start_idx:end_idx].cpu().numpy()

    # Calculate RMS energy in small windows
    window_size = 1600  # 100ms windows at 16kHz
    energies = []
    for i in range(0, len(segment) - window_size, window_size):
        window = segment[i : i + window_size]
        energy = (window**2).mean() ** 0.5
        energies.append((i + start_idx, energy))

    quietest_idx, _ = min(energies, key=lambda x: x[1])
    return quietest_idx


app = FastAPI(
    title="Ichigo ASR API",
    description="API for audio transcription using Ichigo ASR",
    version="0.0.1",
    lifespan=lifespan,
)


def format_time_ago(prev_datetime: datetime) -> str:
    time_difference = datetime.now() - prev_datetime
    days = time_difference.days
    hours = time_difference.seconds // 3600
    minutes = (time_difference.seconds // 60) % 60
    seconds = time_difference.seconds % 60
    if days > 0:
        return f"{days}d {hours}h ago"
    elif hours > 0:
        return f"{hours}h {minutes}m ago"
    elif minutes > 0:
        return f"{minutes}m {seconds}s ago"
    else:
        return f"{seconds}s ago"


@app.get("/health/server-status")
async def _():
    date_fmt = "%Y-%m-%d %H:%M:%S"
    time_ago = format_time_ago(LAST_DEPLOYED_TIME)
    return dict(
        status="OK",
        server_time=datetime.now().strftime(date_fmt),
        last_deployed_time=f"{LAST_DEPLOYED_TIME.strftime(date_fmt)} ({time_ago})",
        message="OK",
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
    wav = preprocess_audio(wav, sr)

    CHUNK_SIZE = SAMPLE_RATE * 20  # 20 seconds
    OVERLAP_SIZE = SAMPLE_RATE * 1  # 1 second

    chunks = []
    i = 0
    while i < wav.shape[1]:
        if i + CHUNK_SIZE >= wav.shape[1]:
            # Handle the last chunk
            chunks.append(wav[:, i:])
            break

        # Find the best split point in the overlap region
        search_start = i + CHUNK_SIZE - OVERLAP_SIZE
        search_end = min(i + CHUNK_SIZE + OVERLAP_SIZE, wav.shape[1])
        split_point = _find_split_point(wav, search_start, search_end)

        # Extract chunk up to the split point
        chunks.append(wav[:, i:split_point])
        i = split_point

    outputs = []
    with torch.no_grad(), MODEL_LOCK:
        for chunk in chunks:
            # using new quantizer. better for Vietnamese.
            embs, n_frames = S2R_MODEL(chunk.to(DEVICE))
            dequantize_embed = NEW_QUANTIZER(embs, n_frames)
            outputs.append(R2T_MODEL(dequantize_embed)[0].text)
    output = " ".join(outputs)

    return dict(text=output)


@app.post("/s2r")
def _(file: UploadFile = File(...)):
    wav, sr = torchaudio.load(file.file)
    wav = preprocess_audio(wav, sr)

    # using old quantizer, for compatibility with Ichigo-LLM v0.4
    with torch.no_grad(), MODEL_LOCK:
        embs, n_frames = S2R_MODEL(wav.to(DEVICE))
        token_ids = NEW_QUANTIZER.quantize(embs, n_frames).squeeze(0).tolist()

    output = "".join(f"<|sound_{tok:04d}|>" for tok in token_ids)
    output = f"<|sound_start|>{output}<|sound_end|>"

    return dict(model=NEW_QUANTIZER_NAME, tokens=output)


class R2TRequest(BaseModel):
    tokens: str


@app.post("/r2t")
def _(req: R2TRequest):
    """tokens will have format <|sound_start|><|sound_0000|><|sound_end|>"""
    token_ids = [int(x) for x in req.tokens.split("|><|sound_")[1:-1]]
    token_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)

    # NOTE: we will get CUDA device-side assert if index > codebook size.
    # hence, we have to clip it here.
    token_ids = token_ids.clip(max=NEW_QUANTIZER.vq_codes - 1)

    # using old quantizer, for compatibility with Ichigo-LLM v0.4
    with torch.no_grad(), MODEL_LOCK:
        embeds = NEW_QUANTIZER.dequantize(token_ids.to(DEVICE))
        output = R2T_MODEL(embeds)[0].text

    return dict(model=NEW_QUANTIZER_NAME, text=output)
