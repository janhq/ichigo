import os
import tempfile
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ichigo.asr import transcribe, get_stoks

app = FastAPI(
    title="Ichigo ASR API",
    description="API for audio transcription using Ichigo ASR",
    version="0.0.1",
)


@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe an audio file uploaded via HTTP POST request

    Args:
        file: Audio file to transcribe
    """
    if not file.filename.lower().endswith((".wav", ".mp3", ".flac")):
        raise HTTPException(400, "Unsupported file format")

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(file.filename).suffix
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = transcribe(tmp_path, output_path=None)
        transcript, metadata = result
        response = {"transcript": transcript, **metadata}
        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(500, str(e))

    finally:
        os.unlink(tmp_path)


@app.post("/get_stoks/")
async def get_semantic_tokens(file: UploadFile = File(...)):
    """
    Get semantic tokens from an audio file uploaded via HTTP POST request

    Args:
        file: Audio file to process
    """
    if not file.filename.lower().endswith((".wav", ".mp3", ".flac")):
        raise HTTPException(400, "Unsupported file format")

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(file.filename).suffix
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = get_stoks(tmp_path)
        response = {
            "semantic_tokens": result.tolist() if torch.is_tensor(result) else result
        }
        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(500, str(e))

    finally:
        os.unlink(tmp_path)


@app.get("/")
async def root():
    return {
        "message": "Welcome to Ichigo ASR API. Use /transcribe/ for text transcription or /get_stoks/ for semantic tokens."
    }
