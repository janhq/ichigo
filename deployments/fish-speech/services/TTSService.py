from http import HTTPStatus
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from tools.schema import ServeTTSRequest, ServeReferenceAudio
from tools.server.model_manager import ModelManager
from tools.server.api_utils import (
    buffer_to_async_generator,
    get_content_type,
    inference_async,
)
from tools.server.inference import inference_wrapper as inference
import torch
import os
import io
from pathlib import Path
import soundfile as sf
from utils.utils import encode_audio_to_base64

class TTSService:
    def __init__(self, ):
        

    def inference(self, req: ServeTTSRequest):
        if req.reference_id is None and len(req.references) == 0:
            req.references.append(self.reference_audio)
        engine = self.model_manager.tts_inference_engine
        sample_rate = engine.decoder_model.spec_transform.sample_rate

        # Check if the text is too long
        if self.max_text_length > 0 and len(req.text) > self.max_text_length:
            raise HTTPException(
                HTTPStatus.BAD_REQUEST,
                detail=f"Text is too long, max length is {self.max_text_length}",
            )

        # Check if streaming is enabled
        if req.streaming and req.format != "wav":
            raise HTTPException(
                HTTPStatus.BAD_REQUEST,
                detail="Streaming only supports WAV format",
            )

        # Perform TTS
        if req.streaming:
            return StreamingResponse(
                content=inference_async(req, engine),
                headers={
                    "Content-Disposition": f"attachment; filename=audio.{req.format}",
                },
                media_type=get_content_type(req.format),
            )
        else:
            fake_audios = next(inference(req, engine))
            buffer = io.BytesIO()
            sf.write(
                buffer,
                fake_audios,
                sample_rate,
                format=req.format,
            )

            return {"audio": encode_audio_to_base64(buffer.getvalue())}
            """
             StreamingResponse(
                content=buffer_to_async_generator(buffer.getvalue()),
                headers={
                    "Content-Disposition": f"attachment; filename=audio.{req.format}",
                },
                media_type=get_content_type(req.format),
            )
"""

_tts_service = None


def get_tts_service():
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service
