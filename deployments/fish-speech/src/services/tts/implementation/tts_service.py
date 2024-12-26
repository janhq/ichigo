import io
import os
from pathlib import Path
import torch
import soundfile as sf
from http import HTTPStatus
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from services.tts.tts_model import TTSModel, ServeReferenceAudio
from services.tts.tts_interface import TTSInterface
from variables.fish_speech_variable import FishSpeechVariable
from common.utility.convert_utility import ConvertUtility
from tools.server.model_manager import ModelManager

from tools.server.api_utils import (
    get_content_type,
    inference_async,
)
from tools.server.inference import inference_wrapper as inference
class TTSService(TTSInterface):
    _tts_service = None

    @staticmethod
    def get_tts_service():
        if TTSService._tts_service is None:
            TTSService._tts_service = TTSService()
        return TTSService._tts_service

    def __init__(self):
        self.fish_speech_variable = FishSpeechVariable()
        self.main_directory = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.realpath(__file__)))))))
        self.max_text_length = self.fish_speech_variable.max_text_length
        self.load_model()
        self.load_reference()

    def load_reference(self,):
        reference_text = ""
        reference_byte = b''
        with open(self.main_directory/"references"/self.fish_speech_variable.reference_name/"ref.lab", "r") as text:
            reference_text = text.read()
        with open(self.main_directory/"references"/self.fish_speech_variable.reference_name/"ref.mp3", "rb") as byte_:
            reference_byte = byte_.read()
        self.reference_audio = ServeReferenceAudio(
            audio=reference_byte, text=reference_text)

    def load_model(self,):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compile = False
        if device == "cuda" and os.name != 'nt':
            os.environ["PATH"] = "/usr/bin/:" + os.environ.get("PATH", "")
            compile = True

        self.model_manager = ModelManager(
            mode="tts", device=device, decoder_config_name=self.fish_speech_variable.decoder_config_name,
            half=True, compile=compile, asr_enabled=False,
            llama_checkpoint_path=self.main_directory /
            "downloads"/self.fish_speech_variable.model_name,
            decoder_checkpoint_path=self.main_directory /
            "downloads"/self.fish_speech_variable.model_name /
            self.fish_speech_variable.decoder_checkpoint)  # type: ignore

    async def inference(self, req: TTSModel.Request) -> TTSModel.Response:
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

            return TTSModel.Response(audio=ConvertUtility.encode_to_base64(buffer.getvalue()),sample_rate=44100)
