import os
import tempfile
from pathlib import Path
from typing import Tuple

import torch
import torchaudio
from fastapi import HTTPException
from huggingface_hub import hf_hub_download

from common.utility.convert_utility import ConvertUtility
from common.utility.logger_utility import LoggerUtility
from services.audio.audio_interface import AudioInterface
from services.audio.audio_model import AudioFormat, AudioModel
from services.audio.implementation.audio_utils import CustomRQBottleneckTransformer
from services.model.implementation.model_service import ModelService
from variables.whisper_variable import WhisperVariable


class AudioService(AudioInterface):
    _audio_service = None

    @staticmethod
    def get_audio_service():
        if AudioService._audio_service is None:
            AudioService._audio_service = AudioService()
        return AudioService._audio_service

    def __init__(self,):
        self.model_service = ModelService()
        self.logger = LoggerUtility.get_logger()
        self.whisper_variable = WhisperVariable()
        self.available_backends = torchaudio.list_audio_backends()
        self.download_folder = Path(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))))/"downloads"
        self.load_vq_model()

    def load_vq_model(self):
        self.has_ffmpeg = "ffmpeg" in self.available_backends
        if not self.has_ffmpeg:
            self.logger.warning(
                "FFMPEG backend not available. Some formats may not be supported")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not os.path.exists(self.download_folder/self.whisper_variable.whisper_model_path):
            hf_hub_download(
                repo_id=self.whisper_variable.repo_id,
                filename=self.whisper_variable.whisper_model_path,
                local_dir=self.download_folder,
            )
        self.vq_model = CustomRQBottleneckTransformer.load_vq_only(
            self.download_folder / self.whisper_variable.whisper_model_path
        ).to(device)
        self.vq_model.load_encoder(device)
        self.vq_model.eval()

    def _get_best_backend(self, format: AudioFormat) -> str:
        """Determine the best backend for the given format"""
        supported_backends = AudioModel.FORMAT_BACKENDS[format]
        for backend in supported_backends:
            if backend in self.available_backends:
                return backend
        raise ValueError(f"No available backend supports format {format}")

    def load_audio(
        self,
        file_obj: bytes,
        format: AudioFormat,
        target_sr: int = 16000
    ) -> Tuple[torch.Tensor, int]:
        """
        Load audio from bytes object with format handling

        Args:
            file_obj: Audio file bytes
            format: Audio format enum
            target_sr: Target sample rate (default: 16000)

        Returns:
            Tuple[torch.Tensor, int]: Audio tensor and sample rate
        """
        try:
            # Get appropriate backend
            backend = self._get_best_backend(format)
            torchaudio.set_audio_backend(backend)
            self.logger.info(f"Using {backend} backend for {format} format")

            if format == AudioFormat.PCM:
                # Handle raw PCM
                wav = torch.frombuffer(file_obj, dtype=torch.int16)
                wav = wav.float() / 32768.0  # Normalize to [-1, 1]
                wav = wav.unsqueeze(0)  # Add channel dimension
                sr = target_sr
            else:
                # For formats that might need ffmpeg processing
                if os.name == "nt":  # for windows
                    wav, sr = torchaudio.load(io.BytesIO(file_obj))
                else:
                    with tempfile.NamedTemporaryFile(suffix=f".{format}") as temp_file:
                        # Write bytes to temporary file
                        temp_file.write(file_obj)
                        temp_file.flush()

                        # Load audio
                        wav, sr = torchaudio.load(temp_file.name)

            # Convert to mono if stereo
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)

            # Resample if needed
            if sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)
                sr = target_sr

            return wav, sr

        except Exception as e:
            self.logger.error(f"Error loading audio: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Error processing {format} audio: {str(e)}"
            )

    def get_format_info(self) -> dict:
        """Get information about supported formats"""
        supported_formats = {}
        for format in AudioFormat:
            try:
                backend = self._get_best_backend(format)
                supported_formats[format] = {
                    "supported": True,
                    "backend": backend
                }
            except ValueError:
                supported_formats[format] = {
                    "supported": False,
                    "backend": None
                }
        return supported_formats

    async def inference(self, req: AudioModel.Request) -> AudioModel.Response:
        try:
            wav, sr = self.load_audio(ConvertUtility.decode_base64(req.data), req.format)

            # Ensure we're using CUDA if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            wav = wav.to(device)

            # Generate tokens
            with torch.no_grad():
                codes = self.vq_model.encode_audio(wav)
                codes = codes[0].cpu().tolist()

            # Format result
            result = ''.join(f'<|sound_{num:04d}|>' for num in codes)

            return AudioModel.Response(tokens=f'<|sound_start|>{result}<|sound_end|>', sample_rate=sr, format=req.format)

        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing request: {str(e)}"
            )
