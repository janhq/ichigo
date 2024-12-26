import io
import os
from pathlib import Path
from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import whisper
from huggingface_hub import hf_hub_download
from whisper.model import AudioEncoder, ModelDimensions
from whisperspeech.vq_stoks import RQBottleneckTransformer, Tunables

from services.model.implementation.model_service import ModelService
from variables.whisper_variable import WhisperVariable


class CustomWhisperEncoder(nn.Module):
    """
    Lightweight wrapper that only loads the AudioEncoder part of Whisper
    """

    def __init__(self, name: str, device: str = None, download_root: str = None, in_memory: bool = False,):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_variable = WhisperVariable()
        checkpoint = self.download(download_root, name, in_memory, device)
        dims = ModelDimensions(**checkpoint["dims"])
        self.encoder = AudioEncoder(
            dims.n_mels,
            dims.n_audio_ctx,
            dims.n_audio_state,
            dims.n_audio_head,
            dims.n_audio_layer,
        )

        self.encoder.load_state_dict(checkpoint["model_state_dict"])

        if device:
            self.to(device)

        self.eval()

    def download(self, download_root: str, name: str, in_memory: bool, device: str) -> Any:
        if download_root is None:
            download_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))))/"downloads"

        if name in self.whisper_variable.HF_MODELS:
            checkpoint_file = ModelService.download_encoder(
                self.whisper_variable.HF_MODELS[name], download_root, in_memory)
        elif os.path.isfile(name):
            checkpoint_file = open(name, "rb").read() if in_memory else name
        else:
            raise RuntimeError(
                f"Model {name} not found available models={self.available_models()}"
            )

        # Load weights
        with (
            io.BytesIO(checkpoint_file) if in_memory else open(
                checkpoint_file, "rb")
        ) as fp:
            checkpoint = torch.load(fp, map_location=device)
        del checkpoint_file
        return checkpoint

    def available_models(self) -> List[str]:
        """Returns the names of available models"""
        return list(self.whisper_variable.HF_MODELS.keys())

    def forward(self, mel: torch.Tensor):
        return self.encoder(mel)


class CustomRQBottleneckTransformer(RQBottleneckTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def load_vq_only(cls, ref="collabora/spear-tts-pytorch:whisper-vq-stoks-medium-en+pl.model",
                     repo_id=None, filename=None, local_filename=None):
        if repo_id is None and filename is None and local_filename is None:
            if ":" in str(ref):
                repo_id, filename = ref.split(":", 1)
            else:
                local_filename = ref
        if not local_filename:
            local_filename = hf_hub_download(
                repo_id=repo_id, filename=filename)

        # Load the spec
        spec = torch.load(local_filename)

        # Create instance with minimal required components
        instance = cls(**spec['config'], tunables=Tunables(**
                       Tunables.upgrade(spec.get('tunables', {}))))

        # Load only necessary state dict entries
        required_components = {
            'rq', 'mlp', 'mlp_ln'
        }
        filtered_state_dict = {
            k: v for k, v in spec['state_dict'].items()
            if any(k.startswith(comp) for comp in required_components)
        }

        instance.load_state_dict(filtered_state_dict, strict=False)
        instance.eval()
        return instance

    def load_encoder(self, device=None):
        if self.whmodel is not None:
            return
        device = device or self.device
        # Use our custom encoder-only model
        if self.whmodel is None:
            encoder = CustomWhisperEncoder(
                self.whisper_model_name, device=device)
            self.whmodel = encoder
        multilingual = not self.whisper_model_name.endswith('.en')
        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual)

    def optimzed_encode_mel(self, mel):
        assert len(
            mel.shape) == 3, "invalid mel spectrogram shape, expect (batch,chn,time)"
        self.load_encoder()
        n = mel.shape[-1]
        if n > whisper.audio.N_FRAMES:
            padding = 0
            padded = mel[:, :, :whisper.audio.N_FRAMES]
        else:
            padding = -n % whisper.audio.N_FRAMES
            padded = F.pad(mel, (0, padding), value=-1.5)
        # .to(self.whmodel[0].device))#[:,:n//2]
        embs = self.whmodel.encoder(padded)
        stoks = self.quantize(embs)
        if self.tunables.mask_embs:
            return stoks[:, :n//2//self.downsample]
        else:
            return stoks
    # overide

    def encode_audio(self, audio):
        if isinstance(audio, str):
            x, sr = torchaudio.load(audio)
            x = torchaudio.transforms.Resample(sr, 16000)(x)[0]
            audio = x.unsqueeze(0)
        return self.optimzed_encode_mel(self.log_mel_spectrogram(audio).to(self.device))
