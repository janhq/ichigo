import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper


class Speech2Rep(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config["s2r"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(config["whisper_name"], device=device)
        del self.model.decoder

    def forward(self, wav):
        mel = whisper.log_mel_spectrogram(wav)
        n_frames = mel.shape[-1]

        if n_frames > whisper.audio.N_FRAMES:
            padding = 0
            padded = mel[:, :, : whisper.audio.N_FRAMES]
            n_frames = whisper.audio.N_FRAMES
        else:
            padding = -n_frames % whisper.audio.N_FRAMES
            padded = F.pad(mel, (0, padding), value=-1.5)

        embs = self.model.encoder(padded)

        return embs, n_frames
