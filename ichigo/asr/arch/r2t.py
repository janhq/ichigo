import torch
import torch.nn as nn
import whisper


class Rep2Text(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config["r2t"]
        self.whisper_name = config["whisper_name"]
        self.decoding_options = whisper.DecodingOptions(
            **self.config["decoding_options"]
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(self.whisper_name, device=device)
        del self.model.encoder

    def forward(self, dequantize_embed):
        return self.model.decode(dequantize_embed, self.decoding_options)
