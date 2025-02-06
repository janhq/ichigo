import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import ResidualVQ

from ichigo.asr.arch.layers import LayerNorm, ResidualAttentionBlock


class Quantizer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        qconfig = config["quantizer"]

        # Model Architecture
        self.n_head = qconfig["n_head"]
        self.head_width = qconfig["head_width"]
        self.width = self.n_head * self.head_width
        self.ffn_mult = qconfig["ffn_mult"]
        self.depth = qconfig["depth"]
        self.downsample = qconfig["downsample"]

        # Vector Quantization
        self.vq_codes = qconfig["vq_codes"]
        self.num_quantizers = qconfig["num_quantizers"]
        self.codebook_dim = qconfig["codebook_dim"]
        self.threshold_ema_dead_code = qconfig["threshold_ema_dead_code"]
        self.use_cosine_sim = qconfig["use_cosine_sim"]
        self.codebook_decay = qconfig["codebook_decay"]
        self.commitment_weight = qconfig["commitment_weight"]
        self.query_mult = qconfig["query_mult"]
        self.q_depth = qconfig["q_depth"]

        # Features
        self.rope = qconfig["rope"]
        self.mask_embs = qconfig["mask_embs"]
        self.downsample_conv = qconfig["downsample_conv"]
        self.downsample_mean = qconfig["downsample_mean"]

        #! HARDCODE values
        self.stoks_len = 1500 // self.downsample
        self.positions = torch.arange(0, 1500, dtype=torch.long)

        # Initialize components
        self._init_model_components()

    def _init_model_components(self):
        # Quantizer
        n_mlp = self.width * self.ffn_mult
        self.mlp = nn.Sequential(
            nn.Linear(self.width, n_mlp), nn.GELU(), nn.Linear(n_mlp, self.width)
        )
        self.mlp_ln = LayerNorm(self.width)

        if self.downsample_conv:
            self.downsample_conv = nn.Conv1d(
                self.width, self.width, kernel_size=3, stride=self.downsample, padding=1
            )
        else:
            self.downsample_conv = None

        if self.mask_embs:
            vq_codes = self.vq_codes + 1

        self.rq = ResidualVQ(
            dim=self.width,
            codebook_size=vq_codes,
            decay=self.codebook_decay,
            commitment_weight=self.commitment_weight,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
            use_cosine_sim=self.use_cosine_sim,
            codebook_dim=self.codebook_dim,
            num_quantizers=self.num_quantizers,
        )

        # Transformer
        qk_scale = self.query_mult * 8 / math.sqrt(self.head_width)

        self.positional_embedding = nn.Embedding(1500, self.width)
        self._out_blocks = nn.Sequential(
            *[
                ResidualAttentionBlock(
                    self.width,
                    self.n_head,
                    qk_scale=qk_scale,
                    ffn_mult=self.ffn_mult,
                    rope=self.rope,
                )
                for _ in range(self.depth)
            ]
        )
        self.ln_post = LayerNorm(self.width)

    def out_blocks(self, x):
        for l in self._out_blocks:
            x = l(x, self.positions)
        return x

    def downsample_embeddings(self, x):
        if self.downsample_conv is not None:
            return x[:, :: self.downsample] + self.downsample_conv(
                x.transpose(-1, -2)
            ).transpose(-2, -1)
        elif self.downsample_mean:
            bs, slen, depth = x.shape
            return x.reshape(bs, slen // self.downsample, self.downsample, depth).mean(
                -2
            )
        else:
            return x[:, :: self.downsample]

    @torch.no_grad()
    def quantize(self, embs, n_frames):
        x = self.downsample_embeddings(embs)
        x = x + self.mlp(self.mlp_ln(x))

        _, stoks, _ = self.rq(x)
        stoks = stoks.squeeze(-1)

        if self.mask_embs:
            return stoks[:, : n_frames // 2 // self.downsample]
        else:
            return stoks

    def dequantize(self, stoks):
        stoks = stoks.squeeze()

        # Dequantize
        assert self.q_depth == 1
        assert len(stoks.shape) == 1, "batch processing is not supported"

        stoks = F.pad(
            stoks,
            (0, self.stoks_len - stoks.shape[-1]),
            value=2048 if self.mask_embs else 0,  # TODO: DONT HARDCODE
        )

        x = self.rq.layers[0]._codebook.embed[0, stoks.to(torch.long).view(-1)]
        x = x.repeat_interleave(self.downsample, -2)

        project_out = (
            getattr(self.rq, "project_out", None) or self.rq.layers[0].project_out
        )
        x = project_out(x).unsqueeze(0)

        positions = torch.arange(0, x.shape[-2], dtype=torch.long, device=x.device)
        x = x + self.positional_embedding(positions)

        return self.ln_post(self.out_blocks(x))

    def forward(self, embs, n_frames, return_stoks=False):
        stoks = self.quantize(embs, n_frames)

        if return_stoks:
            return stoks
        else:
            dequantize_embed = self.dequantize(stoks)
            return dequantize_embed
