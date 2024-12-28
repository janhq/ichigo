import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import whisper
from huggingface_hub import hf_hub_download
from vector_quantize_pytorch import ResidualVQ

from data.utils import get_tokenizer
from models.layers import ResidualAttentionBlock
from models.modules import LayerNorm


class RQBottleneckTransformer(nn.Module):
    """
    Residual Quantized Bottleneck Transformer for speech processing.

    This model combines vector quantization with a transformer architecture for efficient
    speech representation learning. It can process audio inputs through a quantization
    bottleneck and generate text outputs using a transformer decoder.

    Args:
        vq_codes (int): Number of codes in the vector quantizer codebook
        q_depth (int): Depth of the quantizer
        depth (int): Number of transformer layers
        n_head (int): Number of attention heads
        head_width (int): Dimension of each attention head
        ffn_mult (int): Multiplier for FFN layer width
        codebook_dim (int): Dimension of codebook entries
        threshold_ema_dead_code (float): Threshold for EMA dead code detection
        use_cosine_sim (bool): Whether to use cosine similarity in VQ
        kl_loss_mul (float): Multiplier for KL divergence loss
        downsample (int): Downsampling factor
        no_quantize (bool): If True, skip quantization
        whisper_model_name (str): Name of the Whisper model to use
        config (object): Configuration object with additional parameters
    """

    def __init__(
        self,
        vq_codes=512,
        q_depth=12,
        depth=1,
        n_head=2,
        head_width=64,
        ffn_mult=4,
        codebook_dim=2,
        threshold_ema_dead_code=2,
        use_cosine_sim=False,
        kl_loss_mul=1,
        downsample=1,
        no_quantize=False,
        whisper_model_name="tiny.en",
        config=None,
    ):
        super().__init__()
        self._init_attributes(locals())
        self._init_model_components()
        self._init_loss_functions()
        self._init_buffers()
        self.apply(self.init_transformer)

    def _init_attributes(self, params):
        """Initialize model attributes from parameters"""
        # Store initialization arguments
        self.__stored_args__ = {k: v for k, v in params.items() if k != "self"}

        self.width = params["n_head"] * params["head_width"]
        self.base_width = 3 * params["head_width"]
        for k, v in params.items():
            if k != "self":
                setattr(self, k, v)
        self.stoks_len = 1500 // self.downsample
        self.stoks_per_sec = self.stoks_len // 30
        self.whmodel = None
        self.positions = torch.arange(
            0, 1500, dtype=torch.long
        )  # TODO: hardcorded this? calculated from whisper? -> (80, n_frames)

    def _init_model_components(self):
        """Initialize the model's neural network components"""
        if not self.no_quantize:
            self._init_quantization_components()
            self._init_transformer_components()

    def _init_quantization_components(self):
        """Initialize components related to vector quantization"""
        n_mlp = self.width * self.ffn_mult
        self.mlp = nn.Sequential(
            nn.Linear(self.width, n_mlp), nn.GELU(), nn.Linear(n_mlp, self.width)
        )
        self.mlp_ln = LayerNorm(self.width)

        # Downsample convolution if specified
        if self.config.downsample_conv:
            self.downsample_conv = nn.Conv1d(
                self.width, self.width, kernel_size=3, stride=self.downsample, padding=1
            )
        else:
            self.downsample_conv = None

        # Adjust vq_codes if using mask embeddings - force embeddings corresponding to the input audio padding to a constant value
        if self.config.mask_embs:
            vq_codes = self.vq_codes + 1

        # Initialize ResidualVQ
        self.rq = ResidualVQ(
            dim=self.width,
            codebook_size=vq_codes,
            decay=self.config.codebook_decay,
            commitment_weight=1.0,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
            use_cosine_sim=self.use_cosine_sim,
            codebook_dim=self.codebook_dim,
            num_quantizers=1,
        )

        self.register_buffer("_codebook_usage", torch.zeros(vq_codes))

    def _init_transformer_components(self):
        """Initialize transformer-specific components"""
        qk_scale = self.config.query_mult * 8 / math.sqrt(self.head_width)

        self.positional_embedding = nn.Embedding(
            1500, self.width
        )  # FIXME: should be self.stoks_len  -> 1500 ~ length of semantic tokens

        self._out_blocks = nn.Sequential(
            *[
                ResidualAttentionBlock(
                    self.width,
                    self.n_head,
                    qk_scale=qk_scale,
                    ffn_mult=self.ffn_mult,
                    rope=self.config.rope,
                )
                for _ in range(self.depth)
            ]
        )

        self.ln_post = LayerNorm(self.width)

    def _init_loss_functions(self):
        """Initialize loss functions"""
        self.ce_lossf = nn.CrossEntropyLoss(ignore_index=-100)
        self.kl_lossf = nn.KLDivLoss(reduction="batchmean")

    def _init_buffers(self):
        """Initialize model buffers"""
        self.register_buffer("val_true", torch.zeros(1))
        self.register_buffer("val_total", torch.zeros(1))

    def init_transformer(self, m):
        """Initialize transformer weights"""
        if isinstance(m, nn.Linear):
            self._init_linear_layer(m)
        elif isinstance(m, nn.Embedding):
            self._init_embedding_layer(m)
        elif isinstance(m, nn.LayerNorm):
            self._init_layernorm(m)

    def _init_linear_layer(self, m):
        """Initialize linear layer weights"""
        m.lr_scale = 1 / (m.weight.shape[1] / self.base_width)
        std = self.config.init_std / m.weight.shape[1]
        torch.nn.init.trunc_normal_(m.weight, std=std, a=-3 * std, b=3 * std)
        if m.bias is not None:
            torch.nn.init.trunc_normal_(m.bias, std=std, a=-3 * std, b=3 * std)

    def _init_embedding_layer(self, m):
        """Initialize embedding layer weights"""
        m.no_weight_decay = True
        m.lr_scale = self.config.embeddings_lr_scale
        std = self.config.embeddings_std
        torch.nn.init.trunc_normal_(m.weight, std=std, a=-3 * std, b=3 * std)

    def _init_layernorm(self, m):
        """Initialize layer normalization weights"""
        m.no_weight_decay = True
        torch.nn.init.constant_(m.bias, 0)
        torch.nn.init.constant_(m.weight, 1)

    def forward(self, samples, mask=None, input_toks=None, output_toks=None):
        """
        Forward pass of the model.
        Args:
            samples (torch.Tensor): Input audio samples [B, 1, T]
            mask (torch.Tensor, optional): Attention mask [B, 1, T] (only for training)
            input_toks (torch.Tensor, optional): Input tokens [B, 1, S] (only for training)
            output_toks (torch.Tensor, optional): Target output tokens [B, 1, S] (only for training)
        Returns:
            During training: tuple (list_loss, logits, loss)
            During inference: Whisper decoded result
        """
        # Training mode: Extract teacher embeddings and logit
        embs, teacher_logits = self.extract_teacher(samples, input_toks, output_toks)

        if not self.no_quantize:
            # Process through quantization pipeline
            x = self._process_quantization(embs, mask)
            logits = self.whmodel[0].decoder(input_toks.squeeze(1), x)
            loss, list_loss = self._compute_loss(
                logits, output_toks.squeeze(1), teacher_logits
            )
        else:
            logits = self.whmodel[0].decoder(input_toks.squeeze(1), embs)
            loss, list_loss = self._compute_loss(
                logits, output_toks.squeeze(1), teacher_logits
            )
            loss = loss + self.fake_parameter

        # Update validation metrics if not training
        if not self.training:
            self._update_validation_metrics(logits, output_toks.squeeze(1))

        return list_loss, logits, loss

    def _process_quantization(self, embs, mask=None):
        """
        Process embeddings through the quantization pipeline.

        Args:
            embs (torch.Tensor): Input embeddings (8, 1500, 1024)
            mask (torch.Tensor): Attention mask (8, 1500)
        Returns:
            torch.Tensor: Processed and quantized embeddings
        """
        x = self.downsample_embeddings(embs)
        x = x + self.mlp(self.mlp_ln(x))  # (8, 750, 1024)

        # VQ bottleneck
        quantized, indices, self.commit_loss = self.rq(x)
        # (8, 750, 1024), (8, 750, 1), (1, 1)
        # indices of seq len, range [0, vq_codes (w/ mask code)]
        # 1024 this is dim = self.width, not codebook_dim
        self.commit_loss = self.commit_loss.mean()

        # ! Update codebook usage tracking (per step)
        if self.training:
            self._codebook_usage.zero_()
            for sample_indices in indices:  # sample_indices shape: [750, 1]
                sample_indices_flat = sample_indices.view(-1)
                unique_indices, counts = torch.unique(
                    sample_indices_flat, return_counts=True
                )
                assert (
                    unique_indices < self.vq_codes + 1  # +1 for masked embeddings
                ).all(), f"Found index >= {self.vq_codes}"

                self._codebook_usage.scatter_add_(
                    0, unique_indices, counts.float() / indices.size(0)
                )

        # Post-quantization processing
        x = quantized.repeat_interleave(self.downsample, -2)  # (8, 1500, 1024)

        # Handle masked embeddings only during training
        if self.training and self.config.mask_embs and mask is not None:
            project_out = (
                getattr(self.rq, "project_out", None) or self.rq.layers[0].project_out
            )
            # self.rq.layers[0]._codebook.embed[0, self.vq_codes] dim 64
            x[~mask] = project_out(self.rq.layers[0]._codebook.embed[0, self.vq_codes])

        # Add positional embeddings and apply transformer
        x = x + self.positional_embedding(self.positions.to(x.device))
        x = self.ln_post(self.out_blocks(x))

        return x

    def _compute_loss(self, logits, output_toks, teacher_logits):
        """
        Compute the total loss combining CE, KL, and commitment losses.
        Args:
            logits (torch.Tensor): Model predictions
            output_toks (torch.Tensor): Target tokens
            teacher_logits (torch.Tensor): Teacher model logits
        Returns:
            torch.Tensor: Combined loss value
        """
        self.ce_loss = self.ce_lossf(
            logits.view(-1, logits.shape[-1]), output_toks.view(-1)
        )

        # Only compute KL loss in Phase 1
        if hasattr(self, "phase") and self.phase == 1:
            self.kl_loss = self.kl_lossf(
                F.log_softmax(logits, dim=-1), F.softmax(teacher_logits, dim=-1)
            )
        else:
            self.kl_loss = 0

        loss = self.ce_loss + self.kl_loss_mul * self.kl_loss

        if not self.no_quantize:
            loss += self.commit_loss
        return loss, [self.ce_loss, self.kl_loss, self.commit_loss]

    def _update_validation_metrics(self, logits, output_toks):
        """Update validation metrics"""
        valid_toks = output_toks != -100
        self.val_true += (
            (logits.detach().argmax(-1)[valid_toks] == output_toks[valid_toks])
            .float()
            .sum()
        )
        self.val_total += valid_toks.float().sum()

    # @torch.no_grad()
    # def quantize(self, audio):
    #     print(self.rq.layers[0]._codebook.embed[0, self.vq_codes].shape)
    #     if isinstance(audio, str):
    #         x, sr = torchaudio.load(audio)
    #         x = torchaudio.transforms.Resample(sr, 16000)(x)[0]
    #         audio = x.unsqueeze(0)
    #     mel_1 = self.log_mel_spectrogram(audio)
    #     n = mel_1.shape[-1]
    #     audio_max_length = 30 * 16000
    #     if audio.shape[-1] > audio_max_length:
    #         audio = audio[:audio_max_length]
    #     else:
    #         audio = F.pad(audio, (0, audio_max_length - audio.shape[-1]), value=0)

    #     # Encode Mel
    #     mel = self.log_mel_spectrogram(audio)
    #     embs = self.whmodel[0].encoder(mel)

    #     # Quantize
    #     x = self.downsample_embeddings(embs)
    #     x = x + self.mlp(self.mlp_ln(x))
    #     _, stoks, _ = self.rq(x)  # quantizer.shape = (1, 750, 1024)
    #     # stoks = stoks[:,:n//2//self.downsample]
    #     stoks = stoks.squeeze(-1)
    #     stoks = stoks[:,:n//2//self.downsample]

    #     return stoks[0]
    @torch.no_grad()
    def quantize(self, audio):
        if isinstance(audio, str):
            x, sr = torchaudio.load(audio)
            x = torchaudio.transforms.Resample(sr, 16000)(x)[0]
            audio = x.unsqueeze(0)
        mel = self.log_mel_spectrogram(audio)
        n = mel.shape[-1]
        if n > whisper.audio.N_FRAMES:
            padding = 0
            padded = mel[:,:,:whisper.audio.N_FRAMES]
        else:
            padding = -n % whisper.audio.N_FRAMES
            padded = F.pad(mel, (0, padding), value=-1.5)
        embs = self.whmodel[0].encoder(padded)
        x = self.downsample_embeddings(embs)
        x = x + self.mlp(self.mlp_ln(x))
        _, stoks, _ = self.rq(x)
        stoks = stoks.squeeze(-1)
        stoks = stoks[:,:n//2//self.downsample]
        return stoks[0]

    def dequantize(self, stoks):
        # Dequantize
        assert self.q_depth == 1
        assert len(stoks.shape) == 1, "batch processing is not supported"

        padding = torch.nonzero(stoks == 2048)
        if padding.any():
            stoks = stoks[: padding[0, 0]]

        stoks = F.pad(
            stoks,
            (0, self.stoks_len - stoks.shape[-1]),
            value=2048 if self.config.mask_embs else 0,
        )  # 750
        # print(stoks)

        x = self.rq.layers[0]._codebook.embed[
            0, stoks.to(torch.long).view(-1)
        ]  # (750, 64)
        x = x.repeat_interleave(self.downsample, -2)  # (1500, 64)

        project_out = (
            getattr(self.rq, "project_out", None) or self.rq.layers[0].project_out
        )
        x = project_out(x).unsqueeze(0)  # (1500, 1024)

        positions = torch.arange(0, x.shape[-2], dtype=torch.long, device=x.device)
        x = x + self.positional_embedding(positions)

        return self.ln_post(self.out_blocks(x))
    @torch.no_grad()
    def inference(self, samples):
        """Perform inference on input samples"""

        # Quantize and Dequantize
        stoks = self.quantize(samples)
        dequantize_embed = self.dequantize(stoks).to(self.whmodel[0].device)

        # Decode text
        return self.whmodel[0].decode(dequantize_embed, self.decoding_options)

    @torch.no_grad()
    def extract_teacher(self, samples, input_toks, output_toks):
        """
        Extract embeddings and logits from teacher model.

        Args:
            samples (torch.Tensor): Input audio samples (B, 16000 * T)
            input_toks (torch.Tensor): Input tokens (B, MaxToks)
            output_toks (torch.Tensor): Target tokens (B, MaxToks)

        Returns:
            tuple: (embeddings, teacher_logits)
        """
        # self.log_mel_spectrogram(samples).shape = [8, 80, 1500]
        embs = self.whmodel[0].encoder(
            self.log_mel_spectrogram(samples)
        )  # [8, 1500, 1024]

        teacher_logits = self.whmodel[0].decoder(input_toks, embs)  # [8, 200, 51865]
        # Create mask and apply it properly
        mask = (output_toks.squeeze(1) == -100).unsqueeze(-1)
        teacher_logits = teacher_logits.masked_fill(mask, 0)  # [8, 200, 51865]
        return embs, teacher_logits

    def downsample_embeddings(self, x):
        """
        Downsample embeddings using configured method (conv, mean, or stride).

        Args:
            x (torch.Tensor): Input embeddings [B, T, D]

        Returns:
            torch.Tensor: Downsampled embeddings
        """
        if self.downsample_conv is not None:
            return x[:, :: self.downsample] + self.downsample_conv(
                x.transpose(-1, -2)
            ).transpose(-2, -1)
        elif self.config.downsample_mean:
            bs, slen, depth = x.shape  # [8, 1500, 1024]
            return x.reshape(bs, slen // self.downsample, self.downsample, depth).mean(
                -2
            )
        else:
            return x[:, :: self.downsample]

    def out_blocks(self, x):
        """Process through transformer blocks"""
        for l in self._out_blocks:
            x = l(x, self.positions)
        return x

    def get_metrics(self):
        """Get validation metrics"""
        metrics = {
            "acc_0": (self.val_true / self.val_total).item(),
        }
        self.val_true[:] = 0
        self.val_total[:] = 0
        return metrics

    def setup(self, device, language=None, prompt=None):
        """Setup the model on specified device"""
        self.ensure_whisper(device=device, language=language, prompt=prompt)

    def ensure_whisper(self, device=None, language=None, prompt=None):
        """Ensure Whisper model is loaded"""
        if self.whmodel is not None:
            return
        device = device or self.device

        if self.whmodel is None:
            self.whmodel = [whisper.load_model(self.whisper_model_name, device=device)]
        params = {}
        if language is not None:
            params['language'] = language
            if language in ['en', 'vi']:
                print(f"🍓 Setting testing options for {language}")
                
        if prompt is not None:
            print("🚀 Setting decoding options for demo with custom prompt")
            params['prompt'] = prompt
            
        self.decoding_options = whisper.DecodingOptions(**params)
        
        self.tokenizer = get_tokenizer(self.whisper_model_name, None)

    @property
    def device(self):
        """Get device of the model"""
        return next(self.parameters()).device

    def log_mel_spectrogram(self, samples):
        """Convert audio samples to log mel spectrogram"""
        return whisper.log_mel_spectrogram(
            samples, 128 if self.whisper_model_name == "large-v3" else 80
        )

    @classmethod
    def load_model(
        cls,
        ref,
        repo_id=None,
        filename=None,
        local_dir=None,
        local_filename=None,
    ):
        """Load model from file or Hugging Face Hub.

        Args:
            ref (str): Either a local path or "repo_id:filename" format
            repo_id (str, optional): Hugging Face repository ID
            filename (str, optional): Filename in the repository
            local_dir (str, optional): Local directory for downloads
            local_filename (str, optional): Direct path to local file

        Returns:
            RQBottleneckTransformer: Loaded model instance

        Raises:
            ValueError: If the model file or config is invalid
            FileNotFoundError: If the file cannot be found
        """
        try:
            # Parse reference string
            if repo_id is None and filename is None and local_filename is None:
                if ":" in ref:
                    repo_id, filename = ref.split(":", 1)
                else:
                    local_filename = ref

            # Download or use local file
            if not local_filename:
                local_filename = hf_hub_download(
                    repo_id=repo_id, filename=filename, local_dir=local_dir
                )

            # Load and validate spec
            spec = torch.load(local_filename)
            if "config" not in spec or "state_dict" not in spec:
                raise ValueError("Invalid model file format")

            # Initialize and load model
            model = cls(**spec["config"], config=spec.get("config", None))
            model.load_state_dict(spec["state_dict"])
            model.eval()
            return model

        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}") from e

    def save_model(self, fname, store_parameters=True):
        """Save model to file"""
        torch.save(
            dict(
                config=self.__stored_args__,
                state_dict=self.state_dict() if store_parameters else None,
            ),
            fname,
        )

    def get_codebook_stats(self):
        """Calculate codebook utilization statistics for current batch"""
        total_codes = self.vq_codes + 1  # +1 for masked embeddings
        total_usage = self._codebook_usage.sum().item()  # 750 * 8
        used_codes = (self._codebook_usage > 0).sum().item()
        utilization = used_codes / total_codes * 100

        usage_dist = self._codebook_usage / (total_usage + 1e-7)
        entropy = -(usage_dist * torch.log2(usage_dist + 1e-7)).sum().item()

        return {
            "used_codes": used_codes,
            "utilization": utilization,
            "entropy": entropy,
            "usage_per_code": self._codebook_usage.cpu().tolist(),
        }