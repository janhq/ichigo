from vq_config import VQConfig
from models.vq_transformer import RQBottleneckTransformer


def make_vq_model(
    size: str,
    no_quantize=False,
    config: VQConfig = VQConfig(),
    dataset=None,
):
    common = dict(
        q_depth=1,
        depth=1,
        threshold_ema_dead_code=0,
        use_cosine_sim=True,
        config=config,
        no_quantize=no_quantize,
    )

    model_configs = {
        "medium-vi-2d-512c-dim64": dict(
            codebook_dim=64,
            vq_codes=512,
            n_head=16,
            downsample=2,
            whisper_model_name="medium",
        ),
        "medium-vi-2d-1024c-dim64": dict(
            codebook_dim=64,
            vq_codes=1024,
            n_head=16,
            downsample=2,
            whisper_model_name="medium",
        ),
        "medium-vi-2d-2048c-dim64": dict(
            codebook_dim=64,
            vq_codes=2048,
            n_head=16,
            downsample=2,
            whisper_model_name="medium",
        ),
        "merge-medium-vi-2d-2560c-dim64": dict(
            codebook_dim=64,
            vq_codes=2560,
            n_head=16,
            downsample=2,
            whisper_model_name="medium",
        ),
        "large-v3-vi-2d-512c-dim64": dict(
            codebook_dim=64,
            vq_codes=512,
            n_head=20,
            head_width=64,
            downsample=2,
            whisper_model_name="large-v3",
        ),
        "large-v3-vi-2d-1024c-dim64": dict(
            codebook_dim=64,
            vq_codes=1024,
            n_head=20,
            head_width=64,
            downsample=2,
            whisper_model_name="large-v3",
        ),
        "large-v3-vi-2d-2048c-dim64": dict(
            codebook_dim=64,
            vq_codes=2048,
            n_head=20,
            head_width=64,
            downsample=2,
            whisper_model_name="large-v3",
        ),
    }

    if size in model_configs:
        return RQBottleneckTransformer(**model_configs[size], **common)

    raise ValueError(f"Unknown model size: {size}")
