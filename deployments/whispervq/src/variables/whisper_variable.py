import os

from dotenv import load_dotenv


class WhisperVariable:

    def __init__(self):
        load_dotenv()
        self.whisper_model_path: str | None = os.getenv("WHISPER_MODEL_PATH")
        self.repo_id: str | None = os.getenv("REPO_ID")
        self.HF_MODELS: dict[str, str | None] = {
            os.getenv("WHISPER_ENCODER_NAME", "medium"): os.getenv("WHISPER_ENCODER_PATH")}
