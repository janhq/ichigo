import os

from dotenv import load_dotenv


class FishSpeechVariable:
    def __init__(self):
        load_dotenv()
        self.max_text_length : int = int(os.getenv("MAX_TEXT_LENGTH", 2048))
        self.repo_id : str = os.getenv("REPO_ID")
        self.reference_name : str = os.getenv("REFERENCE_NAME")
        self.model_name : str = os.getenv("MODEL_NAME")
        self.decoder_checkpoint = os.getenv("DECODER_CHECKPOINT")
        self.decoder_config_name = os.getenv("DECODER_CONFIG_NAME")