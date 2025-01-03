import os

from dotenv import load_dotenv

class IchigoVariables:
    def __init__(self):
        load_dotenv()
        self.latency : str = os.getenv("LATENCY")
        self.format: str = os.getenv("FORMAT")
        self.max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS"))
        self.chunk_length: int = int(os.getenv("CHUNK_LENGTH"))
        self.repetition_penalty: float = float(os.getenv("REPETITION_PENALTY"))
