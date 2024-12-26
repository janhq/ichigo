from dataclasses import dataclass
from enum import Enum


class ModelModel:
    @dataclass()
    class Request:
        whisper_model_path: str
        whisper_encoder_name: str
        whisper_encoder_path: str
        repo_id: str

    @dataclass()
    class Response:
        class StatusEnum(Enum):
            OK = "OK"
            ERROR = "ERROR"
        status: StatusEnum
        message: str
