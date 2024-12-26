from dataclasses import dataclass
from enum import Enum


class ModelModel:
    @dataclass()
    class Request:
        repo_id: str

    @dataclass()
    class Response:
        class StatusEnum(Enum):
            OK = "OK"
            ERROR = "ERROR"
        status: StatusEnum
        message: str
