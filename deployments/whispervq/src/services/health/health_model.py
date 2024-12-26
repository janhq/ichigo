from dataclasses import dataclass
from enum import Enum

class ServerStatusModel:
    @dataclass()
    class Response:
        class StatusEnum(Enum):
            OK = "OK"
            ERROR = "ERROR"
        status: StatusEnum
        message: str