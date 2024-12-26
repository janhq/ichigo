from abc import ABC, abstractmethod

from services.tts.tts_model import TTSModel


class TTSInterface(ABC):

    @abstractmethod
    async def inference(self, req: TTSModel.Request) -> TTSModel.Response:
        pass