from abc import ABC, abstractmethod

from services.audio.audio_model import AudioModel


class AudioInterface(ABC):

    @abstractmethod
    async def inference(self, req: AudioModel.AudioCompletionRequest) -> AudioModel.Response:
        pass
    