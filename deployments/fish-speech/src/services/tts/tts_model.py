from tools.schema import ServeTTSRequest, ServeReferenceAudio
from dataclasses import dataclass

class TTSModel:
    @dataclass()
    class Response:
        audio: str
        sample_rate: int
        
    class Request(ServeTTSRequest):
        pass
    
    class ReferenceAudio(ServeReferenceAudio):
        pass