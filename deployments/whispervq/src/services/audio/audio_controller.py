
from common.abstract.controller_abstract import ControllerAbstract
from common.constant.fastapi_constant import RestConstant
from services.audio.audio_model import AudioModel
from services.audio.implementation.audio_service import AudioService


class AudioController(ControllerAbstract):

    _prefix = "/inference"

    def __init__(self):
        super().__init__(prefix=self._prefix)

    def _setup_routes(self):
        self.add_api_route("", self.inference,
                           methods=[RestConstant.post])

    def _setup_services(self):
        self.audio_service = AudioService.get_audio_service()

    async def inference(self, req: AudioModel.Request) -> AudioModel.Response:
        return await self.audio_service.inference(req)
