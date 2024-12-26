from common.abstract.controller_abstract import ControllerAbstract
from common.constant.fastapi_constant import RestConstant
from services.tts.tts_model import TTSModel
from services.tts.implementation.tts_service import TTSService


class TTSController(ControllerAbstract):
    _prefix = "/inference"

    def __init__(self):
        super().__init__(prefix=self._prefix)

    def _setup_routes(self):
        self.add_api_route("", self.inference,
                           methods=[RestConstant.post])

    def _setup_services(self):
        self.tts_service = TTSService.get_tts_service()

    async def inference(self, req: TTSModel.Request) -> TTSModel.Response:
        return await self.tts_service.inference(req)
