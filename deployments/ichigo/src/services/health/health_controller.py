from common.abstract.controller_abstract import ControllerAbstract
from common.constant.fastapi_constant import RestConstant
from services.health.health_model import ServerStatusModel
from services.health.implementation.health_service import HealthService


class HealthController(ControllerAbstract):

    _prefix = "/health"

    def __init__(self):
        super().__init__(prefix=self._prefix)

    def _setup_routes(self):
        self.add_api_route("", self.server_status,
                           methods=[RestConstant.get])

    def _setup_services(self):
        self.health_service = HealthService()

    async def server_status(self) -> ServerStatusModel.Response:
        return await self.health_service.server_status()
