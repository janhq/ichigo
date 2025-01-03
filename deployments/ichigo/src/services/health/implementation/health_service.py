from logging import Logger

from common.utility.logger_utility import LoggerUtility
from services.health.health_interface import HealthInterface
from services.health.health_model import ServerStatusModel


class HealthService(HealthInterface):

    def __init__(self,):
        self.logger: Logger = LoggerUtility.get_logger()

    async def server_status(self) -> ServerStatusModel.Response:
        status = status = ServerStatusModel.Response.StatusEnum.OK
        message = "Still alive!"
        return ServerStatusModel.Response(status=status, message=message)
