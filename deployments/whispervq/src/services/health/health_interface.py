from abc import ABC, abstractmethod

from services.health.health_model import ServerStatusModel


class HealthInterface(ABC):

    @abstractmethod
    async def server_status(self) -> ServerStatusModel.Response:
        pass
