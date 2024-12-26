from abc import ABC, abstractmethod

from services.model.model_model import ModelModel


class ModelInterface(ABC):

    @abstractmethod
    async def download_model(self, req: ModelModel.Request) -> ModelModel.Response:
        pass