import os
from pathlib import Path

from huggingface_hub import hf_hub_download

from common.utility.logger_utility import LoggerUtility
from services.model.model_interface import ModelInterface
from services.model.model_model import ModelModel


class ModelService(ModelInterface):
    def __init__(self,):
        self.logger = LoggerUtility.get_logger()
        self.download_folder = Path(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))))/"downloads"

    async def download_model(self, req: ModelModel.Request) -> ModelModel.Response:
        if not os.path.exists(self.download_folder/req.whisper_model_path):
            hf_hub_download(
                repo_id=req.repo_id,
                local_dir=self.download_folder,
            )
        return ModelModel.Response(status=ModelModel.Response.StatusEnum.OK, message="downloaded successfully!")
