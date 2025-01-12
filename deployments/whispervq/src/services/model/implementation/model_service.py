import os
import urllib
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

    @staticmethod
    def download_encoder(url: str, root: str, in_memory: bool):
        logger = LoggerUtility.get_logger()
        os.makedirs(root, exist_ok=True)
        download_target = os.path.join(root, os.path.basename(url))

        if os.path.exists(download_target) and not os.path.isfile(download_target):
            raise RuntimeError(
                f"{download_target} exists and is not a regular file")
        if os.path.isfile(download_target):
            with open(download_target, "rb") as f:
                model_bytes = f.read()
            return model_bytes if in_memory else download_target
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
            total = int(source.info().get("Content-Length"))
            downloaded = 0
            count = 0
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                count += 1
                output.write(buffer)
                downloaded += len(buffer)
                if count % 1000 == 0:
                    logger.info(f"Downloaded {downloaded}/{total} bytes")

        model_bytes = open(download_target, "rb").read()
        return model_bytes if in_memory else download_target

    async def download_model(self, req: ModelModel.Request) -> ModelModel.Response:
        if not os.path.exists(self.download_folder/req.whisper_model_path):
            hf_hub_download(
                repo_id=req.repo_id,
                filename=req.whisper_model_path,
                local_dir=self.download_folder,
            )
        ModelService.download_encoder(req.whisper_encoder_path,
                                      self.download_folder, False)
        return ModelModel.Response(status=ModelModel.Response.StatusEnum.OK, message="downloaded successfully!")
