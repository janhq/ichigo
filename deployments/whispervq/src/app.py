
import argparse
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, List

import uvicorn
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI

from common.utility.logger_utility import LoggerUtility
from services.audio.audio_controller import AudioController
from services.audio.implementation.audio_service import AudioService
from services.health.health_controller import HealthController


@asynccontextmanager
async def application_lifecycle(app: FastAPI) -> AsyncGenerator[None, None]:
    try:
        AudioService.get_audio_service()
    except Exception as e:
        LoggerUtility.get_logger().error(f"Error initializing audio service: {e}")
        raise e
    yield


def create_app() -> FastAPI:
    routes: List[APIRouter] = [
        HealthController(),
        AudioController()
    ]
    app = FastAPI(lifespan=application_lifecycle)
    for route in routes:
        app.include_router(route)
    return app


def parse_argument():
    parser = argparse.ArgumentParser(description="WhisperVQ Application")
    parser.add_argument('--log_path', type=str,
                        default='whisper.log', help='The log file path')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'TRACE'], help='The log level')
    parser.add_argument('--port', type=int, default=3348,
                        help='The port to run the WhisperVQ app on')
    parser.add_argument('--device_id', type=str, default="0",
                        help='The port to run the WhisperVQ app on')
    parser.add_argument('--package_dir', type=str, default="",
                        help='The package-dir to be extended to sys.path')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_argument()
    LoggerUtility.init_logger(__name__, args.log_level, args.log_path)

    env_path = Path(os.path.dirname(os.path.realpath(__file__))) / "variables" / ".env"
    load_dotenv(dotenv_path=env_path)
    app: FastAPI = create_app()
    print("Server is running at: 0.0.0.0:", args.port)
    uvicorn.run(app=app, host="0.0.0.0", port=args.port)
