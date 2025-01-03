
import argparse
import os
import sys
from pathlib import Path

from contextlib import asynccontextmanager

from typing import AsyncGenerator, List

import uvicorn
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI

from common.utility.logger_utility import LoggerUtility
from services.audio.audio_controller import AudioController
from services.audio.implementation.audio_service import AudioService
from services.health.health_controller import HealthController


def create_app() -> FastAPI:
    routes: List[APIRouter] = [
        HealthController(),
        AudioController()
    ]
    app = FastAPI()
    for route in routes:
        app.include_router(route)
    return app


def parse_argument():
    parser = argparse.ArgumentParser(description="Ichigo-wrapper Application")
    parser.add_argument('--log_path', type=str,
                        default='Ichigo-wrapper.log', help='The log file path')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'TRACE'], help='The log level')
    parser.add_argument('--port', type=int, default=22310,
                        help='The port to run the Ichigo-wrapper app on')
    parser.add_argument('--device_id', type=str, default="0",
                        help='The port to run the Ichigo-wrapper app on')
    parser.add_argument('--package_dir', type=str, default="",
                        help='The package-dir to be extended to sys.path')
    parser.add_argument('--whisper_port', type=int, default=3348,
                        help='The port of whisper vq model')
    parser.add_argument('--ichigo_port', type=int, default=39281,
                        help='The port of ichigo model')
    parser.add_argument('--fish_speech_port', type=int, default=22312,
                        help='The port of fish speech model')
    parser.add_argument('--ichigo_model', type=str, default="ichigo:8b-gguf-q4-km",
                        help='The ichigo model name')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_argument()
    LoggerUtility.init_logger(__name__, args.log_level, args.log_path)

    env_path = Path(os.path.dirname(os.path.realpath(__file__))
                    ) / "variables" / ".env"
    AudioService.initialize(
        args.whisper_port, args.ichigo_port, args.fish_speech_port, args.ichigo_model)
    load_dotenv(dotenv_path=env_path)
    app: FastAPI = create_app()
    print("Server is running at: 0.0.0.0:", args.port)
    uvicorn.run(app=app, host="0.0.0.0", port=args.port)
