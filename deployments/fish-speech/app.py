import argparse, os,sys
parser = argparse.ArgumentParser(description="Fish-speech Application")
parser.add_argument('--log_path', type=str,
                    default='fish-speech.log', help='The log file path')
parser.add_argument('--log_level', type=str, default='INFO',
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'TRACE'], help='The log level')
parser.add_argument('--port', type=int, default=22312,
                    help='The port to run the Fish-speech app on')
parser.add_argument('--device_id', type=str, default="0",
                    help='The port to run the Fish-speech app on')
parser.add_argument('--package_dir', type=str, default="",
                    help='The package-dir to be extended to sys.path')
args = parser.parse_args()
sys.path.insert(0, args.package_dir)
os.environ["CUDA_VISIBLE_DEVICES"] =args.device_id # Use the first Nvidia GPU

import logging
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
import os
import time
import psutil
import threading
logging.basicConfig(level=args.log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(args.log_path),
                        # logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)


# after set up logger we can import and use services

from services.TTSService import get_tts_service
from routes.TTSRoute import audio_inference_router

@asynccontextmanager
async def lifespan(app: FastAPI):
   
    # on startup
    get_tts_service()
    yield
    # on shutdown

app = FastAPI(lifespan=lifespan)

# include the routes
app.include_router(audio_inference_router)

def self_terminate():
    time.sleep(1)
    parent = psutil.Process(psutil.Process(os.getpid()).ppid())
    parent.kill()


@app.delete("/destroy")
async def destroy():
    threading.Thread(target=self_terminate, daemon=True).start()
    return {"success": True}

@app.get("/health")
async def health():
    return {"status": "OK"}

if __name__ == "__main__":
    import uvicorn
    from uvicorn.config import LOGGING_CONFIG

    LOGGING_CONFIG["handlers"]["default"] = {
        "class": "logging.FileHandler",
        "filename": args.log_path,
        "formatter": "default"
    }
    LOGGING_CONFIG["handlers"]["access"] = {
        "class": "logging.FileHandler",
        "filename": args.log_path,
        "formatter": "access"
    }
    LOGGING_CONFIG["loggers"]["uvicorn.error"]["level"] = args.log_level
    LOGGING_CONFIG["loggers"]["uvicorn.access"]["level"] = args.log_level

# Print supported formats at startup

    uvicorn.run(app, host="0.0.0.0", port=args.port)