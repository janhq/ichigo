import argparse, os,sys
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

from services.IchigoService import get_ichigo_service
from routes.IchigoRoute import ichigo_inference_router

@asynccontextmanager
async def lifespan(app: FastAPI):
   
    # on startup
    get_ichigo_service(args.whisper_port, args.ichigo_port, args.fish_speech_port,ichigo_model=args.ichigo_model)
    yield
    # on shutdown

app = FastAPI(lifespan=lifespan)

# include the routes
app.include_router(ichigo_inference_router)

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