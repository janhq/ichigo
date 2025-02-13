FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /code

RUN apt update && \
    apt install -y git ffmpeg && \
    apt clean

COPY . /code
RUN pip install ichigo
RUN pip install whisper
RUN pip install --no-cache-dir "fastapi[standard]==0.115.*"

RUN python -c "import whisper; whisper.load_model('medium')"
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download('homebrewltd/ichigo-whisper', 'merge-medium-vi-2d-2560c-dim64.pth')"

EXPOSE 8000

ENTRYPOINT ["fastapi", "run", "asr.py", "--port", "8000"]
