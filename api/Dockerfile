FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /code

RUN apt update && \
    apt install -y git ffmpeg && \
    apt clean

COPY . /code
RUN pip install ichigo
RUN pip install --no-cache-dir "fastapi[standard]==0.115.*"

CMD ["fastapi", "run", "asr.py", "--port", "8000"]