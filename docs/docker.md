# Docker

1. Pull the container from Docker Hub

```bash
docker pull menloltd/ichigo:v0.0.1
```

2. Create a `docker-compose.yml` file

```yaml
services:
  asr:
    image: menloltd/ichigo:v0.0.1
    ports:
      - "8000:8000"
    command: ["fastapi", "run", "asr.py", "--port", "8000"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [ gpu ]
```

3. Start the container

```bash
docker compose up -d
```

4. Test with `curl`

```bash
curl "http://localhost:8000/r2t" -X POST \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  --data '{"tokens":"<|sound_start|><|sound_1012|><|sound_1508|><|sound_1508|><|sound_0636|><|sound_1090|><|sound_0567|><|sound_0901|><|sound_0901|><|sound_1192|><|sound_1820|><|sound_0547|><|sound_1999|><|sound_0157|><|sound_0157|><|sound_1454|><|sound_1223|><|sound_1223|><|sound_1223|><|sound_1223|><|sound_1808|><|sound_1808|><|sound_1573|><|sound_0065|><|sound_1508|><|sound_1508|><|sound_1268|><|sound_0568|><|sound_1745|><|sound_1508|><|sound_0084|><|sound_1768|><|sound_0192|><|sound_1048|><|sound_0826|><|sound_0192|><|sound_0517|><|sound_0192|><|sound_0826|><|sound_0971|><|sound_1845|><|sound_1694|><|sound_1048|><|sound_0192|><|sound_1048|><|sound_1268|><|sound_end|>"}'

```
