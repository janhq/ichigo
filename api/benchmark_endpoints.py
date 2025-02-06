from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import requests


ASR_HOST = "http://localhost:8000"


def asr():
    url = f"{ASR_HOST}/v1/audio/transcriptions"

    filename = "sample_10s.mp3"
    payload = dict(
        file=open(filename, "rb"),
        model=(None, "ichigo"),
    )

    t0 = time.perf_counter()
    resp = requests.post(url, files=payload)
    resp.raise_for_status()
    t1 = time.perf_counter()

    return t1 - t0


def s2r():
    url = f"{ASR_HOST}/s2r"

    t0 = time.perf_counter()
    resp = requests.post(url, files=dict(file=open("sample_10s.mp3", "rb")))
    resp.raise_for_status()
    t1 = time.perf_counter()

    return t1 - t0


def r2t():
    url = f"{ASR_HOST}/r2t"
    tokens = "<|sound_start|><|sound_1012|><|sound_1508|><|sound_1508|><|sound_0636|><|sound_1090|><|sound_0567|><|sound_0901|><|sound_0901|><|sound_1192|><|sound_1820|><|sound_0547|><|sound_1999|><|sound_0157|><|sound_0157|><|sound_1454|><|sound_1223|><|sound_1223|><|sound_1223|><|sound_1223|><|sound_1808|><|sound_1808|><|sound_1573|><|sound_0065|><|sound_1508|><|sound_1508|><|sound_1268|><|sound_0568|><|sound_1745|><|sound_1508|><|sound_0084|><|sound_1768|><|sound_0192|><|sound_1048|><|sound_0826|><|sound_0192|><|sound_0517|><|sound_0192|><|sound_0826|><|sound_0971|><|sound_1845|><|sound_1694|><|sound_1048|><|sound_0192|><|sound_1048|><|sound_1268|><|sound_end|>"

    t0 = time.perf_counter()
    resp = requests.post(url, json=dict(tokens=tokens))
    resp.raise_for_status()
    t1 = time.perf_counter()

    return t1 - t0


def main():
    num_concurrent_calls = 10

    print(f"Running {num_concurrent_calls} concurrent API calls...")
    with ThreadPoolExecutor(max_workers=num_concurrent_calls) as executor:
        print("ASR endpoint")
        futures = [executor.submit(asr) for _ in range(num_concurrent_calls)]
        for i, future in enumerate(as_completed(futures)):
            latency = future.result()
            print(f"Call {i}: Latency = {latency:.3f} seconds")

        print("S2R endpoint")
        futures = [executor.submit(s2r) for _ in range(num_concurrent_calls)]
        for i, future in enumerate(as_completed(futures)):
            latency = future.result()
            print(f"Call {i}: Latency = {latency:.3f} seconds")

        print("R2T endpoint")
        futures = [executor.submit(r2t) for _ in range(num_concurrent_calls)]
        for i, future in enumerate(as_completed(futures)):
            latency = future.result()
            print(f"Call {i}: Latency = {latency:.3f} seconds")


if __name__ == "__main__":
    main()
