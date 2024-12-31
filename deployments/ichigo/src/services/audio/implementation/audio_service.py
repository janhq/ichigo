import aiohttp
import json
import io
import wave
from fastapi.responses import StreamingResponse
from common.utility.convert_utility import ConvertUtility
from variables.ichigo_variables import IchigoVariables
from services.audio.audio_model import AudioModel, FishSpeechRequest
from services.audio.audio_interface import AudioInterface
from common.constant.tts_constant import TTSConstant


class AudioService(AudioInterface):
    _audio_service = None

    @staticmethod
    def get_audio_service():
        if AudioService._audio_service is None:
            raise Exception("AudioService is not initialized")
        return AudioService._audio_service

    def __init__(self,  whisper_port, ichigo_port, fish_speech_port, ichigo_model):
        self.whisper_port = whisper_port
        self.ichigo_port = ichigo_port
        self.fish_speech_port = fish_speech_port
        self.ichigo_model = ichigo_model
        self.variables = IchigoVariables()
        self.example_tts_body = {
            "normalize": True,
            "format": "wav",
            "latency": "balanced",
            "max_new_tokens": 4096,
            "chunk_length": 200,
            "repetition_penalty": 1.5,
            "temperature": 0.7,
        }

    def split(self, byte_array: bytes):
        for i in range(0, len(byte_array), TTSConstant.chunk_length):
            data = byte_array[i:i+TTSConstant.chunk_length]
            temp_file = io.BytesIO()
            with wave.open(temp_file, 'wb') as temp_input:
                temp_input.setnchannels(TTSConstant.channels)  # Mono
                # Sample width in bytes
                temp_input.setsampwidth(TTSConstant.sample_width)
                temp_input.setframerate(TTSConstant.sample_rate)  # Sample rate
                temp_input.writeframes(data)
            temp_file.seek(0)
            yield temp_file.read()

    async def send_to_tts(self, session: aiohttp.ClientSession, text: str):
        body = FishSpeechRequest(text=text)
        async with session.post(f"http://localhost:{self.fish_speech_port}/inference", json=body.dict()) as response:
            byte = await response.read()
            # print("Putting bytes:" ,len(byte), str(datetime.now()))
            return byte

    async def tokenize_audio(self, session: aiohttp.ClientSession, request: AudioModel.AudioCompletionRequest) -> str:
        async with session.post(f"http://localhost:{self.whisper_port}/inference", json=request.input_audio.dict(), headers={"accept": "application/json"}) as response:
            if response.status == 200:
                response = await response.json()
                tokens = response["tokens"]
        return tokens

    async def streamchatcompletion(self, session: aiohttp.ClientSession, request: AudioModel.AudioCompletionRequest) -> str:
        async with session.post(f"http://localhost:{self.ichigo_port}/v1/chat/completions", json=request.dict(), headers={"Accept": "text/event-stream"}) as response:
            async for line in response.content:
                yield line

    async def inference_stream_text(self, request: AudioModel.AudioCompletionRequest):
        async with aiohttp.ClientSession() as session:
            tokens = await self.tokenize_audio(session, request)
            request.messages = request.messages + \
                [{"role": "user", "content": tokens}]
            if self.ichigo_model:
                request.model = self.ichigo_model

            async for line in self.streamchatcompletion(session, request):
                yield line

    async def inference_stream_audio(self, session: aiohttp.ClientSession, request: AudioModel.AudioCompletionRequest):
        async with aiohttp.ClientSession() as session:
            tokens = await self.tokenize_audio(session, request)
            request.messages = request.messages + \
                [{"role": "user", "content": tokens}]
            if self.ichigo_model:
                request.model = self.ichigo_model
            final_answer = ""
            answer = ""
            tokens_processed = 0
            chunk_size = 10
            currentCount = 0
            async for line in self.streamchatcompletion(session, request):
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    line = line[6:]
                    if line.startswith("[DONE]"):
                        if answer:
                            byte = await self.send_to_tts(session, answer)
                            for data in self.split(byte):
                                yield ("data: "+json.dumps({"audio": ConvertUtility.encode_to_base64(data)})+"\n").encode('utf-8')

                        yield ("data: "+json.dumps({"audio": None, "messages": request.messages + [{"role": "assistant", "content": final_answer}]}) + "\n").encode('utf-8')
                        yield "data: [DONE]\n".encode('utf-8')
                        break
                    object = json.loads(line)

                    if object["choices"][0]["delta"].get("content"):
                        delta_content = object["choices"][0]["delta"]["content"]
                        final_answer += delta_content

                        if currentCount < chunk_size:
                            answer += delta_content
                        elif currentCount < 60 and delta_content in [".", ",", ":", ";"]:
                            byte = await self.send_to_tts(session, answer)
                            for data in self.split(byte):
                                yield ("data: "+json.dumps({"audio": ConvertUtility.encode_to_base64(data)})+"\n").encode('utf-8')

                            answer = ""  # Reset answer
                            currentCount = 0
                            chunk_size = 60
                        elif chunk_size == 10:
                            answer += delta_content
                        else:
                            byte = await self.send_to_tts(session, answer)
                            for data in self.split(byte):
                                yield ("data: "+json.dumps({"audio": ConvertUtility.encode_to_base64(data)})+"\n").encode('utf-8')
                            answer = delta_content  # Reset answer
                            currentCount = 0
                            if chunk_size == 60:
                                chunk_size = 200

                        tokens_processed += 1
                        currentCount += 1

    async def inference_nontream(self, request: AudioModel.AudioCompletionRequest) -> AudioModel.Response:
        async with aiohttp.ClientSession() as session:
            tokens = await self.tokenize_audio(session, request)
            request.messages = request.messages + \
                [{"role": "user", "content": tokens}]
            if self.ichigo_model:
                request.model = self.ichigo_model

            async with session.post(f"http://localhost:{self.ichigo_port}/v1/chat/completions", json=request.dict(), headers={"accept": "application/json"}) as response:
                if response.status == 200:
                    response = await response.json()
                    if not request.output_audio:
                        return response
                    text = response["choices"][0]["message"]["content"]
            fish_speech_request = FishSpeechRequest(text=text)
            async with session.post(f"http://localhost:{self.fish_speech_port}/inference", json=fish_speech_request.dict(), headers={"accept": "application/json"}) as response:
                if response.status == 200:
                    response = await response.json()
                    audio = response["audio"]
            return AudioModel.Response(audio=audio, messages=request.messages + [{"role": "assistant", "content": text}])

    async def inference(self, request: AudioModel.AudioCompletionRequest):
        if request.stream:
            if request.output_audio:
                return StreamingResponse(self.inference_stream_audio(request))
            else:
                return StreamingResponse(self.inference_stream_text(request))
        else:
            return await self.inference_nontream(request)
