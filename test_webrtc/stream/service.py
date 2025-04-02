import base64
import uuid
import cv2
import numpy as np
from typing import Any
import time
import httpx
import io
from PIL import Image
import wave
from .utils import answer_user_query, generate_tts
from .types import GenerateRequest, VoiceRecognitionResponse, FaceRecognitionResponse


class ProcessRequest:
    def __init__(self):
        self.transcription = ""
        self.voice_user = []
        self.face_user = []

    def convert_to_wav(self,audio_data: bytes, channels: int = 1, sampwidth: int = 2, framerate: int = 48000) -> bytes:
        """
        Convert raw PCM audio data to a WAV file in memory.
        Assumes 16-bit PCM.
        """
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sampwidth)
            wav_file.setframerate(framerate)
            wav_file.writeframes(audio_array.tobytes())
        wav_buffer.seek(0)
        return wav_buffer.read()

    
    async def process_audio(self,audio_data):
        print("Processing audio data...")
        try:
            start = time.time()
            if len(audio_data) == 0:
                print("Empty audio data")
                return
            wav_data = self.convert_to_wav(audio_data)
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://0.0.0.0:8000/voice/process",
                    files={"file": ("audio.wav", wav_data, "audio/wav")}
                )
                if response.status_code == 200:
                    res =  response.json()
                    print("time taken to process voice: ", time.time() - start)
                    response=  VoiceRecognitionResponse(**res)
                    self.transcription += response.transcription
                    self.voice_user.append((response.userid, response.score))
                    
        except Exception as e:
            print("Error processing audio:", e)
            return None
        
    async def process_video(self,img):
        """
        Process a video frame for face recognition.
        """
        print("Processing video frame for face recognition...")
        try:
            # Convert image to JPEG bytes
            start = time.time()
            img_bytes_io = io.BytesIO()
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(img_bytes_io, format='JPEG')
            img_bytes = img_bytes_io.getvalue()
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://0.0.0.0:8003/api/v1/identify-face",
                    files={"image": ("image.jpg", img_bytes, "image/jpeg")}
                )
                if response.status_code == 200:
                    res = response.json()
                    print("time taken to process face: ", time.time() - start)
                    result = FaceRecognitionResponse(userid=res[0], score=res[1])
                    self.face_user.append((result.userid, result.score))
        except Exception as e:
            print("Error processing video:", e)
        
    
    async def process_input(self, audio_data, img_data):
        if audio_data:
            await self.process_audio(audio_data)
        if img_data is not None:
            await self.process_video(img_data)

    async def __call__(self, audio_payload, img_payload, is_end:bool):
        if is_end:
            print("send transcription to RAG")
            answer = await answer_user_query(GenerateRequest(user_id = uuid.uuid4(), question = self.transcription))
            print("Answer from RAG: ", answer.generation)

            return await generate_tts(answer.generation)
        
        if not audio_payload and not img_payload:
            print("Missing audio and video payload; skipping this message.")
            return None
        
        audio_data = None
        img = None
        
        if audio_payload:
            try:
                audio_data = base64.b64decode(audio_payload)
            except Exception as e:
                print("Error decoding audio payload:", e)
        
        if img_payload:
            try:
                video_bytes = base64.b64decode(img_payload)
                np_arr = np.frombuffer(video_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception as e:
                print("Error decoding video payload:", e)


        await self.process_input(audio_data, img)
        
