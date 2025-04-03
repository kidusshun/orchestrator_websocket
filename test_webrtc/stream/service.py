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
from .utils import answer_user_query, generate_tts, add_voice_user, add_face_user
from .types import GenerateRequest, VoiceRecognitionResponse, FaceRecognitionResponse
from fastapi import UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile
from io import BytesIO


class ProcessRequest:
    def __init__(self):
        self.transcription = ""
        self.voice_user = []
        self.face_user = []
        self.user_id = None
        self.image = None
        self.audio = None
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
                    "http://127.0.0.1:8005/voice/process",
                    files={"file": ("audio.wav", wav_data, "audio/wav")},
                    timeout=30.0
                )
                if response.status_code == 200:
                    res =  response.json()
                    print("time taken to process voice: ", time.time() - start)
                    return  VoiceRecognitionResponse(**res)

                    
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
                    "http://127.0.0.1:8000/api/v1/identify-face",
                    files={"image": ("image.jpg", img_bytes, "image/jpeg")}
                )
                if response.status_code == 200:
                    res = response.json()
                    print("time taken to process face: ", time.time() - start)
                    return FaceRecognitionResponse(userid=res[0], score=res[1])
        except Exception as e:
            print("Error processing video:", e)
        
    
    async def process_input(self, audio_data, img_data):
        voice_user = None
        face_user = None
        if audio_data:
            voice_user = await self.process_audio(audio_data)
        if img_data is not None:
            face_user = await self.process_video(img_data)

        if voice_user and face_user:
            self.user_id = await self.identify_user(voice_user, face_user)
        



    async def identify_user(self, voice_user:VoiceRecognitionResponse, face_user:FaceRecognitionResponse) -> str:
        """
        Identify the user based on voice and face recognition results.
        """
        if str(voice_user.userid) == face_user.userid:
            return face_user.userid
        
        elif face_user.userid == 'Unknown' and not voice_user.userid:
            print("USER VOICE AND FACE NOT IDENTIFIED")
            uu = uuid.uuid4()
            await add_voice_user(uu, self.audio)
            await add_face_user(uu, self.image)
            return str(uu)
        elif not voice_user.userid:
            print("VOICE NOT IDENTIFIED")
            await add_voice_user(uuid.UUID(face_user.userid), self.audio)
            return face_user.userid
        elif face_user.userid == 'Unknown':
            print("FACE NOT IDENTIFIED")
            await add_face_user(voice_user.userid, self.image)
            return str(voice_user.userid)
        elif face_user.userid == str(voice_user.userid):
            print("USER IDENTIFIED")
            return face_user.userid
        else:
            if face_user.score < voice_user.score:
                print("FACE USER IDENTIFIED")
                await add_voice_user(uuid.UUID(face_user.userid), self.audio)
                return face_user.userid
            else:
                print("VOICE USER IDENTIFIED")
                await add_face_user(voice_user.userid, self.image) #type: ignore
                return str(voice_user.userid)

    async def __call__(self, audio_payload, img_payload):
        
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


        # Convert audio bytes to UploadFile
        audio_upload_file = None
        if audio_data:
            audio_upload_file = UploadFile(
            filename="audio.wav",
            file=BytesIO(audio_data),
            )

        # Convert image bytes to UploadFile
        img_upload_file = None
        if img is not None:
            _, img_encoded = cv2.imencode('.jpg', img)
            img_bytes = img_encoded.tobytes()
            img_upload_file = UploadFile(
            filename="image.jpg",
            file=BytesIO(img_bytes),
            )

        if img_upload_file:
            self.image = img_upload_file
        if audio_upload_file:
            self.audio = audio_upload_file

        # Pass the converted UploadFile objects to process_input
        await self.process_input(audio_data, img)
        

        if self.transcription != "":
            print("send transcription to RAG")
            answer = await answer_user_query(GenerateRequest(user_id = self.user_id if self.user_id else uuid.uuid4(), question = self.transcription))
            print("Answer from RAG: ", answer.generation)

            return await generate_tts(answer.generation)
        else:
            print("No transcription available")
            return None