import asyncio
import io
import json
import time
import base64
import wave
import numpy as np
import cv2
from PIL import Image
import httpx
from fastapi import WebSocket, WebSocketDisconnect, APIRouter, Response
from starlette.websockets import WebSocketState
from .types import VoiceRecognitionResponse, FaceRecognitionResponse, GenerateRequest
import uuid
from .utils import answer_user_query, generate_tts

from .service import ProcessRequest
router = APIRouter(prefix="", tags=["voice"])


def convert_to_wav(audio_data: bytes, channels: int = 1, sampwidth: int = 2, framerate: int = 48000) -> bytes:
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

async def send_results_periodically(websocket: WebSocket, response):
    """
    Send the latest recognition results back to the client every second.
    """
    try:
        print("sending response",type(response), response)
        if response and hasattr(response, 'content'):
            # Send audio content as binary
            await websocket.send_bytes(response.content)
    except WebSocketDisconnect:
        print("Client disconnected from periodic sender")


async def compare_results(results, audio, image):
    voice_user = results[0]
    face_user = results[1]

    print(voice_user, face_user)

    # if not voice_user or not face_user:
    #     print("NO USER IDENTIFIED")
    # elif face_user.userid == 'Unknown' and not voice_user.userid:
    #         print("USER VOICE AND FACE NOT IDENTIFIED")
    #         uu = uuid.uuid4()
    #         await add_voice_user(uu, audio)
    #         await add_face_user(uu, image)
    #         request = GenerateRequest(user_id=str(uu), question=voice_user.transcription)
    # elif not voice_user.userid:
    #     print("VOICE NOT IDENTIFIED")
    #     await add_voice_user(uuid.UUID(face_user.userid), audio)
    #     request = GenerateRequest(user_id=face_user.userid, question=voice_user.transcription)
    # elif face_user.userid == 'Unknown':
    #     print("FACE NOT IDENTIFIED")
    #     await add_face_user(voice_user.userid, image)
    #     request = GenerateRequest(user_id=str(voice_user.userid), question=voice_user.transcription)
    # elif face_user.userid == str(voice_user.userid):
    #     print("USER IDENTIFIED")
    #     request = GenerateRequest(user_id=face_user.userid, question=voice_user.transcription)
    # else:
    #     if face_user.score < voice_user.score:
    #         print("FACE USER IDENTIFIED")
    #         await add_voice_user(uuid.UUID(face_user.userid), audio)
    #         request = GenerateRequest(user_id=face_user.userid, question=voice_user.transcription)
    #     else:
    #         print("VOICE USER IDENTIFIED")
    #         await add_face_user(voice_user.userid, image) #type: ignore
    #         request = GenerateRequest(user_id=str(voice_user.userid), question=voice_user.transcription)

    # return request


request_handler = ProcessRequest()

@router.websocket("/ws/media")
async def websocket_media(websocket: WebSocket):
    """
    WebSocket endpoint that always expects a message with both audio and video.
    Each message should be a JSON object:
      {
         "audio": "<base64-encoded-audio-data>",
         "video": "<base64-encoded-video-data>"
         "is_end": true/false
      }
    """
    
    await websocket.accept()
    
    
    
    try:
        while True:
            # Expect text messages (JSON format) with both audio and video keys.
            message = await websocket.receive_text()
            try:
                data = json.loads(message)
            except Exception as e:
                print("Invalid JSON received:", e)
                continue

            audio_payload = data.get("audio")
            video_payload = data.get("video")
            is_end = data.get("is_end", False)
            
            response = await request_handler(audio_payload, video_payload, is_end)
            if response:
                await send_results_periodically(websocket, response)

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()
        print("WebSocket closed")
    
    