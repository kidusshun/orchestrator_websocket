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
from .service import answer_user_query, generate_tts
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

async def process_audio(audio_data: bytes):
    """
    Process a chunk of audio by converting it to WAV and sending it to the transcription API.
    """
    print("Processing audio data...")
    try:
        start = time.time()
        if len(audio_data) == 0:
            print("Empty audio data")
            return
        wav_data = convert_to_wav(audio_data)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/voice/process",
                files={"file": ("audio.wav", wav_data, "audio/wav")}
            )
            if response.status_code == 200:
                res =  response.json()
                print("time taken to process voice: ", time.time() - start)
                return VoiceRecognitionResponse(**res)
                
            return None
    except Exception as e:
        print("Error processing audio:", e)
        return None

async def process_video(img):
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
                "http://localhost:8003/api/v1/identify-face",
                files={"image": ("image.jpg", img_bytes, "image/jpeg")}
            )
            if response.status_code == 200:
                res = response.json()
                print("time taken to process face: ", time.time() - start)
                return FaceRecognitionResponse(userid=res[0], score=res[1])
            return None
    except Exception as e:
        print("Error processing video:", e)
        return None

async def process_input(audio_data, img):
    """
    Process both audio and video concurrently.
    """
    results = await asyncio.gather(
        process_audio(audio_data),
        process_video(img)
    )

    request = await compare_results(results, audio_data, img)

    start = time.time()
    response = await answer_user_query(request=GenerateRequest(user_id = str(uuid.uuid4()), question = "what is kifiya"))
    print("time taken to generate response: ", time.time() - start)

    if response:
        start = time.time()
        speech = await generate_tts(response.generation)
        print("time taken to generate speech: ", time.time() - start)
        if speech:
            return speech

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

@router.websocket("/ws/media")
async def websocket_media(websocket: WebSocket):
    """
    WebSocket endpoint that always expects a message with both audio and video.
    Each message should be a JSON object:
      {
         "audio": "<base64-encoded-audio-data>",
         "video": "<base64-encoded-video-data>"
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

            # Verify that both audio and video payloads exist
            audio_payload = data.get("audio")
            video_payload = data.get("video")
            if not audio_payload or not video_payload:
                print("Missing audio or video payload; skipping this message.")
                continue

            # Decode and process audio
            try:
                audio_data = base64.b64decode(audio_payload)
            except Exception as e:
                print("Error decoding audio payload:", e)
                continue

            # Decode and process video
            try:
                video_bytes = base64.b64decode(video_payload)
                np_arr = np.frombuffer(video_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception as e:
                print("Error decoding video payload:", e)
                continue

            response = await process_input(audio_data, img)

            await send_results_periodically(websocket, response)

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()
        print("WebSocket closed")
    
    