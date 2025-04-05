import asyncio
import io
import json
import time
import base64
import wave
import numpy as np
# import cv2
from PIL import Image
import httpx
from fastapi import WebSocket, WebSocketDisconnect, APIRouter, Response
from starlette.websockets import WebSocketState
from .types import VoiceRecognitionResponse, FaceRecognitionResponse, GenerateRequest
import uuid
from .utils import answer_user_query, generate_tts
import uuid, os, time, subprocess

from .service import ProcessRequest
router = APIRouter(prefix="", tags=["voice"])

RHUBARB_PATH = os.path.join("rhubarb", "rhubarb.exe")

# Track processing status
isProcessing = False

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

def exec_command(command):
    """Executes a shell command and returns the output or raises an error."""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command failed: {e.stderr}")

def lip_sync_message(message):
    """Converts MP3 to WAV and generates lip-sync JSON."""
    start_time = time.time()
    print(f"Starting conversion for message {message}")
    print(RHUBARB_PATH)

    
    # Convert MP3 to WAV
    # exec_command(f"ffmpeg -y -i audios/message_{message}.mp3 audios/message_{message}.wav")
    # print(f"Conversion done in {int((time.time() - start_time) * 1000)}ms")

    # Generate lip-sync JSON
    exec_command(f'"{RHUBARB_PATH}" -f json -o {message}.json {message}.wav -r phonetic')
    print(f"Lip sync done in {int((time.time() - start_time) * 1000)}ms")

async def send_results_periodically(websocket: WebSocket, response):
    """
    Send the latest recognition results back to the client every second.
    """
    global isProcessing

    try:
        print("sending response",type(response), response)
        if response and hasattr(response, 'content'):
            byte_string = base64.b64encode(response.content).decode('utf-8')
            # save it temporarly
            # Generate a random UUID
            unique_filename = "generated_audio"

            # Decode the base64 string back to bytes
            audio_bytes = base64.b64decode(byte_string)

            # save the audio temporarly
            with open(unique_filename + '.wav', 'wb') as wav_file:
                wav_file.write(audio_bytes)

             ## generate the lipsync
            lip_sync_message(unique_filename)

            ## load the json
            with open(unique_filename + '.json', 'rb') as json_file:
                lipsync_data = json.load(json_file)

            json_data = {'audio': byte_string,
                         'lipsync': lipsync_data}
            # Send audio content as binary
            asyncio.create_task(websocket.send_text(json.dumps(json_data)))
            time.sleep(1)
            isProcessing = False
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
isProcessing = False

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
    global isProcessing
    
    await websocket.accept()
    
    try:
        while True:
            # Expect text messages (JSON format) with both audio and video keys.
            message = await websocket.receive_text()
            start = time.time()
            if isProcessing:
                print("Already processing, ignoring new request.")

            else:
                try:
                    isProcessing = True
                    data = json.loads(message)
                except Exception as e:
                    isProcessing = False
                    print("Invalid JSON received:", e)
                    continue

                audio_payload = data.get("audio")
                video_payload = data.get("video")

                response = await request_handler(audio_payload, video_payload)
                if response:
                    end = time.time()
                    print(f'Total TIme: ' , end - start)
                    asyncio.create_task(send_results_periodically(websocket, response))
                     

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()
        print("WebSocket closed")
    
    