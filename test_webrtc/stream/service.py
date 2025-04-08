import base64
import uuid
import cv2
import numpy as np
from typing import Any, Optional
import time
import httpx
import io
from PIL import Image
import wave
from .utils import answer_user_query, generate_tts, add_voice_user, add_face_user, update_face_user
from .types import GenerateRequest, VoiceRecognitionResponse, FaceRecognitionResponse
from fastapi import UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile
from io import BytesIO

import websockets
import asyncio
import json
import os
from asyncio import Lock
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessRequest:
    def __init__(self):
        self.transcription = ""
        self.voice_user = []
        self.face_user = []
        self.user_id = None
        self.image = None
        self.audio = None
        self.face_rec_ws = None
        self.latest_face_rec_state: Optional[FaceRecognitionResponse] = None
        self.face_rec_config: Dict[str, Any] = {
            "action": "configure",
            "threshold": 0.5,
            "max_faces": 5
        }
        face_host = os.getenv("FACE_RECOGNITION_HOST")
        face_port = os.getenv("FACE_RECOGNITION_PORT")
        self.face_rec_url = f"ws://{face_host}:{face_port}/api/v2/identify"
        self.ws_lock = Lock()
        
    async def _ensure_face_rec_connection(self):
        """Ensures the WebSocket connection for face recognition is active."""
        if self.face_rec_ws and (await self.face_rec_ws.ping()):
            return True

        print("Face Rec WebSocket connection not active. Attempting to connect...")
        try:
            self.face_rec_ws = await websockets.connect(self.face_rec_url)
            print(f"Connected to Face Rec WebSocket at {self.face_rec_url}")
            await self.face_rec_ws.send(json.dumps(self.face_rec_config))
            await asyncio.sleep(0.1)  # Brief pause after configuration
            print(f"Face Rec WebSocket configured: {self.face_rec_config}")
            return True
        except (websockets.exceptions.WebSocketException, ConnectionRefusedError, OSError) as e:
            print(f"Failed to connect or configure Face Rec WebSocket: {e}")
            self.face_rec_ws = None
            self.latest_face_rec_state = None # Clear state on connection failure
            return False
        except Exception as e:
             print(f"An unexpected error occurred during WebSocket connection/configuration: {e}")
             self.face_rec_ws = None
             self.latest_face_rec_state = None
             return False
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
                    "http://127.0.0.1:8001/voice/process",
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
        
    async def _process_video_frame_ws(self, img: np.ndarray):
        """Sends a video frame over WebSocket and updates the face rec state."""
        if not await self._ensure_face_rec_connection():
            print("Cannot process video frame, WebSocket connection failed.")
            return

        if self.face_rec_ws is None:
             print("Cannot process video frame, WebSocket is None.")
             return
        try:
            async with self.ws_lock:  # Enforce sequential access
                # Encode and send frame
                _, buffer = cv2.imencode('.jpg', img)
                if buffer is None:
                    print("Error: cv2.imencode failed.")
                    return
                
                # debug_image_path = f"debug_frame_{int(time.time())}.jpg"
                # cv2.imwrite(debug_image_path, img)
                # print(f"Saved debug image to {debug_image_path}")
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                
                message = json.dumps({
                    "image": image_base64,
                    "timestamp": time.time()
                })

                await self.face_rec_ws.send(message)
                
                # Wait for response (now protected by lock)
                response_str = await asyncio.wait_for(self.face_rec_ws.recv(), timeout=5.0)
                response_data = json.loads(response_str)
                if response_data.get("status") == "error":
                    logger.error(f"Face recognition error: {response_data.get('error')}")
                    return

                self.latest_face_rec_state = FaceRecognitionResponse(**response_data) 
                
        except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.ConnectionClosedOK) as e:
            print(f"Face Rec WebSocket connection closed during send/recv: {e}")
            self.face_rec_ws = None
            self.latest_face_rec_state = None
        except json.JSONDecodeError as e:
            print(f"Failed to decode Face Rec JSON response: {e} - Response: {response_str}")
        except asyncio.TimeoutError:
            print("Timeout waiting for face recognition response.")
        except Exception as e:
            print(f"Error during Face Rec WebSocket send/recv/process: {type(e).__name__}: {e}")
    
    async def process_input(self, audio_data, img_data):
        """
        Processes audio and video input, identifies user using latest face state.
        """
        voice_user_result = None
        if audio_data:
            voice_user_result = await self.process_audio(audio_data)
            if voice_user_result and voice_user_result.transcription:
                self.transcription += voice_user_result.transcription

        print("transcription: ", self.transcription)
        
        if img_data is not None:
            await self._process_video_frame_ws(img_data)

        current_face_state = self.latest_face_rec_state.copy()

        if voice_user_result and current_face_state.matches:
            self.user_id = await self.identify_user(voice_user_result, current_face_state)
            print("User_ID:", self.user_id)
        else:
            print("process_input: No voice result or face state with matches available for identification.")



    async def identify_user(self, voice_user:VoiceRecognitionResponse, face_user:FaceRecognitionResponse) -> str:
        """
        Identify the user based on voice and face recognition results.
        """
        
        voice_id = voice_user.userid
        face_matches = face_user.matches
        processed_faces = len(face_matches)
        face_detected = face_user.face_detected

        known_face_matches = [m for m in face_matches if m.person_id != "Unknown" and m.confidence is not None]
        unknown_face_count = len([m for m in face_matches if m.person_id == "Unknown"])

        # Scenario 1: No voice
        if not voice_id:
            print("SCENARIO 1: NO VOICE")
            if len(face_matches) == 1:
                face = face_matches[0]
                if face.person_id == "Unknown":
                    print("SUB-SCENARIO 1.1: ONE UNKNOWN FACE DETECTED")
                    new_id = uuid.uuid4()
                    await add_voice_user(new_id, self.audio)
                    await add_face_user(new_id, self.image)
                    return str(new_id)
                else:
                    # 1 Known Face + unknown voice
                    print("SUB-SCENARIO 1.2: ONE KNOWN FACE DETECTED, UNKNOWN VOICE")
                    # TODO: possibly check confidence and speaker location and add to voice
                    # await add_voice_user(uuid.UUID(face.person_id), self.audio)
                    return face.person_id
            else:
                print("SUB-SCENARIO 1.3: MULTIPLE FACES DETECTED, NO VOICE")
                return None

        # Scenario 3: 1 face match (recognized or unrecognized) and voice recognized
        elif voice_id and processed_faces == 1:
            print("SCENARIO 3: ONE FACE MATCH AND VOICE RECOGNIZED")
            # Subscenario 3.1: Face not recognized
            if unknown_face_count == 1 and not known_face_matches:
                print("SUB-SCENARIO 3.1: FACE NOT RECOGNIZED")
                await add_face_user(voice_id, self.image)
                return str(voice_id)
            # Subscenario 3.2: Face recognized
            elif len(known_face_matches) == 1:
                print("SUB-SCENARIO 3.2: FACE RECOGNIZED")
                if known_face_matches[0].person_id == str(voice_id):
                    print("SUB-SCENARIO 3.2.1: FACE ID == VOICE ID")
                    return known_face_matches[0].person_id
                else:
                    print("SUB-SCENARIO 3.2.2: FACE ID != VOICE ID")
                    # TODO: possibly check speaker location pre update
                    await update_face_user(voice_id, self.image)
                    return str(voice_id)

        # Scenario 4: Multiple face matches (recognized and/or unrecognized) and voice recognized
        elif voice_id and processed_faces > 1:
            print("SCENARIO 4: MULTIPLE FACE MATCHES AND VOICE RECOGNIZED")
            face_ids = [match.person_id for match in known_face_matches]
            # Subscenario 4.1: All faces not recognized
            if unknown_face_count == processed_faces and not known_face_matches:
                print("SUB-SCENARIO 4.1: ALL FACES NOT RECOGNIZED")
                await add_face_user(voice_id, self.image)
                return str(voice_id)
            # Subscenario 4.2: Mixed recognition on face and voice recognized
            elif known_face_matches and unknown_face_count > 0:
                print("SUB-SCENARIO 4.2: MIXED RECOGNITION ON FACE AND VOICE RECOGNIZED")
                if str(voice_id) in face_ids:
                    print("SUB-SCENARIO 4.2.1: VOICE ID IN LIST OF FACE IDs")
                    return str(voice_id)
                else:
                    print("SUB-SCENARIO 4.2.2: VOICE ID NOT IN LIST OF FACE IDs")
                    await add_face_user(voice_id, self.image)
                    return str(voice_id)
            # Subscenario 4.3: All recognized faces and voice recognized
            elif len(known_face_matches) == processed_faces and unknown_face_count == 0:
                print("SUB-SCENARIO 4.3: ALL RECOGNIZED FACES AND VOICE RECOGNIZED")
                if str(voice_id) in face_ids:
                    print("SUB-SCENARIO 4.3.1: VOICE ID IN LIST OF FACE IDs")
                    return str(voice_id)
                else:
                    print("SUB-SCENARIO 4.3.2: VOICE ID NOT IN LIST OF FACE IDs")
                    print(f"# Handle updating face associated with voice_id {voice_id} when multiple recognized faces are present: {face_ids}")
                    return str(voice_id)

        # Scenario: No face detected (skipped for now)
        elif not face_detected or processed_faces == 0:
            print("SCENARIO 5: NO FACE DETECTED (SKIPPED)")
            if voice_id:
                print("SUB-SCENARIO 5.1: VOICE IDENTIFIED, NO FACE DETECTED")
                return str(voice_id)
            else:
                print("SUB-SCENARIO 5.2: NO VOICE OR FACE DETECTED")
                return None

        print("UNHANDLED SCENARIO.")
        return None

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
        

        if audio_payload and self.transcription != "":
            print("send transcription to RAG")
            answer = await answer_user_query(GenerateRequest(user_id = self.user_id if self.user_id else str(uuid.uuid4()), question = self.transcription))
            print("Answer from RAG: ", answer.generation)
            self.transcription = ""

            start_time_tts = time.time()
            response_tts = await generate_tts(answer.generation)
            end_time_tts = time.time()
            
            print(f'Total TTS TIme: ' , end_time_tts - start_time_tts)
            return response_tts
        else:
            print("No transcription available")
            return None
        
        
    async def process_video(self, img_payload):

        if not img_payload:
            print("Missing video payload; skipping this message.")
            return None

        img = None

        try:
            video_bytes = base64.b64decode(img_payload)
            np_arr = np.frombuffer(video_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            print("Error decoding video payload:", e)

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

        # Pass the converted UploadFile objects to process_input
        await self._process_video_frame_ws(img)
        
    async def close(self):
        """Closes the WebSocket connection."""
        print("Close requested. Shutting down WebSocket connection...")
        if self.face_rec_ws and (await self.face_rec_ws.ping()):
             try:
                  await self.face_rec_ws.send(json.dumps({"action": "close"}))
                  await self.face_rec_ws.close()
                  print("Face Rec WebSocket connection closed.")
             except Exception as e:
                  print(f"Error closing WebSocket: {e}")
             finally:
                  self.face_rec_ws = None
                  self.latest_face_rec_state = {}
        else:
             print("WebSocket connection already closed or never established.")