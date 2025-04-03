import httpx
import requests
from .types import RAGResponse, GenerateRequest, CreateVoiceUserResponse, CreateFaceUserResponse
import uuid
from fastapi import UploadFile

async def generate_tts(
    text: str,
):
    try:
        url = "http://127.0.0.1:8005/voice/tts"
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json={"text":text}, timeout=60.0)
        
        return response
    except:
        raise Exception("Can't generate speech")
    

async def answer_user_query(
    request: GenerateRequest,
):
    try:
        url = "http://localhost:8002/rag/query"
        response = requests.post(url, json={
            "user_id":request.user_id,
            "question":request.question,
        })
        res = response.json()
        return RAGResponse(**res)
    except Exception as e:
        raise e


async def add_voice_user(id: uuid.UUID, audio: UploadFile):
    try:
        url = "http://127.0.0.1:8005/voice/add_user"
        # Reset file pointer before sending
        await audio.seek(0)
        
        files = {"file": ("voice.wav", audio.file, audio.content_type)}
        data = {"user_id": str(id)}
        async with httpx.AsyncClient() as client:
            response = await client.post(url, files=files, data=data, timeout=60.0)

        return CreateVoiceUserResponse(**response.json())
    except:
        raise Exception("can't add voice user")


async def add_face_user(id: uuid.UUID, image: UploadFile):
    try:
        url = "http://127.0.0.1:8000/api/v1/embed"
        # Reset file pointer before sending
        await image.seek(0)
        files = {"image": ("image.jpg", image.file, image.content_type)}
        data = {"person_id": str(id)}
        async with httpx.AsyncClient() as client:
            response = await client.post(url, files=files, data=data, timeout=60.0)

        return CreateFaceUserResponse(user_id = response.json())
    except:
        raise Exception("can't add face user")