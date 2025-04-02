import httpx
import requests
from .types import RAGResponse, GenerateRequest

async def generate_tts(
    text: str,
):
    try:
        url = "http://localhost:8000/voice/tts"
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