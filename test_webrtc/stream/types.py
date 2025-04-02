from pydantic import BaseModel
from uuid import UUID

class VoiceRecognitionResponse(BaseModel):
    userid: UUID | None
    transcription: str
    score: float

class FaceRecognitionResponse(BaseModel):
    userid: str
    score: float
    is_new: bool


class GenerateRequest(BaseModel):
    user_id: str
    question: str

class RAGResponse(BaseModel):
    generation: str