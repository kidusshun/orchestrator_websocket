from pydantic import BaseModel
from typing import List, Tuple, Optional
from uuid import UUID

class VoiceRecognitionResponse(BaseModel):
    userid: UUID | None
    transcription: str
    score: float


class Match(BaseModel):
    person_id: str
    confidence: Optional[float] = None
    bbox: Optional[List[float]] = None
    
class FaceRecognitionResponse(BaseModel):
   matches: List[Match]
   face_detected: bool
   processed_faces: int
   status: str
   tracked: Optional[List[str]] = []
   lip_center: Optional[List[float]] = []
   error: Optional[str] = None


class GenerateRequest(BaseModel):
    user_id: str
    question: str

class RAGResponse(BaseModel):
    generation: str

class CreateVoiceUserResponse(BaseModel):
    user_id: UUID

class CreateFaceUserResponse(BaseModel):
    user_id: str
