from datetime import date, datetime
from typing import Optional, List
from pydantic import BaseModel, ConfigDict


class AllergyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    substance: str
    reaction: Optional[str] = None
    severity: Optional[str] = None


class MedicationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    active: bool


class ConditionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    notes: Optional[str] = None


class PatientOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: Optional[str] = None
    dob: Optional[date] = None
    blood_group: Optional[str] = None
    sex: Optional[str] = None
    allergies_reviewed: bool
    medications_reviewed: bool
    conditions_reviewed: bool
    allergies: List[AllergyOut] = []
    medications: List[MedicationOut] = []
    conditions: List[ConditionOut] = []
    is_complete: bool
    missing_fields: List[str] = []


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    role: str
    content: str
    audio_path: Optional[str] = None
    created_at: datetime


class VoiceIntakeResponse(BaseModel):
    patient_id: int
    transcript: str
    extracted_fields: dict
    profile: PatientOut
    is_complete: bool
    missing_fields: List[str]
    follow_up_prompt: Optional[str] = None
    ai_summary: Optional[str] = None


class DoctorChatRequest(BaseModel):
    message: str
    stream: bool = False


class DoctorChatResponse(BaseModel):
    reply: str


class PatientCreate(BaseModel):
    name: Optional[str] = None


class PatientUpdate(BaseModel):
    """
    For direct corrections to scalar fields -- e.g. extraction mis-heard
    the DOB from a voice note. Only fields explicitly included get
    changed; omit a field to leave it untouched.
    """
    name: Optional[str] = None
    dob: Optional[date] = None
    blood_group: Optional[str] = None
    sex: Optional[str] = None


class ReviewResetRequest(BaseModel):
    field: str  # "allergies" | "medications" | "conditions"


class AllergyIn(BaseModel):
    substance: str
    reaction: Optional[str] = None
    severity: Optional[str] = None


class AllergyUpdate(BaseModel):
    substance: Optional[str] = None
    reaction: Optional[str] = None
    severity: Optional[str] = None


class MedicationIn(BaseModel):
    name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    active: bool = True


class MedicationUpdate(BaseModel):
    name: Optional[str] = None
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    active: Optional[bool] = None


class ConditionIn(BaseModel):
    name: str
    notes: Optional[str] = None


class ConditionUpdate(BaseModel):
    name: Optional[str] = None
    notes: Optional[str] = None
