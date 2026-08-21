import enum
from datetime import datetime, date

from sqlalchemy import (
    Column, Integer, String, Date, Boolean, ForeignKey, DateTime, Text, Enum
)
from sqlalchemy.orm import relationship

from app.database import Base


class Patient(Base):
    """
    The structured patient profile. Only ever changed by the extraction
    step (from voice) or a direct, explicit edit endpoint -- never
    silently inferred from the free-form medical chat.
    """
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=True)
    dob = Column(Date, nullable=True)
    blood_group = Column(String, nullable=True)
    sex = Column(String, nullable=True)

    # "Reviewed" flags let us distinguish "explicitly confirmed empty"
    # from "we just haven't asked yet" -- both look empty in a table,
    # but only the former should count as profile-complete.
    allergies_reviewed = Column(Boolean, default=False)
    medications_reviewed = Column(Boolean, default=False)
    conditions_reviewed = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    allergies = relationship("Allergy", back_populates="patient", cascade="all, delete-orphan")
    medications = relationship("Medication", back_populates="patient", cascade="all, delete-orphan")
    conditions = relationship("Condition", back_populates="patient", cascade="all, delete-orphan")
    messages = relationship("Message", back_populates="patient", cascade="all, delete-orphan")

    @property
    def is_complete(self) -> bool:
        from app.config import REQUIRED_SCALAR_FIELDS, REQUIRED_REVIEWED_FLAGS
        for f in REQUIRED_SCALAR_FIELDS:
            if not getattr(self, f):
                return False
        for f in REQUIRED_REVIEWED_FLAGS:
            if not getattr(self, f):
                return False
        return True

    @property
    def missing_fields(self):
        from app.config import REQUIRED_SCALAR_FIELDS, REQUIRED_REVIEWED_FLAGS
        missing = []
        for f in REQUIRED_SCALAR_FIELDS:
            if not getattr(self, f):
                missing.append(f)
        for f in REQUIRED_REVIEWED_FLAGS:
            if not getattr(self, f):
                missing.append(f)
        return missing


class Allergy(Base):
    __tablename__ = "allergies"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    substance = Column(String, nullable=False)
    reaction = Column(String, nullable=True)
    severity = Column(String, nullable=True)

    patient = relationship("Patient", back_populates="allergies")


class Medication(Base):
    __tablename__ = "medications"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    name = Column(String, nullable=False)
    dosage = Column(String, nullable=True)
    frequency = Column(String, nullable=True)
    active = Column(Boolean, default=True)  # False = past medication, not current

    patient = relationship("Patient", back_populates="medications")


class Condition(Base):
    __tablename__ = "conditions"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    name = Column(String, nullable=False)
    notes = Column(String, nullable=True)

    patient = relationship("Patient", back_populates="conditions")


class MessageRole(str, enum.Enum):
    SYSTEM = "system"          # system prompts to the patient/doctor (e.g. "please tell me your DOB")
    PATIENT_VOICE = "patient_voice"  # transcribed voice note content
    DOCTOR = "doctor"          # typed input from a clinician/operator
    AI = "ai"                  # medical-LLM response
    CORRECTION = "correction"  # an explicit edit/delete to the structured profile, logged for audit


class Message(Base):
    """
    Every doctor message, system prompt, transcribed voice note, and AI
    reply lives here -- deliberately separate from the structured profile
    tables above, so the free-form conversation never silently overwrites
    a clinical fact.
    """
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    role = Column(Enum(MessageRole), nullable=False)
    content = Column(Text, nullable=False)
    audio_path = Column(String, nullable=True)  # set when role == PATIENT_VOICE
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="messages")
