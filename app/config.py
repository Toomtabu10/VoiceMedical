"""
Central configuration: model names, required-field list, and the two
system prompts used by the LLM (one for structured extraction, one for
the medical-info chat).
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Gemini ---------------------------------------------------------------
# Single model handles both structured extraction and the medical chat
# (matches the .env this project is configured with). Swap to two
# different model names here if you'd rather split extraction onto a
# cheaper/faster model and keep a stronger one for the medical chat.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Generation parameters, read from .env. Used for the medical chat.
# Extraction always runs at temperature 0 regardless of this setting,
# since structured-field extraction should be as deterministic as
# possible -- see app/gemini_client.py.
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.3"))
MODEL_TOP_P = float(os.getenv("MODEL_TOP_P", "0.9"))
# Gemini's context window is fixed per-model (not a request-time knob the
# way Ollama's num_ctx is), so this isn't passed to the API. Instead it's
# used as a rough budget for how much conversation history we send on
# each chat call -- see build_history_within_budget() below.
MODEL_NUM_CTX = int(os.getenv("MODEL_NUM_CTX", "4096"))

# --- Speech-to-text ------------------------------------------------------
# "faster-whisper" runs fully offline once the model is downloaded once.
# Swap to another backend by editing app/stt.py; the rest of the app only
# talks to the transcribe() function, not to whisper directly.
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# --- Required fields for a "complete" profile ---------------------------
# A field being "reviewed" is distinct from it being empty: e.g. a patient
# can have NO allergies, but we still want that to be an explicit answer
# ("none reported") rather than a silently-missing field.
REQUIRED_SCALAR_FIELDS = ["name", "dob", "blood_group"]
REQUIRED_REVIEWED_FLAGS = ["allergies_reviewed", "medications_reviewed", "conditions_reviewed"]

FIELD_PROMPTS = {
    "name": "the patient's full name",
    "dob": "the patient's date of birth (YYYY-MM-DD)",
    "blood_group": "the patient's blood group (e.g. O+, A-, unknown if untested)",
    "allergies_reviewed": "any known allergies (or confirmation of none)",
    "medications_reviewed": "any current or past medications (or confirmation of none)",
    "conditions_reviewed": "any existing medical conditions (or confirmation of none)",
}

# --- Extraction prompt ---------------------------------------------------
EXTRACTION_SYSTEM_PROMPT = """You are a structured-data extraction engine for a local
medical intake system. You will be given a voice-note transcript and the
patient's current known profile (which may be partially filled).

Extract ONLY facts explicitly stated in the transcript. Never infer, guess,
or fabricate a value. If something is not clearly stated, leave it out of
the JSON entirely (do not include the key).

Respond with ONLY a single JSON object, no prose, no markdown fences, no
explanation. Use this exact shape (omit any key not mentioned):

{
  "name": string,
  "dob": "YYYY-MM-DD",
  "blood_group": string,
  "sex": string,
  "allergies": [{"substance": string, "reaction": string, "severity": string}],
  "allergies_none_reported": bool,
  "medications": [{"name": string, "dosage": string, "frequency": string}],
  "medications_none_reported": bool,
  "conditions": [{"name": string, "notes": string}],
  "conditions_none_reported": bool
}

Rules:
- "allergies_none_reported" should be true ONLY if the speaker explicitly
  says they have no allergies. Same logic for medications/conditions.
- Dates must be normalized to YYYY-MM-DD if a full date is given; if only
  partial info is given (e.g. just a year), omit "dob" rather than guess.
- Do not invent dosages, reactions, or severities that weren't stated.
"""

# --- Safety framing for the medical-info chat (reused from the ToomMed
# pattern: general info, flag emergencies, never confident-guess dosing) ---
MEDICAL_CHAT_SYSTEM_PROMPT = """You are a local, offline medical-information
assistant. You give general, educational information only. You do NOT
diagnose, prescribe, or replace a licensed clinician.

- If the user's message suggests a medical emergency, say so plainly and
  recommend contacting emergency services immediately, before anything else.
- Do not state confident diagnoses. Offer general, educational information
  and encourage follow-up with a clinician for anything specific to the
  patient's case.
- Do not invent or guess specific prescription dosing.
- You will be given the patient's structured profile (DOB, blood group,
  allergies, medications, conditions) as context. Use it to make your
  general information relevant, but do not silently add to or change it.
"""

EMERGENCY_KEYWORDS = [
    "chest pain", "can't breathe", "cannot breathe", "difficulty breathing",
    "suicidal", "suicide", "want to die", "unconscious", "unresponsive",
    "severe bleeding", "stroke", "numb on one side", "slurred speech",
    "overdose", "anaphylaxis", "throat closing", "seizure",
]

EMERGENCY_BANNER = (
    "\u26a0\ufe0f This may describe a medical emergency. If this is an emergency, "
    "call your local emergency number or go to the nearest emergency room now. "
    "The information below is general and not a substitute for emergency care.\n\n"
)
