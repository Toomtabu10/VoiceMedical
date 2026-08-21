"""
Exercises the full intake flow with stt.transcribe and the Gemini client
mocked out, since real Whisper/Gemini network calls aren't available in
this sandbox.
Run with: python test_flow.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

# Clean slate
if os.path.exists("patients.db"):
    os.remove("patients.db")

from unittest.mock import patch
from fastapi.testclient import TestClient

os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")

# Mock stt.transcribe and gemini_client BEFORE importing main
import app.stt as stt_module

TRANSCRIPT_1 = "This is Priya Sharma, born March 4th 1990, blood group O positive."
TRANSCRIPT_2 = "I have a penicillin allergy, causes a rash, moderate severity. No current medications. No known conditions."

EXTRACTED_1 = {
    "name": "Priya Sharma",
    "dob": "1990-03-04",
    "blood_group": "O+",
}
EXTRACTED_2 = {
    "allergies": [{"substance": "Penicillin", "reaction": "rash", "severity": "moderate"}],
    "medications_none_reported": True,
    "conditions_none_reported": True,
}

call_count = {"n": 0}

def fake_transcribe(audio_bytes, filename_hint="note.wav"):
    call_count["n"] += 1
    return TRANSCRIPT_1 if call_count["n"] == 1 else TRANSCRIPT_2

def fake_extract(transcript, existing_profile, model=None):
    return EXTRACTED_1 if transcript == TRANSCRIPT_1 else EXTRACTED_2

def fake_medical_chat(patient, user_message, recent_messages, model=None, stream=False):
    if stream:
        def gen():
            yield "mock "
            yield "streamed "
            yield "reply"
        return gen()
    return "Mock general-info reply based on profile."

stt_module.transcribe = fake_transcribe

import app.gemini_client as gc
gc.extract_fields_from_transcript = fake_extract
gc.medical_chat = fake_medical_chat

from app.main import app as fastapi_app

client = TestClient(fastapi_app)

def section(title):
    print(f"\n=== {title} ===")

# 1. Create patient
section("Create patient")
r = client.post("/patients", json={})
assert r.status_code == 200, r.text
patient = r.json()
pid = patient["id"]
print("Created patient", pid, "is_complete:", patient["is_complete"], "missing:", patient["missing_fields"])
assert patient["is_complete"] is False
assert set(patient["missing_fields"]) == {"name", "dob", "blood_group", "allergies_reviewed", "medications_reviewed", "conditions_reviewed"}

# 2. First voice note -> partial fields
section("Voice intake #1 (name/dob/blood group)")
r = client.post(f"/patients/{pid}/voice-intake", files={"file": ("note1.wav", b"fakebytes", "audio/wav")})
assert r.status_code == 200, r.text
data = r.json()
print("Transcript:", data["transcript"])
print("Extracted:", data["extracted_fields"])
print("Missing after #1:", data["missing_fields"])
print("Follow-up prompt:", data["follow_up_prompt"])
assert data["is_complete"] is False
assert "allergies_reviewed" in data["missing_fields"]
assert data["profile"]["name"] == "Priya Sharma"
assert data["profile"]["dob"] == "1990-03-04"
assert data["profile"]["blood_group"] == "O+"

# 3. Second voice note -> completes profile, triggers AI summary
section("Voice intake #2 (allergies/meds/conditions)")
r = client.post(f"/patients/{pid}/voice-intake", files={"file": ("note2.wav", b"fakebytes", "audio/wav")})
assert r.status_code == 200, r.text
data = r.json()
print("Transcript:", data["transcript"])
print("Missing after #2:", data["missing_fields"])
print("is_complete:", data["is_complete"])
print("AI summary:", data["ai_summary"])
assert data["is_complete"] is True
assert data["missing_fields"] == []
assert data["ai_summary"] is not None
assert len(data["profile"]["allergies"]) == 1
assert data["profile"]["allergies"][0]["substance"] == "Penicillin"
assert data["profile"]["medications_reviewed"] is True
assert data["profile"]["medications"] == []

# 4. Doctor chat
section("Doctor chat")
r = client.post(f"/patients/{pid}/chat", json={"message": "Any interaction concerns with future antibiotics?", "stream": False})
assert r.status_code == 200, r.text
print("AI reply:", r.json()["reply"])

# 5. Full message history -- confirm separation of profile vs conversation
section("Message history (separate table)")
r = client.get(f"/patients/{pid}/messages")
assert r.status_code == 200, r.text
messages = r.json()
roles = [m["role"] for m in messages]
print("Roles in order:", roles)
assert "patient_voice" in roles
assert "system" in roles
assert "ai" in roles
assert "doctor" in roles

# 6. Duplicate-name sanity check (not enforced -- matches ToomMed pattern discussion)
section("Second patient, same name allowed (by design -- disambiguated by ID)")
r = client.post("/patients", json={"name": "Priya Sharma"})
assert r.status_code == 200
print("Second 'Priya Sharma' created as patient", r.json()["id"], "-- distinct from", pid)

# 7. Correction: fix a misheard DOB directly (extraction never overwrites, so this is the only way)
section("Correction: PATCH scalar field")
r = client.patch(f"/patients/{pid}", json={"dob": "1990-03-05"})
assert r.status_code == 200, r.text
data = r.json()
print("DOB after correction:", data["dob"])
assert data["dob"] == "1990-03-05"

# 8. Correction: fix a wrongly-extracted allergy (edit) and add a missed one (create)
section("Correction: allergy edit + add")
allergy_id = client.get(f"/patients/{pid}").json()["allergies"][0]["id"]
r = client.put(f"/patients/{pid}/allergies/{allergy_id}", json={"severity": "high"})
assert r.status_code == 200, r.text
print("Allergy severity corrected to:", r.json()["severity"])

r = client.post(f"/patients/{pid}/allergies", json={"substance": "Latex", "reaction": "hives", "severity": "moderate"})
assert r.status_code == 200, r.text
print("Added missed allergy:", r.json()["substance"])

# 9. Correction: delete a wrongly-recorded medication
section("Correction: add then delete a medication")
r = client.post(f"/patients/{pid}/medications", json={"name": "Ibuprofen", "dosage": "200mg"})
med_id = r.json()["id"]
r = client.delete(f"/patients/{pid}/medications/{med_id}")
assert r.status_code == 204, r.text
print("Deleted medication", med_id)

# 10. Correction: mark a whole category for re-review
section("Correction: reset conditions for re-review")
r = client.post(f"/patients/{pid}/review-reset", json={"field": "conditions"})
assert r.status_code == 200, r.text
data = r.json()
print("conditions_reviewed now:", data["conditions_reviewed"], "is_complete now:", data["is_complete"])
assert data["conditions_reviewed"] is False
assert data["is_complete"] is False

# 11. Confirm every correction landed in the audit trail as a CORRECTION message
section("Audit trail check")
r = client.get(f"/patients/{pid}/messages")
correction_msgs = [m["content"] for m in r.json() if m["role"] == "correction"]
for c in correction_msgs:
    print(" -", c)
assert len(correction_msgs) >= 5

print("\nAll checks passed.")
