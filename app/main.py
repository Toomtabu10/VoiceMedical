from typing import List

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app import models, schemas, intake, gemini_client, stt
from app.database import Base, engine, get_db

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Voice Medical")


# ---------------------------------------------------------------- patients

@app.post("/patients", response_model=schemas.PatientOut)
def create_patient(payload: schemas.PatientCreate, db: Session = Depends(get_db)):
    patient = models.Patient(name=payload.name)
    db.add(patient)
    db.commit()
    db.refresh(patient)
    _log(db, patient.id, models.MessageRole.SYSTEM, "Patient record created.")
    return patient


@app.get("/patients", response_model=List[schemas.PatientOut])
def list_patients(db: Session = Depends(get_db)):
    return db.query(models.Patient).order_by(models.Patient.created_at.desc()).all()


@app.get("/patients/{patient_id}", response_model=schemas.PatientOut)
def get_patient(patient_id: int, db: Session = Depends(get_db)):
    patient = _get_patient_or_404(db, patient_id)
    return patient


@app.patch("/patients/{patient_id}", response_model=schemas.PatientOut)
def update_patient(patient_id: int, payload: schemas.PatientUpdate, db: Session = Depends(get_db)):
    """
    Direct correction of scalar fields (name, DOB, blood group, sex) --
    e.g. extraction misheard a date. Only fields explicitly present in
    the request body are changed. Every change is logged as a CORRECTION
    message with a before/after diff for audit purposes.
    """
    patient = _get_patient_or_404(db, patient_id)
    changes = payload.model_dump(exclude_unset=True)
    diffs = []
    for field, new_value in changes.items():
        old_value = getattr(patient, field)
        if old_value != new_value:
            diffs.append(f"{field}: {old_value!r} \u2192 {new_value!r}")
            setattr(patient, field, new_value)

    if diffs:
        db.add(patient)
        db.commit()
        db.refresh(patient)
        _log(db, patient.id, models.MessageRole.CORRECTION, "Profile corrected \u2014 " + "; ".join(diffs))

    return patient


@app.post("/patients/{patient_id}/review-reset", response_model=schemas.PatientOut)
def reset_review(patient_id: int, payload: schemas.ReviewResetRequest, db: Session = Depends(get_db)):
    """
    Marks a category (allergies/medications/conditions) as needing
    re-review -- e.g. the extraction pass got the whole section wrong
    and re-recording won't help since the merge logic never overwrites.
    This does NOT clear existing entries; pair it with individual
    DELETE calls if the entries themselves are wrong, or use this alone
    if you just want the app to prompt for the category again.
    """
    field_map = {
        "allergies": "allergies_reviewed",
        "medications": "medications_reviewed",
        "conditions": "conditions_reviewed",
    }
    if payload.field not in field_map:
        raise HTTPException(status_code=400, detail=f"field must be one of {list(field_map)}")

    patient = _get_patient_or_404(db, patient_id)
    setattr(patient, field_map[payload.field], False)
    db.add(patient)
    db.commit()
    db.refresh(patient)
    _log(db, patient.id, models.MessageRole.CORRECTION, f"{payload.field.capitalize()} marked for re-review.")
    return patient


# ------------------------------------------------------- allergy corrections

@app.post("/patients/{patient_id}/allergies", response_model=schemas.AllergyOut)
def add_allergy(patient_id: int, payload: schemas.AllergyIn, db: Session = Depends(get_db)):
    patient = _get_patient_or_404(db, patient_id)
    allergy = models.Allergy(patient_id=patient.id, **payload.model_dump())
    db.add(allergy)
    patient.allergies_reviewed = True
    db.add(patient)
    db.commit()
    db.refresh(allergy)
    _log(db, patient.id, models.MessageRole.CORRECTION, f"Allergy added manually: {allergy.substance}")
    return allergy


@app.put("/patients/{patient_id}/allergies/{allergy_id}", response_model=schemas.AllergyOut)
def update_allergy(patient_id: int, allergy_id: int, payload: schemas.AllergyUpdate, db: Session = Depends(get_db)):
    allergy = _get_allergy_or_404(db, patient_id, allergy_id)
    changes = payload.model_dump(exclude_unset=True)
    diffs = []
    for field, new_value in changes.items():
        old_value = getattr(allergy, field)
        if old_value != new_value:
            diffs.append(f"{field}: {old_value!r} \u2192 {new_value!r}")
            setattr(allergy, field, new_value)
    if diffs:
        db.add(allergy)
        db.commit()
        db.refresh(allergy)
        _log(db, patient_id, models.MessageRole.CORRECTION, f"Allergy corrected ({allergy.substance}) \u2014 " + "; ".join(diffs))
    return allergy


@app.delete("/patients/{patient_id}/allergies/{allergy_id}", status_code=204)
def delete_allergy(patient_id: int, allergy_id: int, db: Session = Depends(get_db)):
    allergy = _get_allergy_or_404(db, patient_id, allergy_id)
    substance = allergy.substance
    db.delete(allergy)
    db.commit()
    _log(db, patient_id, models.MessageRole.CORRECTION, f"Allergy removed: {substance} (was incorrectly recorded)")


# ----------------------------------------------------- medication corrections

@app.post("/patients/{patient_id}/medications", response_model=schemas.MedicationOut)
def add_medication(patient_id: int, payload: schemas.MedicationIn, db: Session = Depends(get_db)):
    patient = _get_patient_or_404(db, patient_id)
    medication = models.Medication(patient_id=patient.id, **payload.model_dump())
    db.add(medication)
    patient.medications_reviewed = True
    db.add(patient)
    db.commit()
    db.refresh(medication)
    _log(db, patient.id, models.MessageRole.CORRECTION, f"Medication added manually: {medication.name}")
    return medication


@app.put("/patients/{patient_id}/medications/{medication_id}", response_model=schemas.MedicationOut)
def update_medication(patient_id: int, medication_id: int, payload: schemas.MedicationUpdate, db: Session = Depends(get_db)):
    medication = _get_medication_or_404(db, patient_id, medication_id)
    changes = payload.model_dump(exclude_unset=True)
    diffs = []
    for field, new_value in changes.items():
        old_value = getattr(medication, field)
        if old_value != new_value:
            diffs.append(f"{field}: {old_value!r} \u2192 {new_value!r}")
            setattr(medication, field, new_value)
    if diffs:
        db.add(medication)
        db.commit()
        db.refresh(medication)
        _log(db, patient_id, models.MessageRole.CORRECTION, f"Medication corrected ({medication.name}) \u2014 " + "; ".join(diffs))
    return medication


@app.delete("/patients/{patient_id}/medications/{medication_id}", status_code=204)
def delete_medication(patient_id: int, medication_id: int, db: Session = Depends(get_db)):
    medication = _get_medication_or_404(db, patient_id, medication_id)
    name = medication.name
    db.delete(medication)
    db.commit()
    _log(db, patient_id, models.MessageRole.CORRECTION, f"Medication removed: {name} (was incorrectly recorded)")


# ----------------------------------------------------- condition corrections

@app.post("/patients/{patient_id}/conditions", response_model=schemas.ConditionOut)
def add_condition(patient_id: int, payload: schemas.ConditionIn, db: Session = Depends(get_db)):
    patient = _get_patient_or_404(db, patient_id)
    condition = models.Condition(patient_id=patient.id, **payload.model_dump())
    db.add(condition)
    patient.conditions_reviewed = True
    db.add(patient)
    db.commit()
    db.refresh(condition)
    _log(db, patient.id, models.MessageRole.CORRECTION, f"Condition added manually: {condition.name}")
    return condition


@app.put("/patients/{patient_id}/conditions/{condition_id}", response_model=schemas.ConditionOut)
def update_condition(patient_id: int, condition_id: int, payload: schemas.ConditionUpdate, db: Session = Depends(get_db)):
    condition = _get_condition_or_404(db, patient_id, condition_id)
    changes = payload.model_dump(exclude_unset=True)
    diffs = []
    for field, new_value in changes.items():
        old_value = getattr(condition, field)
        if old_value != new_value:
            diffs.append(f"{field}: {old_value!r} \u2192 {new_value!r}")
            setattr(condition, field, new_value)
    if diffs:
        db.add(condition)
        db.commit()
        db.refresh(condition)
        _log(db, patient_id, models.MessageRole.CORRECTION, f"Condition corrected ({condition.name}) \u2014 " + "; ".join(diffs))
    return condition


@app.delete("/patients/{patient_id}/conditions/{condition_id}", status_code=204)
def delete_condition(patient_id: int, condition_id: int, db: Session = Depends(get_db)):
    condition = _get_condition_or_404(db, patient_id, condition_id)
    name = condition.name
    db.delete(condition)
    db.commit()
    _log(db, patient_id, models.MessageRole.CORRECTION, f"Condition removed: {name} (was incorrectly recorded)")


# ------------------------------------------------------------ voice intake

@app.post("/patients/{patient_id}/voice-intake", response_model=schemas.VoiceIntakeResponse)
async def voice_intake(patient_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Accepts a voice note, transcribes it, extracts structured fields,
    merges them into the profile, and either:
      - asks a follow-up question if fields are still missing, or
      - runs the medical LLM for an initial general-info summary once the
        profile is complete.
    """
    patient = _get_patient_or_404(db, patient_id)

    audio_bytes = await file.read()
    try:
        transcript = stt.transcribe(audio_bytes, filename_hint=file.filename or "note.wav")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not transcribe audio: {e}")

    if not transcript:
        raise HTTPException(status_code=422, detail="Transcription returned empty text.")

    _log(db, patient.id, models.MessageRole.PATIENT_VOICE, transcript, audio_path=file.filename)

    existing_profile = _profile_snapshot(patient)
    try:
        extracted = gemini_client.extract_fields_from_transcript(transcript, existing_profile)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini extraction call failed: {e}")
    intake.apply_extracted_fields(db, patient, extracted)

    missing = patient.missing_fields
    follow_up = None
    ai_summary = None

    if missing:
        follow_up = intake.build_follow_up_prompt(missing)
        _log(db, patient.id, models.MessageRole.SYSTEM, follow_up)
    else:
        try:
            ai_summary = gemini_client.medical_chat(
                patient,
                user_message=(
                    "The intake profile is now complete. Based on the patient profile "
                    "above, give a brief general-information overview a clinician "
                    "might find useful as a starting point (not a diagnosis)."
                ),
                recent_messages=[],
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Gemini consult call failed: {e}")
        _log(db, patient.id, models.MessageRole.AI, ai_summary)

    return schemas.VoiceIntakeResponse(
        patient_id=patient.id,
        transcript=transcript,
        extracted_fields=extracted,
        profile=patient,
        is_complete=patient.is_complete,
        missing_fields=missing,
        follow_up_prompt=follow_up,
        ai_summary=ai_summary,
    )


# ------------------------------------------------------------ doctor chat

@app.post("/patients/{patient_id}/chat", response_model=schemas.DoctorChatResponse)
def doctor_chat(patient_id: int, payload: schemas.DoctorChatRequest, db: Session = Depends(get_db)):
    patient = _get_patient_or_404(db, patient_id)
    _log(db, patient.id, models.MessageRole.DOCTOR, payload.message)

    recent = (
        db.query(models.Message)
        .filter(models.Message.patient_id == patient.id)
        .filter(models.Message.role.in_([models.MessageRole.DOCTOR, models.MessageRole.AI]))
        .order_by(models.Message.created_at.desc())
        .limit(20)
        .all()
    )
    recent = list(reversed(recent))[:-1]  # drop the message we just logged, caller passes it separately

    if payload.stream:
        try:
            gen = gemini_client.medical_chat(patient, payload.message, recent, stream=True)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Gemini call failed: {e}")

        def wrapper():
            full = []
            for token in gen:
                full.append(token)
                yield token
            _log(db, patient.id, models.MessageRole.AI, "".join(full))

        return StreamingResponse(wrapper(), media_type="text/plain")

    try:
        reply = gemini_client.medical_chat(patient, payload.message, recent, stream=False)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini call failed: {e}")
    _log(db, patient.id, models.MessageRole.AI, reply)
    return schemas.DoctorChatResponse(reply=reply)


@app.get("/patients/{patient_id}/messages", response_model=List[schemas.MessageOut])
def get_messages(patient_id: int, db: Session = Depends(get_db)):
    _get_patient_or_404(db, patient_id)
    return (
        db.query(models.Message)
        .filter(models.Message.patient_id == patient_id)
        .order_by(models.Message.created_at.asc())
        .all()
    )


# --------------------------------------------------------------- helpers

def _get_patient_or_404(db: Session, patient_id: int) -> models.Patient:
    patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


def _get_allergy_or_404(db: Session, patient_id: int, allergy_id: int) -> models.Allergy:
    allergy = (
        db.query(models.Allergy)
        .filter(models.Allergy.id == allergy_id, models.Allergy.patient_id == patient_id)
        .first()
    )
    if not allergy:
        raise HTTPException(status_code=404, detail="Allergy not found for this patient")
    return allergy


def _get_medication_or_404(db: Session, patient_id: int, medication_id: int) -> models.Medication:
    medication = (
        db.query(models.Medication)
        .filter(models.Medication.id == medication_id, models.Medication.patient_id == patient_id)
        .first()
    )
    if not medication:
        raise HTTPException(status_code=404, detail="Medication not found for this patient")
    return medication


def _get_condition_or_404(db: Session, patient_id: int, condition_id: int) -> models.Condition:
    condition = (
        db.query(models.Condition)
        .filter(models.Condition.id == condition_id, models.Condition.patient_id == patient_id)
        .first()
    )
    if not condition:
        raise HTTPException(status_code=404, detail="Condition not found for this patient")
    return condition


def _log(db: Session, patient_id: int, role: models.MessageRole, content: str, audio_path: str = None):
    msg = models.Message(patient_id=patient_id, role=role, content=content, audio_path=audio_path)
    db.add(msg)
    db.commit()


def _profile_snapshot(patient: models.Patient) -> dict:
    return {
        "name": patient.name,
        "dob": str(patient.dob) if patient.dob else None,
        "blood_group": patient.blood_group,
        "sex": patient.sex,
        "allergies": [{"substance": a.substance, "reaction": a.reaction, "severity": a.severity} for a in patient.allergies],
        "medications": [{"name": m.name, "dosage": m.dosage, "frequency": m.frequency} for m in patient.medications],
        "conditions": [{"name": c.name, "notes": c.notes} for c in patient.conditions],
    }


# Serve a minimal static UI if present (see static/index.html)
import os
if os.path.isdir(os.path.join(os.path.dirname(__file__), "..", "static")):
    app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "static"), html=True), name="static")
