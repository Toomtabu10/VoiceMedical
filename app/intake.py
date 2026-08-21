from datetime import datetime

from sqlalchemy.orm import Session

from app import models
from app.config import FIELD_PROMPTS


def apply_extracted_fields(db: Session, patient: models.Patient, extracted: dict) -> None:
    """
    Merge freshly-extracted fields into the patient record.
    Scalar fields only fill if currently empty (never silently overwrite
    an existing value from a later, possibly-misheard voice note -- that
    should go through an explicit edit instead).
    List fields (allergies/medications/conditions) are appended, not
    replaced, since a second voice note might mention an additional item.
    """
    if extracted.get("name") and not patient.name:
        patient.name = extracted["name"]

    if extracted.get("dob") and not patient.dob:
        try:
            patient.dob = datetime.strptime(extracted["dob"], "%Y-%m-%d").date()
        except ValueError:
            pass  # malformed date from the model -- skip rather than crash

    if extracted.get("blood_group") and not patient.blood_group:
        patient.blood_group = extracted["blood_group"]

    if extracted.get("sex") and not patient.sex:
        patient.sex = extracted["sex"]

    allergies = extracted.get("allergies") or []
    if allergies:
        existing = {a.substance.lower() for a in patient.allergies}
        for a in allergies:
            substance = (a.get("substance") or "").strip()
            if substance and substance.lower() not in existing:
                patient.allergies.append(models.Allergy(
                    substance=substance,
                    reaction=a.get("reaction"),
                    severity=a.get("severity"),
                ))
        patient.allergies_reviewed = True
    elif extracted.get("allergies_none_reported"):
        patient.allergies_reviewed = True

    medications = extracted.get("medications") or []
    if medications:
        existing = {m.name.lower() for m in patient.medications}
        for m in medications:
            name = (m.get("name") or "").strip()
            if name and name.lower() not in existing:
                patient.medications.append(models.Medication(
                    name=name,
                    dosage=m.get("dosage"),
                    frequency=m.get("frequency"),
                ))
        patient.medications_reviewed = True
    elif extracted.get("medications_none_reported"):
        patient.medications_reviewed = True

    conditions = extracted.get("conditions") or []
    if conditions:
        existing = {c.name.lower() for c in patient.conditions}
        for c in conditions:
            name = (c.get("name") or "").strip()
            if name and name.lower() not in existing:
                patient.conditions.append(models.Condition(
                    name=name,
                    notes=c.get("notes"),
                ))
        patient.conditions_reviewed = True
    elif extracted.get("conditions_none_reported"):
        patient.conditions_reviewed = True

    db.add(patient)
    db.commit()
    db.refresh(patient)


def build_follow_up_prompt(missing_fields: list[str]) -> str:
    if not missing_fields:
        return ""
    asks = [FIELD_PROMPTS.get(f, f) for f in missing_fields]
    if len(asks) == 1:
        body = asks[0]
    else:
        body = "; ".join(asks[:-1]) + f"; and {asks[-1]}"
    return f"Thanks -- I still need a bit more before the profile is complete: {body}. Could you record a quick voice note covering that?"
