import json
import re

from google import genai
from google.genai import types

from app.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    MODEL_TEMPERATURE,
    MODEL_TOP_P,
    MODEL_NUM_CTX,
    EXTRACTION_SYSTEM_PROMPT,
    MEDICAL_CHAT_SYSTEM_PROMPT,
    EMERGENCY_KEYWORDS,
    EMERGENCY_BANNER,
)

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in."
    )

_client = genai.Client(api_key=GEMINI_API_KEY)

# Rough chars-per-token estimate used only to budget how much conversation
# history we include per request (see _build_history_within_budget) -- not
# passed to the API, since Gemini's context window is fixed per-model.
_CHARS_PER_TOKEN_ESTIMATE = 4


def contains_emergency_language(text: str) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in EMERGENCY_KEYWORDS)


def _extract_json_object(raw: str) -> dict:
    """
    Models don't always respect 'JSON only' instructions perfectly --
    strip markdown fences and pull out the first {...} block defensively
    rather than trusting json.loads on the raw string.
    """
    text = raw.strip()
    text = re.sub(r"^```(json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {raw!r}")
    return json.loads(match.group(0))


def extract_fields_from_transcript(transcript: str, existing_profile: dict, model: str = None) -> dict:
    """
    Calls Gemini with the transcript + current known profile and returns
    a dict of ONLY the fields the model found explicitly stated. Never
    overwrites -- merging into the DB is the caller's job (app/intake.py).
    Runs at temperature 0 regardless of MODEL_TEMPERATURE: extraction is a
    structured-output task where determinism matters more than variety.
    """
    model = model or GEMINI_MODEL
    user_content = (
        f"Current known profile (may be partially filled):\n"
        f"{json.dumps(existing_profile, default=str)}\n\n"
        f"New voice-note transcript:\n\"\"\"\n{transcript}\n\"\"\""
    )
    response = _client.models.generate_content(
        model=model,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=EXTRACTION_SYSTEM_PROMPT,
            temperature=0,
        ),
    )
    raw = response.text or ""
    try:
        return _extract_json_object(raw)
    except (ValueError, json.JSONDecodeError):
        # Fail safe: treat unparseable output as "nothing extracted" rather
        # than crash the request -- a bad extraction pass should never take
        # down the intake flow, and no field is safer than a malformed one.
        return {}


def build_profile_context(patient) -> str:
    lines = [f"Name: {patient.name or 'unknown'}"]
    lines.append(f"DOB: {patient.dob or 'unknown'}")
    lines.append(f"Blood group: {patient.blood_group or 'unknown'}")
    if patient.allergies:
        lines.append("Allergies: " + "; ".join(
            f"{a.substance} ({a.reaction or 'reaction unspecified'}, severity: {a.severity or 'unspecified'})"
            for a in patient.allergies
        ))
    elif patient.allergies_reviewed:
        lines.append("Allergies: none reported")
    if patient.medications:
        lines.append("Medications: " + "; ".join(
            f"{m.name} {m.dosage or ''} {m.frequency or ''}".strip() for m in patient.medications
        ))
    elif patient.medications_reviewed:
        lines.append("Medications: none reported")
    if patient.conditions:
        lines.append("Conditions: " + "; ".join(c.name for c in patient.conditions))
    elif patient.conditions_reviewed:
        lines.append("Conditions: none reported")
    return "\n".join(lines)


def _build_history_within_budget(recent_messages, reserved_chars: int) -> list:
    """
    Walks recent_messages from most-recent backwards, keeping as many as
    fit under MODEL_NUM_CTX (converted to a rough character budget), after
    reserving space for the system prompt/profile/new message. Returns
    them back in chronological order as google.genai Content objects.
    """
    budget_chars = max(0, (MODEL_NUM_CTX * _CHARS_PER_TOKEN_ESTIMATE) - reserved_chars)
    kept = []
    used = 0
    for m in reversed(recent_messages):
        cost = len(m.content) + 16  # small per-message overhead
        if used + cost > budget_chars:
            break
        kept.append(m)
        used += cost
    kept.reverse()

    history = []
    for m in kept:
        role = "model" if m.role.value == "ai" else "user"
        history.append(types.Content(role=role, parts=[types.Part(text=m.content)]))
    return history


def medical_chat(patient, user_message: str, recent_messages, model: str = None, stream: bool = False):
    """
    Runs the actual medical-info chat. `recent_messages` is a list of
    Message ORM rows (already role-filtered/ordered oldest-to-newest by
    caller) used to give the model conversational context, trimmed to fit
    MODEL_NUM_CTX.
    """
    model = model or GEMINI_MODEL

    profile_context = "Patient profile:\n" + build_profile_context(patient)
    system_instruction = MEDICAL_CHAT_SYSTEM_PROMPT + "\n\n" + profile_context

    reserved = len(system_instruction) + len(user_message) + 200
    history = _build_history_within_budget(recent_messages, reserved_chars=reserved)
    contents = history + [types.Content(role="user", parts=[types.Part(text=user_message)])]

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=MODEL_TEMPERATURE,
        top_p=MODEL_TOP_P,
    )

    banner = EMERGENCY_BANNER if contains_emergency_language(user_message) else ""

    if stream:
        def gen():
            if banner:
                yield banner
            for chunk in _client.models.generate_content_stream(model=model, contents=contents, config=config):
                if chunk.text:
                    yield chunk.text
        return gen()

    response = _client.models.generate_content(model=model, contents=contents, config=config)
    return banner + (response.text or "")
