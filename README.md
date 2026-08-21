# Voice Medical Intake

A locally-run FastAPI + SQLite app that builds a structured patient
profile from spoken voice notes. It transcribes audio offline (faster-whisper),
asks Gemini to pull out structured fields, prompts for whatever's still
missing, and — once the profile is complete — calls Gemini again for a
general-information consult about the patient. Voice transcription stays
fully local/offline; only the text (transcript, extracted fields, chat
messages) goes to the Gemini API.

> **Not a medical device.** General, educational information only. Does
> not diagnose, prescribe, or replace a licensed clinician.

## How it works

```
voice note (audio)
      │
      ▼
  transcription (faster-whisper, offline)
      │
      ▼
  field extraction (Gemini, JSON-only prompt, temperature 0)
      │
      ▼
  merge into profile ──► still missing fields? ──► ask a follow-up question
      │                                                (record another note)
      ▼ (complete)
  Gemini consult ──► logged as an "ai" message
```

Two things are deliberately kept in **separate tables**:

1. **The structured profile** (`patients`, `allergies`, `medications`,
   `conditions`) — only ever written by the extraction step, and only
   ever *filled in*, never silently overwritten. A second voice note
   can add a new allergy but won't clobber an existing one.
2. **Everything said** (`messages`) — every transcribed voice note,
   every system follow-up question, every doctor message, and every AI
   reply. This is a full audit trail of the conversation, kept apart
   from the clinical facts so a chat message can never quietly become a
   medical record.

A profile counts as "complete" once name, DOB, and blood group are set
*and* allergies/medications/conditions have each been explicitly
reviewed (even if the honest answer is "none") — see
`REQUIRED_SCALAR_FIELDS` / `REQUIRED_REVIEWED_FLAGS` in `app/config.py`.

## 1. Get a Gemini API key

Grab one at <https://aistudio.google.com/apikey>. This project calls the
Gemini API over the network for both field extraction and the medical
consult — unlike the local-Ollama version, this part is **not** offline;
only speech-to-text stays local.

## 2. Set up the Python project

```
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then edit .env and paste in your real key
```

`.env` is already in `.gitignore` — don't commit your real key. If a key
has ever been pasted somewhere it could be logged or shared (a chat, a
ticket, a screenshot), treat it as compromised and rotate it in
[Google AI Studio](https://aistudio.google.com/apikey).

`.env` variables used:

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | required — your Gemini API key |
| `GEMINI_MODEL` | model for both extraction and the medical consult (default `gemini-3.5-flash-lite`) |
| `MODEL_TEMPERATURE` | sampling temperature for the medical consult (extraction always runs at 0 for determinism) |
| `MODEL_TOP_P` | nucleus sampling for the medical consult |
| `MODEL_NUM_CTX` | rough token budget used to decide how much chat history to include per request (Gemini's real context window is fixed per-model — this just bounds what *we* send) |

`faster-whisper` will download its model weights (~150MB for `base`) the
first time it runs, then works fully offline after that.

## 3. Run it

```
uvicorn app.main:app --reload --port 8000
```

Open **http://127.0.0.1:8000/** — create a patient, click **Start
recording**, speak a note (e.g. *"This is Priya Sharma, born March 4th
1990, blood group O positive"*), stop recording. The transcript and
extracted fields appear immediately; the completeness ring fills in as
fields land. Keep recording notes until the ring is full — the app will
tell you exactly what's still missing after each note. Once complete,
switch to the **Consult** tab to see the AI's initial summary and chat
further.

Swagger UI for raw API access: **http://127.0.0.1:8000/docs**

## Project structure

```
voice-medical-intake/
  app/
    config.py         # Gemini settings, required fields, system prompts, emergency keywords
    database.py        # SQLite engine/session
    models.py            # Patient, Allergy, Medication, Condition, Message tables
    schemas.py            # Pydantic request/response models
    stt.py                # speech-to-text, abstracted (default: faster-whisper)
    gemini_client.py       # Gemini calls (extraction + medical chat), emergency banner, history budgeting
    intake.py                # merges extracted fields into the profile, builds follow-up prompts
    main.py                    # FastAPI routes
  static/index.html            # single-page UI (recorder, profile view, chat)
  test_flow.py                  # end-to-end test with STT/Ollama mocked
  requirements.txt
  .env.example
```

## API summary

| Method | Path | Purpose |
|---|---|---|
| POST | `/patients` | Create a patient (name optional) |
| GET | `/patients` | List patients |
| GET | `/patients/{id}` | Get profile + completeness status |
| PATCH | `/patients/{id}` | **Correct** scalar fields (name/DOB/blood group/sex) — only fields you include change; logged as a `correction` message |
| POST | `/patients/{id}/review-reset` | **Correct** — mark a whole category (`allergies`/`medications`/`conditions`) as needing re-review; existing entries stay until edited/deleted individually |
| POST / PUT / DELETE | `/patients/{id}/allergies[/{allergy_id}]` | **Correct** — add, edit, or remove an individual allergy |
| POST / PUT / DELETE | `/patients/{id}/medications[/{medication_id}]` | **Correct** — add, edit, or remove an individual medication |
| POST / PUT / DELETE | `/patients/{id}/conditions[/{condition_id}]` | **Correct** — add, edit, or remove an individual condition |
| POST | `/patients/{id}/voice-intake` | Upload a voice note (multipart `file`) — transcribes, extracts, merges, and either returns a follow-up prompt or an AI summary |
| POST | `/patients/{id}/chat` | Doctor/operator chat with the medical LLM (`stream: true` for a plain-text streaming response) |
| GET | `/patients/{id}/messages` | Full conversation log (system/patient_voice/doctor/ai/**correction**) |

### Correcting a wrong extraction

The voice-intake merge logic never overwrites an existing field — it only
fills gaps or appends new list items. That's deliberate (a second,
possibly-misheard note shouldn't silently clobber a correct earlier
answer), but it also means **re-recording won't fix a wrong extraction**.
Use the correction endpoints instead, all reachable from the UI's
**Correct** button and the ✎ / ✕ icons next to each allergy/medication/
condition:

- **Wrong scalar value** (e.g. DOB off by a digit) → `PATCH /patients/{id}`
- **Wrong or incomplete list item** (e.g. severity misheard) → `PUT` the specific allergy/medication/condition
- **Extraction invented or duplicated an item** → `DELETE` it
- **A whole category was extracted wrong and needs redoing** → `POST /patients/{id}/review-reset` with `{"field": "allergies"}`, then either record a fresh voice note (the category will prompt again since it's no longer "reviewed") or add entries manually

Every correction is written to the `messages` table with role `correction`
and a before/after diff, so the full history of what was changed — not
just the current state — stays auditable in the **Full history** tab.

## Design notes / limitations

- **Extraction never overwrites, only fills gaps or appends.** If a voice
  note mischeard a name, use the correction endpoints (`PATCH`, or the
  per-item `PUT`/`DELETE`) rather than re-recording — see "Correcting a
  wrong extraction" above. Corrections are the intended fix path, not a
  workaround.
- **JSON parsing from the LLM is defensive, not naive.** Even hosted
  models don't always follow "output JSON only" perfectly —
  `gemini_client.py` strips markdown fences and pulls the first `{...}`
  block out rather than trusting the raw string; if nothing parses, that
  pass extracts nothing rather than crashing the request.
- **Gemini calls go over the network.** Transcription (faster-whisper)
  stays fully local; the transcript text, extracted fields, and chat
  messages are sent to Google's API. If that's not acceptable for your
  use case, swap `gemini_client.py` back for a local Ollama client (the
  previous version of this project used that pattern) — everything else
  (`intake.py`, `models.py`, `main.py`) is unaffected either way.
- **Emergency keyword check** runs on every doctor/consult message
  (reused from the same pattern as basic keyword pre-checks) and prepends
  a banner before the model's own reply — it's a basic net, not a
  clinical safety system.
- **Same-name patients are allowed by design**, same as most simple
  systems — they're disambiguated by ID. The UI shows the ID next to the
  name in the sidebar for this reason.
- **Not yet built, worth adding next:** authentication (this currently
  assumes single-user local use, like the original ToomMed project) and
  a duplicate-name warning at patient-creation time.
