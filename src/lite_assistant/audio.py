"""
The audio endpoints beyond OpenAI's: taking the noise out of speech, who is
speaking, with voices enrolled by name, the pitch for tuning an instrument,
and reading audio from a request as the sound endpoints take it.
"""
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, Request
from fastapi.responses import Response

from .api_common import APIError, run_engine

router = APIRouter()
# The longest audio noise removal takes, in seconds: it runs at about real time
MAX_DENOISE = 64
# The least audio a voice is enrolled or recognized from, in seconds
MIN_VOICE = 1.0
# The longest audio the tuner hears, in seconds: a pitch every 50 ms takes a second for each second
MAX_PITCH = 10


async def read_audio(request: Request) -> Tuple[bytes, Dict[str, Any]]:
    """
    The audio a request sends, as a multipart file field, as transcriptions
    takes it, or as the body itself (e.g. Content-Type: audio/wav), and its
    options, from form fields and query parameters.
    """
    options: Dict[str, Any] = dict(request.query_params)
    kind = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if kind == "multipart/form-data":
        form = await request.form()
        options.update({k: v for k, v in form.items() if isinstance(v, str)})
        upload = form.get("file")
        if upload is None or isinstance(upload, str):
            raise APIError(400, "The form needs the audio as a file field.")
        audio = await upload.read()
    elif kind.startswith("audio/") or kind == "application/octet-stream":
        audio = await request.body()
    else:
        raise APIError(400, "Send the audio as a multipart file or as an audio body.")
    if not audio:
        raise APIError(400, "The audio is empty.")
    return audio, options


def denoise_engine(e, model=None):
    """ The noise removal engine a request names, or the first. """
    if not e.denoise:
        raise APIError(500, "No noise removal engines loaded.", "server_error")
    return e.denoise.get(model) or next(iter(e.denoise.values()))


def denoised(engine, audio: bytes):
    """ Audio's samples with the noise taken out, at the engine's rate. Raises ValueError for audio it can't take. """
    from .engines.tflite import open_audio
    samples = open_audio(audio, engine.RATE)
    if len(samples) > MAX_DENOISE * engine.RATE:
        raise ValueError(f"noise removal takes up to {MAX_DENOISE} seconds; this is {len(samples) / engine.RATE:.0f}")
    return engine.denoise(samples)


@router.post("/v1/audio/denoise")
async def denoise(request: Request):
    """
    Speech with the background noise taken out, as a 16 kHz WAV: not an
    OpenAI endpoint. The audio comes as read_audio takes it, up to 64
    seconds; it takes about as long as the audio to clean.
    """
    from .engines.tflite import wav
    e = request.app.state.engines
    audio, options = await read_audio(request)
    engine = denoise_engine(e, options.get("model"))
    cleaned = await run_engine(lambda data: wav(denoised(engine, data), engine.RATE), audio, "Noise removal", "audio")
    return Response(cleaned, media_type="audio/wav")


def speaker_engine(e):
    if not e.speaker:
        raise APIError(500, "No speaker engines loaded.", "server_error")
    return next(iter(e.speaker.values()))


async def voice_of(request: Request):
    """ The speaker engine, the request's audio at its rate, and the request's options. """
    from .engines.tflite import open_audio
    engine = speaker_engine(request.app.state.engines)
    audio, options = await read_audio(request)
    samples = await run_engine(lambda data: open_audio(data, engine.RATE), audio, "Speaker", "audio")
    return engine, samples, options


@router.post("/v1/audio/speakers")
async def enroll_speaker(request: Request):
    """
    Enrolls a voice by name, from a recording of it speaking alone (name,
    a form field, and file). Each recording is kept; a few, in different
    rooms or moods, recognize the voice better.
    """
    from .speakers import Voices
    engine, samples, options = await voice_of(request)
    name = str(options.get("name") or "").strip()
    if not name or len(name) > 64:
        raise APIError(400, "name must say whose voice it is, in up to 64 characters.")
    if len(samples) < MIN_VOICE * engine.RATE:
        raise APIError(400, f"The recording must be at least {MIN_VOICE:g} s of speech.")
    may_change(request, Voices().owner(name), name)
    vector = await run_engine(engine.embed, samples, "Speaker", "audio")
    recordings = Voices().add(name, vector, request.user.name)
    return {"object": "speaker", "name": name, "owner": Voices().owner(name), "recordings": recordings}


def may_change(request: Request, owner: Optional[str], name: str):
    """ Refuses to change a voice someone else enrolled, unless the user is an admin. """
    if owner is not None and owner != request.user.name and not request.user.admin:
        raise APIError(403, f"The voice {name!r} is someone else's; only they or an admin may change it.",
                       code="not_your_speaker")


@router.get("/v1/audio/speakers")
def list_speakers():
    from .speakers import Voices
    return {"object": "list", "data": [{"name": n, **v} for n, v in Voices().names().items()]}


@router.delete("/v1/audio/speakers/{name}")
def delete_speaker(request: Request, name: str):
    from .speakers import Voices
    voices = Voices()
    owner = voices.owner(name)
    if owner is None:
        raise APIError(404, f"No voice named {name!r} is enrolled.", code="speaker_not_found")
    may_change(request, owner, name)
    voices.remove(name)
    return {"object": "speaker", "name": name, "deleted": True}


@router.post("/v1/audio/speakers/identify")
async def identify_speaker(request: Request):
    """ Which enrolled voice one voice's recording is, such as a spoken command: speaker is null if none. """
    from .speakers import Voices
    engine, samples, _ = await voice_of(request)
    if len(samples) < MIN_VOICE * engine.RATE:
        raise APIError(400, f"The recording must be at least {MIN_VOICE:g} s of speech.")
    vector = await run_engine(engine.embed, samples, "Speaker", "audio")
    name, distance = Voices().match(vector, engine.SAME)
    return {"object": "speaker.identification", "model": engine.get_model_id(), "speaker": name,
            "distance": round(distance, 3)}


@router.post("/v1/audio/diarization")
async def diarization(request: Request):
    """
    Who spoke when in a recording: each turn's start and end, in seconds,
    and its speaker, an enrolled name or "speaker 1", "speaker 2" and so on,
    with how long each spoke.
    """
    from .speakers import Voices, diarize
    engine, samples, _ = await voice_of(request)
    found = await run_engine(lambda s: diarize(engine, s, Voices()), samples, "Diarization", "audio")
    return {"object": "audio.diarization", "model": engine.get_model_id(), **found}


@router.post("/v1/audio/pitch")
async def pitch(request: Request):
    """
    The note in audio, for tuning an instrument or a voice: not an OpenAI
    endpoint. The audio comes as read_audio takes it, up to 10 seconds.
    Options: instrument, such as guitar or violin, for the nearest string
    and how far off it; reference, A4's frequency (default 440); and
    confidence, how sure the engine must be that a note is playing (0 to
    1, default 0.5).
    """
    from .pitch import CONFIDENCE, TUNINGS, advice, tune
    e = request.app.state.engines
    if not e.pitch:
        raise APIError(500, "No pitch engines loaded.", "server_error")
    audio, options = await read_audio(request)
    engine = e.pitch.get(options.get("model")) or next(iter(e.pitch.values()))
    instrument = options.get("instrument") or None
    if instrument and instrument not in TUNINGS:
        raise APIError(400, f"instrument must be one of {', '.join(TUNINGS)}.")
    try:
        reference = float(options.get("reference", 440))
        confidence = float(options.get("confidence", CONFIDENCE))
    except (TypeError, ValueError):
        raise APIError(400, "reference and confidence must be numbers.")
    if not 400 <= reference <= 480 or not 0 < confidence <= 1:
        raise APIError(400, "reference must be from 400 to 480 Hz, and confidence above 0 and at most 1.")

    def hear(data: bytes):
        from .engines.tflite import open_audio
        samples = open_audio(data, engine.RATE)
        if len(samples) > MAX_PITCH * engine.RATE:
            raise ValueError(f"the tuner takes up to {MAX_PITCH} seconds; this is {len(samples) / engine.RATE:.0f}")
        return tune(engine.pitches(samples), engine.HOP, reference, instrument, confidence)
    found = await run_engine(hear, audio, "Pitch", "audio")
    return {"object": "audio.pitch", "model": engine.get_model_id(), "advice": advice(found), **found}
