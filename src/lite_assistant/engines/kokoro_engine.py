"""
Kokoro TTS engine implementation plugin.
"""
import os
import numpy as np
import urllib.request
from typing import Iterator, Dict, Any, List
from .tts_base import TTSEngine
from ..paths import data_dir

# Kokoro's model files are GitHub release assets, so they live in the data
# directory rather than the Hugging Face cache
RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/"
FILES = ("kokoro-v1.0.onnx", "voices-v1.0.bin")
# A voice's first letter is its language: the espeak language that phonemizes
# its text, and the language's name. af_heart is American English, ff_siwis French.
LANGUAGES = {"a": ("en-us", "American English"), "b": ("en-gb", "British English"), "e": ("es", "Spanish"),
             "f": ("fr-fr", "French"), "h": ("hi", "Hindi"), "i": ("it", "Italian"), "j": ("ja", "Japanese"),
             "p": ("pt-br", "Brazilian Portuguese"), "z": ("cmn", "Mandarin Chinese")}


def voice_language(voice: str) -> str:
    """ The espeak language of a Kokoro voice, American English if it doesn't say. """
    return LANGUAGES.get(voice[:1], LANGUAGES["a"])[0]


class KokoroEngine(TTSEngine):
    """
    Kokoro TTS Engine Implementation.
    High-quality, on-device AI voice. Yields float32 which is downsampled to 16-bit PCM.
    """
    def __init__(self, models_dir=None):
        self.models_dir = models_dir or data_dir("kokoro")
        self.kokoro = None
        self.default_voice = "af_heart" # standard fallback for Kokoro
        self._initialize_kokoro()

    def _initialize_kokoro(self):
        try:
            from kokoro_onnx import Kokoro
            onnx_path, voices_path = (os.path.join(self.models_dir, name) for name in FILES)
            
            if os.path.exists(onnx_path) and os.path.exists(voices_path):
                print(f"[Kokoro] Loading Kokoro ONNX model into memory...")
                self.kokoro = Kokoro(onnx_path, voices_path)
                print(f"[Kokoro] Model loaded and ready.")
            else:
                print(f"[Kokoro] Models not found in {self.models_dir}; run 'lite-assistant setup'. "
                      "KokoroEngine will be disabled.")
        except ImportError:
            print("[Kokoro] kokoro-onnx package is not installed. Skipping KokoroEngine.")

    def get_available_voices(self) -> List[Dict[str, Any]]:
        if not self.kokoro:
            return []
            
        available = []
        voices = self.kokoro.get_voices()
        for voice_id in voices:
            # Kokoro's voice names start with their language and gender: af_heart is American English, female
            language = LANGUAGES.get(voice_id[:1], (None, "Unknown"))[1]
            gender = "Female" if voice_id[1:2] == "f" else ("Male" if voice_id[1:2] == "m" else "Unknown")
            name = voice_id.split("_", 1)[-1].capitalize() if "_" in voice_id else voice_id.capitalize()

            display_name = f"{name} ({language} {gender})"
            available.append({"id": voice_id, "name": display_name})

        return sorted(available, key=lambda x: x["name"])

    def get_default_voice(self) -> str:
        return self.default_voice

    def generate_pcm(self, text: str, voice: str) -> Iterator[bytes]:
        if not self.kokoro:
            yield b""
            return
            
        import audioop
        # Kokoro generates audio natively at 24kHz float32
        # Without its language, a voice's text is phonemized as English
        audio, sample_rate = self.kokoro.create(text, voice=voice, speed=1.0, lang=voice_language(voice))
        
        # Convert float32 [-1.0, 1.0] to int16
        audio_int16 = (audio * 32767).astype(np.int16).tobytes()
        
        state = None
        # In case the sample rate isn't 24000 (though Kokoro is natively 24000)
        if sample_rate != 24000:
            audio_int16, state = audioop.ratecv(
                audio_int16, 
                2, # 16-bit 
                1, # Mono
                sample_rate, 
                24000, 
                state
            )
            
        # Yield the generated PCM
        yield audio_int16

    @classmethod
    def setup(cls):
        models_dir = data_dir("kokoro")
        os.makedirs(models_dir, exist_ok=True)
        for name in FILES:
            dest = os.path.join(models_dir, name)
            if not os.path.exists(dest):
                print(f"[Kokoro] Downloading {dest} from {RELEASE + name}")
                # Download beside the destination, then rename, so an
                # interrupted download never looks complete
                partial = dest + ".partial"
                urllib.request.urlretrieve(RELEASE + name, partial)
                os.replace(partial, dest)
