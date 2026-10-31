"""
Engine registry discovery. Dynamically finds and imports every kind of engine
in this package: TTS, STT, LLM, embedding, depth, detection, sound, text
classification, image editing, image quality, OCR, segmentation, faces, poses,
image classification, noise removal, speakers, lines, actions, pitch and
reranking.
"""
import importlib
import inspect
import pkgutil
from typing import Dict, Sequence

from .tts_base import TTSEngine
from .stt_base import STTEngine
from .llm_base import LLMEngine, Thought, ToolCall
from .embedding_base import EmbeddingEngine
from .depth_base import DepthEngine, DepthMap
from .detection_base import Detection, DetectionEngine
from .sound_base import SoundEngine, Sounds
from .classifier_base import ClassifierEngine
from .image_edit_base import ImageEditEngine
from .quality_base import QualityEngine
from .ocr_base import OCREngine, TextLine, as_text
from .segment_base import SegmentEngine
from .face_base import FaceEngine
from .pose_base import PoseEngine
from .image_classifier_base import ImageClassifierEngine
from .denoise_base import DenoiseEngine
from .speaker_base import SpeakerEngine
from .line_base import LineEngine
from .action_base import ActionEngine
from .pitch_base import PitchEngine
from .rerank_base import RerankEngine
from .lazy import LazyEngine

# Each kind of engine's base class, by the name its registry goes by
KINDS = {"tts": TTSEngine, "stt": STTEngine, "llm": LLMEngine, "embed": EmbeddingEngine, "depth": DepthEngine,
         "detect": DetectionEngine, "sound": SoundEngine, "classify": ClassifierEngine,
         "edit": ImageEditEngine, "quality": QualityEngine, "ocr": OCREngine,
         "segment": SegmentEngine, "face": FaceEngine, "pose": PoseEngine,
         "recognize": ImageClassifierEngine, "denoise": DenoiseEngine,
         "speaker": SpeakerEngine, "line": LineEngine, "action": ActionEngine, "pitch": PitchEngine,
         "rerank": RerankEngine}
# The kinds loaded at startup and kept, since a conversation uses them at once;
# the rest load on first use and unload when idle, as does any engine whose class has LAZY = True
EAGER = {"tts", "stt", "llm", "depth"}


def engine_classes() -> Dict[str, Dict[str, type]]:
    """
    Finds every engine class in this package's *_engine modules: for each
    kind in KINDS, a dictionary of classes keyed by engine id.
    """
    found = {kind: {} for kind in KINDS}
    for module_info in pkgutil.iter_modules(__path__):
        if not module_info.name.endswith("_engine"):
            continue
        module = importlib.import_module(f".{module_info.name}", package=__name__)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # A class whose name starts with _ is what a module's engines share, not an engine
            if name.startswith("_"):
                continue
            for kind, base in KINDS.items():
                if issubclass(obj, base) and obj is not base:
                    found[kind][name.lower().replace("engine", "")] = obj
    return found


def load_engines(lazy: bool = True, skip: Sequence[str] = ()) -> Dict[str, Dict[str, object]]:
    """
    Discovers every engine in this package: for each kind in KINDS, a
    dictionary of engines keyed by engine id. Those of the EAGER kinds are
    initialized now, unless their class says LAZY, and with lazy, the rest
    when first used. The kinds in skip are left empty, not loaded.
    """
    return {kind: {engine_id: LazyEngine(cls) if lazy and getattr(cls, "LAZY", kind not in EAGER) else cls()
                   for engine_id, cls in classes.items() if kind not in skip}
            for kind, classes in engine_classes().items()}

__all__ = ["TTSEngine", "STTEngine", "LLMEngine", "Thought", "ToolCall", "EmbeddingEngine", "DepthEngine", "DepthMap",
           "Detection", "DetectionEngine", "SoundEngine", "Sounds", "ClassifierEngine", "ImageEditEngine", "QualityEngine", "OCREngine", "TextLine", "as_text", "SegmentEngine", "FaceEngine", "PoseEngine", "ImageClassifierEngine", "DenoiseEngine", "SpeakerEngine", "LineEngine", "ActionEngine", "PitchEngine", "LazyEngine", "KINDS", "EAGER", "engine_classes", "load_engines"]
