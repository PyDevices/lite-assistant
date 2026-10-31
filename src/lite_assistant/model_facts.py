"""
What chat model files don't say, or say wrongly, about the models in them:
whether a model thinks, whether it calls tools, and the sampling its makers
recommend. Gemma 4's files say it does neither of the first two, though it
does both, and Qwen3.5-4B's leave its sampling at zero.

The facts are found by a model id's name; the first pattern that matches it
wins, and what no pattern says falls back to what the file says.
"""
import re
from typing import Any, Dict, List, Tuple

# (pattern on the model id, facts): "thinking" and "tools" are whether it
# does each, and "sampler" its makers' recommended top_k, top_p and temperature
KNOWN: List[Tuple[str, Dict[str, Any]]] = [
    # Google's model card: temperature 1.0, top_p 0.95, top_k 64
    (r"^gemma-?4", {"thinking": True, "tools": True,
                    "sampler": {"top_k": 64, "top_p": 0.95, "temperature": 1.0}}),
    (r"^gemma-?3", {"thinking": False, "sampler": {"top_k": 64, "top_p": 0.95, "temperature": 1.0}}),
    # Qwen's card for thinking mode, general tasks
    (r"^qwen3\.5", {"thinking": True, "tools": True,
                    "sampler": {"top_k": 20, "top_p": 0.95, "temperature": 1.0}}),
    (r"^qwen3(?![.\d])", {"thinking": True, "tools": True,
                          "sampler": {"top_k": 20, "top_p": 0.95, "temperature": 0.6}}),
    (r"^qwen2\.5", {"thinking": False, "sampler": {"top_k": 20, "top_p": 0.8, "temperature": 0.7}}),
    # It always thinks, and DeepSeek recommends 0.6
    (r"deepseek-r1", {"thinking": True, "sampler": {"top_p": 0.95, "temperature": 0.6}}),
    (r"^phi-4-mini", {"thinking": False}),
]
# Gemma's recommended sampling, for a model nothing here or in its file covers
FALLBACK_SAMPLER = {"top_k": 64, "top_p": 0.95, "temperature": 1.0}


def known(model_id: str) -> Dict[str, Any]:
    """ The facts this list has about a model, by its id; empty if none. """
    for pattern, facts in KNOWN:
        if re.search(pattern, model_id, re.IGNORECASE):
            return facts
    return {}


def sampler(model_id: str, stored: Dict[str, Any]) -> Dict[str, Any]:
    """
    The sampling a model gets when a request asks for sampling but sets only
    some of it: what the file stores, else what this list says, else Gemma's.
    Unset fields come back from a file as None or 0.
    """
    listed = known(model_id).get("sampler", {})
    return {key: stored.get(key) or listed.get(key) or FALLBACK_SAMPLER[key] for key in FALLBACK_SAMPLER}


def abilities(model_id: str, info: Dict[str, Any]) -> Dict[str, bool]:
    """
    Whether a model thinks and calls tools: this list's word, else its file's
    (info's "thinking" and "tools"), which errs toward false.
    """
    listed = known(model_id)
    return {key: bool(listed.get(key, info.get(key, False))) for key in ("thinking", "tools")}
