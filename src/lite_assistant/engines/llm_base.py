"""
Base interfaces and definitions for Large Language Model (LLM) engines.
"""
import dataclasses
import uuid
from typing import Iterator, Dict, Any, List, Optional, Union


@dataclasses.dataclass
class ToolCall:
    """ A function call the model made, which generate yields in place of text. """
    name: str
    # The arguments as a JSON object string, as OpenAI's API carries them
    arguments: str
    id: str = dataclasses.field(default_factory=lambda: f"call_{uuid.uuid4().hex[:24]}")


@dataclasses.dataclass
class Thought:
    """ Reasoning a model writes before its reply, which generate yields as it's written. """
    text: str


class LLMEngine:
    """
    Abstract base class for all LLM chat engines.
    """

    def get_model_id(self) -> str:
        """
        Returns the ID of the model this engine has loaded.
        """
        raise NotImplementedError

    def get_modalities(self) -> List[str]:
        """ The inputs the loaded model takes: "text", and "image" and "audio" if it can see and hear. """
        return ["text", "image", "audio"]

    def get_models(self) -> Dict[str, Dict[str, Any]]:
        """
        The models this engine can switch between, when it can load more than
        one, by id, each with what /v1/models says about it beyond its id,
        such as its "modalities". Only one is loaded at a time.
        """
        return {}

    def use(self, model: Optional[str]) -> Optional["LLMEngine"]:
        """
        An engine that generates with the model named, one of get_models() or
        an alias, switching this engine to it first if another is loaded.
        None if the name isn't one of this engine's models. Raises LookupError
        for a model this engine knows but can't load, such as one not downloaded.
        """
        return None

    def generate(self, messages: List[Dict[str, Any]], max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None, top_p: Optional[float] = None,
                 usage: Optional[Dict[str, int]] = None,
                 tools: Optional[List[Dict[str, Any]]] = None,
                 thinking: Union[bool, int, None] = None) -> Iterator[Union[str, ToolCall, Thought]]:
        """
        Generates a reply to OpenAI-style chat messages, yielding text chunks as
        they are decoded. If a usage dict is passed, it is filled with
        prompt_tokens and completion_tokens once generation finishes.

        tools are chat completions function definitions ({"type": "function",
        "function": {...}}), passed only when a request has some. The model may
        then yield ToolCalls; the messages that follow carry them back as an
        assistant message's tool_calls and "tool" messages with the results.

        thinking turns reasoning on (True, or a token budget for it) or off
        (False); None leaves it as the model has it. A model that reasons
        yields Thoughts before its text. Callers pass it only when it's set.

        response_schema, a JSON Schema, holds the reply to JSON that fits
        it. Callers pass it only when a request sets one, and never with tools.
        """
        raise NotImplementedError

    @classmethod
    def setup(cls):
        """
        Performs first-time setup, such as downloading default models.
        """
        pass
