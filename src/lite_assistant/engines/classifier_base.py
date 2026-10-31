"""
Base interfaces and definitions for text classifiers, which sort a text into
labels the request names, such as a command's intent.
"""
from typing import List, Optional, Sequence


class ClassifierEngine:
    """
    Abstract base class for all zero-shot text classification engines.
    """

    def get_model_id(self) -> str:
        """
        Returns the ID of the model this engine has loaded.
        """
        raise NotImplementedError

    def logits(self, text: str, labels: Sequence[str], prompt: Optional[str] = None) -> List[float]:
        """
        How well each label fits the text, as one logit per label: softmax
        over them picks one label, and each one's sigmoid says whether it
        fits on its own. A prompt describes the task, such as "The intent of
        this voice command:". Raises ValueError for a request too long or
        with too many labels.
        """
        raise NotImplementedError

    @classmethod
    def setup(cls):
        """
        Performs first-time setup, such as downloading default models.
        """
        pass
